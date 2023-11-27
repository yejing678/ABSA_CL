# -*- coding: utf-8 -*-

import argparse
import logging
import math
import os
import random
import sys
import time
from time import strftime, localtime
import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import metrics
from torch.utils.data import DataLoader, random_split
from transformers import BertModel
from data_utils import build_tokenizer, build_embedding_matrix, Tokenizer4Bert, ABSADataset
from models.bert_spc import BERT_SPC
from Loss.Triplet_Loss import TripletLoss
from pathlib import Path
from tqdm import tqdm
import train_picture

sys.path.append(os.pardir)  # 为了导入父目录的文件而进行的设定

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))


def have_different_sample(targets):
    flag = 0
    target1 = targets[0]
    for i in range(len(targets)):
        a = targets[i]
        if a != target1:
            flag = 1
            break
    return flag


class Instructor:
    def __init__(self, opt):
        self.opt = opt
        print('building embedding...')
        if 'bert' in opt.model_name:
            tokenizer = Tokenizer4Bert(opt.max_seq_len, opt.pretrained_bert_name)
            bert = BertModel.from_pretrained(opt.pretrained_bert_name, return_dict=False)
            self.pretrained_bert_state_dict = bert.state_dict()
            self.model = opt.model_class(bert, opt).to(opt.device)
        else:
            tokenizer = build_tokenizer(
                fnames=[opt.dataset_file['train']],
                max_seq_len=opt.max_seq_len,
                dat_fname='{0}_tokenizer.dat'.format(opt.dataset))
            embedding_matrix = build_embedding_matrix(
                word2idx=tokenizer.word2idx,
                embed_dim=opt.embed_dim,
                dat_fname='{0}_{1}_embedding_matrix.dat'.format(str(opt.embed_dim), opt.dataset))
            self.model = opt.model_class(embedding_matrix, opt).to(opt.device)
        print('loading dataset...')
        self.trainset = ABSADataset(opt.dataset_file['train'], tokenizer)
        if opt.device.type == 'cuda':
            logger.info('cuda memory allocated: {}'.format(torch.cuda.memory_allocated(device=opt.device.index)))
        self._print_args()

    def _print_args(self):
        n_trainable_params = 0
        n_nontrainable_params = 0
        for p in self.model.parameters():
            n_params = torch.prod(torch.tensor(p.shape))
            if p.requires_grad:
                n_trainable_params += n_params
            else:
                n_nontrainable_params += n_params
        logger.info('> n_trainable_params: {0}, n_nontrainable_params: {1}'
                    .format(n_trainable_params, n_nontrainable_params))
        logger.info('> training arguments:')
        for arg in vars(self.opt):
            logger.info('>>> {0}: {1}'.format(arg, getattr(self.opt, arg)))

    def _reset_params(self):
        print('resetting parameters...')
        for child in self.model.children():
            if type(child) != BertModel:  # skip bert params
                for p in child.parameters():
                    if p.requires_grad:
                        if len(p.shape) > 1:
                            self.opt.initializer(p)
                        else:
                            stdv = 1. / math.sqrt(p.shape[0])
                            torch.nn.init.uniform_(p, a=-stdv, b=stdv)
            else:
                self.model.load_state_dict(self.pretrained_bert_state_dict, strict=False)

    def _pre_train(self, criterion1, criterion2, optimizer, train_data_loader):
        print('staring train...')
        global_step = 0
        update_num_list, train_loss_list, train_acc_list = [], [], []
        path = None
        Path(
            f"plots/pre-train/{self.opt.model_name}/{self.opt.dataset}/{self.opt.lr}/{self.opt.dropout}/{self.opt.seed}") \
            .mkdir(parents=True, exist_ok=True)
        writer = open(
            f"plots/pre-train/{self.opt.model_name}/{self.opt.dataset}/{self.opt.lr}/{self.opt.dropout}/{self.opt.seed}/logs.csv",
            "w")

        for i_epoch in tqdm(range(self.opt.num_epoch)):
            start = time.time()
            step = 0
            logger.info('>' * 120)
            logger.info('epoch: {}'.format(i_epoch))
            n_correct, n_total, loss_total = 0, 0, 0
            # switch model to training mode
            self.model.train()
            for i_batch, batch in enumerate(train_data_loader):
                global_step += 1
                step += 1
                # clear gradient accumulators
                optimizer.zero_grad()

                inputs = [batch[col].to(self.opt.device) for col in self.opt.inputs_cols]
                aspect_logic, sentence_features = self.model(inputs)
                aspect_sentiment = batch['polarity'].to(self.opt.device)

                loss1 = criterion1(aspect_logic, aspect_sentiment)
                if have_different_sample(aspect_sentiment):
                    sentence_features = F.normalize(sentence_features, p=2, dim=1)
                    loss2 = criterion2(sentence_features, aspect_sentiment)
                else:
                    loss2 = 0
                loss = self.opt.lamda * loss1 + (1 - self.opt.lamda) * loss2
                loss.backward()
                optimizer.step()

                n_correct += (torch.argmax(aspect_logic, -1) == aspect_sentiment).sum().item()
                n_total += len(aspect_logic)
                loss_total += loss.item() * len(aspect_logic)
                if step % self.opt.log_step == 0:
                    train_acc = n_correct / n_total
                    train_loss = loss_total / n_total
                    logger.info(
                        '[epoch %2d] [step %3d] train_loss: %.4f train_acc: %.4f'
                        % (i_epoch, step, train_loss, train_acc))
                    update_num_list.append(global_step)
                    train_loss_list.append(train_loss)
                    train_acc_list.append(train_acc)
                    writer.write(f"{global_step},{train_acc:.4f}\n")

                if global_step % self.opt.save_step == 0:
                    save_path = "state_dict/pre-train/{0}/{1}/{2}/".\
                        format(self.opt.model_name, self.opt.dataset, self.opt.seed)
                    dirname = os.path.dirname(save_path)
                    if not os.path.exists(dirname):
                        os.makedirs(dirname)
                    path = save_path + 'epoch{0}_step{1}'.format(i_epoch, step)
                    if i_epoch > 0:
                        torch.save(self.model.state_dict(), path)
                        logger.info('>> saved: {}'.format(path))

            end = time.time()
            logger.info('time: {:.4f}s'.format(end - start))

            # draw picture
            train_picture.plot_jasons_lineplot(update_num_list, train_loss_list, 'updates', 'training loss',
                                               f"{self.opt.model_name}  {self.opt.dataset}",
                                               f"plots/pre-train/{self.opt.model_name}/{self.opt.dataset}/{self.opt.lr}/{self.opt.dropout}/{self.opt.seed}/train_loss.png")
            train_picture.plot_jasons_lineplot(update_num_list, train_acc_list, 'updates', 'train accuracy',
                                               f"{self.opt.model_name}  {self.opt.dataset}",
                                               f"plots/pre-train/{self.opt.model_name}/{self.opt.dataset}/{self.opt.lr}/{self.opt.dropout}/{self.opt.seed}/train_acc.png")
        return path

    def run(self):
        criterion1 = nn.CrossEntropyLoss()
        criterion2 = TripletLoss()
        _params = filter(lambda p: p.requires_grad, self.model.parameters())
        optimizer = self.opt.optimizer(_params, lr=self.opt.lr, weight_decay=self.opt.l2reg)
        train_data_loader = DataLoader(dataset=self.trainset, batch_size=self.opt.batch_size, shuffle=True)
        self._reset_params()
        self._pre_train(criterion1, criterion2, optimizer, train_data_loader)

def main():
    # Hyper Parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='bert_spc', type=str)
    parser.add_argument('--dataset', default='yelp_restaurant', type=str, help='amazon_laptop, yelp_restaurant')
    parser.add_argument('--optimizer', default='adamw', type=str)
    parser.add_argument('--initializer', default='xavier_uniform_', type=str)
    parser.add_argument('--lr', default=2e-5, type=float, help='try 5e-5, 2e-5 for BERT, 1e-3 for others')
    parser.add_argument('--dropout', default=0.1, type=float)
    parser.add_argument('--l2reg', default=1e-3, type=float)
    parser.add_argument('--num_epoch', default=20, type=int, help='try larger number for non-BERT models')
    parser.add_argument('--batch_size', default=64, type=int, help='try 16, 32, 64 for BERT models')
    parser.add_argument('--log_step', default=10, type=int)
    parser.add_argument('--save_step', default=1000, type=int)
    parser.add_argument('--embed_dim', default=300, type=int)
    parser.add_argument('--hidden_dim', default=300, type=int)
    parser.add_argument('--bert_dim', default=768, type=int)
    parser.add_argument('--pretrained_bert_name', default="bert-base-uncased", type=str)
    parser.add_argument('--max_seq_len', default=85, type=int)
    parser.add_argument('--polarities_dim', default=3, type=int)
    parser.add_argument('--device', default='cuda:9', type=str, help='e.g. cuda:0')
    parser.add_argument('--seed', default=42, type=int, help='set seed for reproducibility')
    parser.add_argument('--lamda', default=0.5, type=int)
    opt = parser.parse_args()

    if opt.seed is not None:
        random.seed(opt.seed)
        numpy.random.seed(opt.seed)
        torch.manual_seed(opt.seed)
        torch.cuda.manual_seed(opt.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ['PYTHONHASHSEED'] = str(opt.seed)

    model_classes = {
        'bert_spc': BERT_SPC,
    }
    dataset_files = {
        'amazon_laptop': {
            'train': "/home/jye/ABSA/datasets/amazon/amazon_laptops.txt",
        },
        'yelp_restaurant': {
            'train': "/home/jye/ABSA/datasets/yelp/yelp_restaurants.txt",
        },
    }
    input_colses = {

        'bert_spc': ['concat_bert_indices', 'concat_segments_indices'],
    }
    initializers = {
        'xavier_uniform_': torch.nn.init.xavier_uniform_,
        'xavier_normal_': torch.nn.init.xavier_normal_,
        'orthogonal_': torch.nn.init.orthogonal_,
    }
    optimizers = {
        'adadelta': torch.optim.Adadelta,  # default lr=1.0
        'adagrad': torch.optim.Adagrad,  # default lr=0.01
        'adam': torch.optim.Adam,  # default lr=0.001
        'adamax': torch.optim.Adamax,  # default lr=0.002
        'asgd': torch.optim.ASGD,  # default lr=0.01
        'rmsprop': torch.optim.RMSprop,  # default lr=0.01
        'sgd': torch.optim.SGD,
        'adamw': torch.optim.AdamW,
    }
    opt.model_class = model_classes[opt.model_name]
    opt.dataset_file = dataset_files[opt.dataset]
    opt.inputs_cols = input_colses[opt.model_name]
    opt.initializer = initializers[opt.initializer]
    opt.optimizer = optimizers[opt.optimizer]
    opt.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') \
        if opt.device is None else torch.device(opt.device)

    log_file = '{0}-{1}-{2}-{3}-{4}-{5}-{6}-pre-train-{7}.log'.format(opt.model_name, opt.dataset, opt.lr,
                                                                      opt.l2reg, opt.batch_size, opt.dropout, opt.seed,
                                                                      strftime("%y%m%d-%H%M", localtime()))
    logger.addHandler(logging.FileHandler(log_file))

    ins = Instructor(opt)
    ins.run()


if __name__ == '__main__':
    main()
