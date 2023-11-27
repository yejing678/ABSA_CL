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
from transformers import RobertaTokenizer, RobertaModel
from data_prepare import build_tokenizer, build_embedding_matrix, Tokenizer4Roberta, ABSADataset
from robert_spc import Roberta_SPC
from roberta_lcf import LCF_Roberta
from roberta_attention import LCF_ROBERTA_Attention
from models.SupConLoss import SupConLoss
from Loss.Triplet_Loss import TripletLoss
from pathlib import Path
from tqdm import tqdm
import train_picture

sys.path.append(os.pardir)  # 为了导入父目录的文件而进行的设定

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))


def has_opposite_labels(labels):
    return not (labels.sum().item() <= 1 or (1 - labels).sum().item() <= 1)


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
        if 'roberta' in opt.model_name:
            tokenizer = Tokenizer4Roberta(opt.max_seq_len)
            bert = RobertaModel.from_pretrained(opt.pretrained_bert_name)
            self.pretrained_bert_state_dict = bert.state_dict()
            self.model = opt.model_class(bert, opt).to(opt.device)
        else:
            tokenizer = build_tokenizer(
                fnames=[opt.dataset_file['train'], opt.dataset_file['test']],
                max_seq_len=opt.max_seq_len,
                dat_fname='{0}_tokenizer.dat'.format(opt.dataset))
            embedding_matrix = build_embedding_matrix(
                word2idx=tokenizer.word2idx,
                embed_dim=opt.embed_dim,
                dat_fname='{0}_{1}_embedding_matrix.dat'.format(str(opt.embed_dim), opt.dataset))
            self.model = opt.model_class(embedding_matrix, opt).to(opt.device)
        print('loading dataset...')
        self.trainset = ABSADataset(opt.dataset_file['train'], tokenizer)
        self.testset = ABSADataset(opt.dataset_file['test'], tokenizer)
        # self.valset = ABSADataset(opt.dataset_file['val'], tokenizer)
        assert 0 <= opt.valset_ratio < 1
        if opt.valset_ratio > 0:
            valset_len = int(len(self.trainset) * opt.valset_ratio)
            self.trainset, self.valset = random_split(self.trainset, (len(self.trainset) - valset_len, valset_len))
        else:
            self.valset = self.testset

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
        logger.info(
            '> n_trainable_params: {0}, n_nontrainable_params: {1}'.format(n_trainable_params, n_nontrainable_params))
        logger.info('> training arguments:')
        for arg in vars(self.opt):
            logger.info('>>> {0}: {1}'.format(arg, getattr(self.opt, arg)))

    def _reset_params(self):
        print('resetting parameters...')
        for child in self.model.children():
            if type(child) != RobertaModel:  # skip bert params
                for p in child.parameters():
                    if p.requires_grad:
                        if len(p.shape) > 1:
                            self.opt.initializer(p)
                        else:
                            stdv = 1. / math.sqrt(p.shape[0])
                            torch.nn.init.uniform_(p, a=-stdv, b=stdv)
            else:
                self.model.load_state_dict(self.pretrained_bert_state_dict, strict=False)

    def _train(self, CE, TL, optimizer, train_data_loader, val_data_loader):
        print('staring train...')
        max_val_acc, max_val_f1, max_val_epoch = 0, 0, 0
        global_step = 0
        update_num_list, train_loss_list, train_acc_list, val_acc_list = [], [], [], []
        path = None
        Path(
            f"plots/triplet loss/{self.opt.model_name}/{self.opt.dataset}/{self.opt.lamda}/{self.opt.seed}") \
            .mkdir(parents=True, exist_ok=True)
        writer = open(
            f"plots/triplet loss/{self.opt.model_name}/{self.opt.dataset}/{self.opt.lamda}/{self.opt.seed}/logs.csv",
            "w")

        for i_epoch in tqdm(range(self.opt.num_epoch)):
            logger.info(">" * 100)
            start = time.time()
            step = 0
            n_correct, n_total, loss_total = 0, 0, 0
            # switch model to training mode
            self.model.train()
            for i_batch, batch in enumerate(train_data_loader):
                global_step += 1
                step += 1
                # clear gradient accumulators
                optimizer.zero_grad()

                inputs = [batch[col].to(self.opt.device) for col in self.opt.inputs_cols]
                outputs, outputs_feature = self.model(inputs)
                aspect_sentiments = batch['polarity'].to(self.opt.device)
                # implicit_labels = batch['implicit_label'].to(self.opt.device)

                loss1 = CE(outputs, aspect_sentiments)
                if have_different_sample(aspect_sentiments):
                    sentence_features = F.normalize(outputs_feature, p=2, dim=1)
                    loss2 = TL(sentence_features, aspect_sentiments)
                    # print(loss2)
                else:
                    loss2 = 0

                loss = loss1 + self.opt.lamda * loss2
                # loss = self.opt.lamda * loss1 + (1-self.opt.lamda) * loss2

                loss.backward()
                optimizer.step()

                n_correct += (torch.argmax(outputs, -1) == aspect_sentiments).sum().item()
                n_total += len(outputs)
                loss_total += loss.item() * len(outputs)
                if step % self.opt.log_step == 0:
                    train_acc = n_correct / n_total
                    train_loss = loss_total / n_total
                    val_acc, val_f1 = self._val_evaluate_acc_f1(val_data_loader)
                    logger.info(
                        '[epoch %2d] [step %3d] train_loss: %.4f train_acc: %.4f val_acc: %.4f val_f1: %.4f'
                        % (i_epoch, step, train_loss, train_acc, val_acc, val_f1))

                    update_num_list.append(global_step)
                    val_acc_list.append(val_acc)
                    train_loss_list.append(train_loss)
                    train_acc_list.append(train_acc)
                    writer.write(f"{global_step},{val_acc:.4f},{train_acc:.4f}\n")

                    if val_acc > max_val_acc:
                        max_val_acc = val_acc
                        max_val_epoch = i_epoch
                        save_path = "state_dict/triplet loss train/{0}/{1}/{2}/{3}/{4}/" \
                            .format(self.opt.model_name, self.opt.dataset, self.opt.local_context_focus, self.opt.seed,
                                    self.opt.lamda)
                        dirname = os.path.dirname(save_path)
                        if not os.path.exists(dirname):
                            os.makedirs(dirname)

                        path = save_path + 'epoch{0}_step{1}_acc_{2}_f1{3}' \
                            .format(i_epoch, step, round(val_acc, 4), round(val_f1, 4))
                        if i_epoch > 0:
                            # timestamp = datetime.datetime.now().strftime('%Y-%m-%d%H%M%S')
                            torch.save(self.model.state_dict(), path)
                            logger.info('>> saved: {}'.format(path))
                    if val_f1 > max_val_f1:
                        max_val_f1 = val_f1

            end = time.time()
            logger.info('time: {:.4f}s'.format(end - start))
            if i_epoch - max_val_epoch >= self.opt.patience:
                logger.info('>> early stop.')
                break

            # draw picture
            train_picture.plot_jasons_lineplot(update_num_list, train_loss_list, 'updates', 'training loss',
                                               f"{self.opt.model_name}  {self.opt.dataset}  max val acc={max(val_acc_list):.3f}",
                                               f"plots/triplet loss/{self.opt.model_name}/{self.opt.dataset}/{self.opt.lamda}/{self.opt.seed}/train_loss.png")
            train_picture.plot_jasons_lineplot(update_num_list, val_acc_list, 'updates', 'validation accuracy',
                                               f"{self.opt.model_name}  {self.opt.dataset}  max val acc={max(val_acc_list):.3f}",
                                               f"plots/triplet loss/{self.opt.model_name}/{self.opt.dataset}/{self.opt.lamda}/{self.opt.seed}/val_acc{max(val_acc_list):.3f}.png")
            train_picture.plot_jasons_lineplot(update_num_list, train_acc_list, 'updates', 'train accuracy',
                                               f"{self.opt.model_name}  {self.opt.dataset}  max val acc={max(val_acc_list):.3f}",
                                               f"plots/triplet loss/{self.opt.model_name}/{self.opt.dataset}/{self.opt.lamda}/{self.opt.seed}/train_acc.png")

        logger.info('>> max_val_acc: {:.4f}, max_val_f1: {:.4f}'.format(max_val_acc, max_val_f1))
        return path

    def _train_only_CE(self, CE, TL, optimizer, train_data_loader, val_data_loader):
        print('staring train...')
        max_val_acc, max_val_f1, max_val_epoch = 0, 0, 0
        global_step = 0
        update_num_list, train_loss_list, train_acc_list, val_acc_list = [], [], [], []
        path = None
        Path(
            f"plots/triplet_loss_train/{self.opt.model_name}/{self.opt.dataset}/{self.opt.lamda}/{self.opt.seed}") \
            .mkdir(parents=True, exist_ok=True)
        writer = open(
            f"plots/triplet_loss_train/{self.opt.model_name}/{self.opt.dataset}/{self.opt.lamda}/{self.opt.seed}/logs.csv",
            "w")

        for i_epoch in tqdm(range(self.opt.num_epoch)):
            logger.info(">" * 100)
            start = time.time()
            step = 0
            n_correct, n_total, loss_total = 0, 0, 0
            # switch model to training mode
            self.model.train()
            for i_batch, batch in enumerate(train_data_loader):
                global_step += 1
                step += 1
                # clear gradient accumulators
                optimizer.zero_grad()

                inputs = [batch[col].to(self.opt.device) for col in self.opt.inputs_cols]
                outputs, outputs_feature = self.model(inputs)
                aspect_sentiments = batch['polarity'].to(self.opt.device)
                # implicit_labels = batch['implicit_label'].to(self.opt.device)

                loss1 = CE(outputs, aspect_sentiments)
                if have_different_sample(aspect_sentiments):
                    sentence_features = F.normalize(outputs_feature, p=2, dim=1)
                    loss2 = TL(sentence_features, aspect_sentiments)
                    # print(loss2)
                else:
                    loss2 = 0
                loss = self.opt.lamda * loss1 + (1 - self.opt.lamda) * loss2

                loss.backward()
                optimizer.step()

                n_correct += (torch.argmax(outputs, -1) == aspect_sentiments).sum().item()
                n_total += len(outputs)
                loss_total += loss.item() * len(outputs)
                if step % self.opt.log_step == 0:
                    train_acc = n_correct / n_total
                    train_loss = loss_total / n_total
                    val_acc, val_f1 = self._val_evaluate_acc_f1(val_data_loader)
                    logger.info(
                        '[epoch %2d] [step %3d] train_loss: %.4f train_acc: %.4f val_acc: %.4f val_f1: %.4f'
                        % (i_epoch, step, train_loss, train_acc, val_acc, val_f1))

                    update_num_list.append(global_step)
                    val_acc_list.append(val_acc)
                    train_loss_list.append(train_loss)
                    train_acc_list.append(train_acc)
                    writer.write(f"{global_step},{val_acc:.4f},{train_acc:.4f}\n")

                    if val_acc > max_val_acc:
                        max_val_acc = val_acc
                        max_val_epoch = i_epoch
                        save_path = "state_dict/train/{0}/{1}/{2}/{3}/{4}/" \
                            .format(self.opt.model_name, self.opt.dataset, self.opt.local_context_focus, self.opt.seed,
                                    self.opt.lamda)
                        dirname = os.path.dirname(save_path)
                        if not os.path.exists(dirname):
                            os.makedirs(dirname)

                        path = save_path + 'epoch{0}_step{1}_acc_{2}_f1{3}' \
                            .format(i_epoch, step, round(val_acc, 4), round(val_f1, 4))
                        if i_epoch > 0:
                            # timestamp = datetime.datetime.now().strftime('%Y-%m-%d%H%M%S')
                            torch.save(self.model.state_dict(), path)
                            logger.info('>> saved: {}'.format(path))
                    if val_f1 > max_val_f1:
                        max_val_f1 = val_f1

            end = time.time()
            logger.info('time: {:.4f}s'.format(end - start))
            if i_epoch - max_val_epoch >= self.opt.patience:
                logger.info('>> early stop.')
                break

            # draw picture
            train_picture.plot_jasons_lineplot(update_num_list, train_loss_list, 'updates', 'training loss',
                                               f"{self.opt.model_name}  {self.opt.dataset}  max val acc={max(val_acc_list):.3f}",
                                               f"plots/triplet_loss_train/{self.opt.model_name}/{self.opt.dataset}/{self.opt.lamda}/{self.opt.seed}/train_loss.png")
            train_picture.plot_jasons_lineplot(update_num_list, val_acc_list, 'updates', 'validation accuracy',
                                               f"{self.opt.model_name}  {self.opt.dataset}  max val acc={max(val_acc_list):.3f}",
                                               f"plots/triplet_loss_train/{self.opt.model_name}/{self.opt.dataset}/{self.opt.lamda}/{self.opt.seed}/val_acc{max(val_acc_list):.3f}.png")
            train_picture.plot_jasons_lineplot(update_num_list, train_acc_list, 'updates', 'train accuracy',
                                               f"{self.opt.model_name}  {self.opt.dataset}  max val acc={max(val_acc_list):.3f}",
                                               f"plots/triplet_loss_train/{self.opt.model_name}/{self.opt.dataset}/{self.opt.lamda}/{self.opt.seed}/train_acc.png")

        logger.info('>> max_val_acc: {:.4f}, max_val_f1: {:.4f}'.format(max_val_acc, max_val_f1))
        return path

    def _val_evaluate_acc_f1(self, data_loader):
        n_correct, n_total = 0, 0
        t_targets_all, t_outputs_all = None, None
        with torch.no_grad():
            for i_batch, t_batch in enumerate(data_loader):
                t_inputs = [t_batch[col].to(self.opt.device) for col in self.opt.inputs_cols]
                t_targets = t_batch['polarity'].to(self.opt.device)
                t_outputs, _ = self.model(t_inputs)
                n_correct += (torch.argmax(t_outputs, -1) == t_targets).sum().item()
                n_total += len(t_outputs)

                if t_targets_all is None:
                    t_targets_all = t_targets
                    t_outputs_all = t_outputs
                else:
                    t_targets_all = torch.cat((t_targets_all, t_targets), dim=0)
                    t_outputs_all = torch.cat((t_outputs_all, t_outputs), dim=0)

        acc = n_correct / n_total
        f1 = metrics.f1_score(t_targets_all.cpu(), torch.argmax(t_outputs_all, -1).cpu(), labels=[0, 1, 2],
                              average='macro')
        return acc, f1

    def _test_evaluate_acc_f1(self, data_loader):
        n_correct, n_total = 0, 0
        t_targets_all, t_outputs_all = None, None
        predict_wrong = 0
        Neu2Neg, Neu2Pos, Neg2Pos, Neg2Neu, Pos2Neu, Pos2Neg = 0, 0, 0, 0, 0, 0
        # switch model to evaluation mode
        self.model.eval()
        with torch.no_grad():
            for i_batch, t_batch in enumerate(data_loader):
                t_inputs = [t_batch[col].to(self.opt.device) for col in self.opt.inputs_cols]
                t_targets = t_batch['polarity'].to(self.opt.device)
                t_outputs, _ = self.model(t_inputs)

                n_predict = torch.argmax(t_outputs, -1)
                n_correct += (n_predict == t_targets).sum().item()
                n_total += len(t_outputs)
                for i in range(len(t_batch['text'])):
                    if not n_predict[i] == t_targets[i]:
                        predict_wrong += 1
                        logger.info('============================Prediction Error=============================')
                        logger.info(
                            '{0} \n>>>aspect:{1}\n>>>ground_truth: {2}\n>>>predict：{3}'.format(
                                t_batch['text'][i], t_batch['aspect'][i], t_batch['polarity'][i] - 1, n_predict[i] - 1))
                        if t_targets[i] == 1 and n_predict[i] == 0:
                            Neu2Neg += 1
                        elif t_targets[i] == 1 and n_predict[i] == 2:
                            Neu2Pos += 1
                        elif t_targets[i] == 0 and n_predict[i] == 2:
                            Neg2Pos += 1
                        elif t_targets[i] == 0 and n_predict[i] == 1:
                            Neg2Neu += 1
                        elif t_targets[i] == 2 and n_predict[i] == 1:
                            Pos2Neu += 1
                        elif t_targets[i] == 2 and n_predict[i] == 0:
                            Pos2Neg += 1

                if t_targets_all is None:
                    t_targets_all = t_targets
                    t_outputs_all = t_outputs
                else:
                    t_targets_all = torch.cat((t_targets_all, t_targets), dim=0)
                    t_outputs_all = torch.cat((t_outputs_all, t_outputs), dim=0)

        acc = n_correct / n_total
        f1 = metrics.f1_score(t_targets_all.cpu(), torch.argmax(t_outputs_all, -1).cpu(), labels=[0, 1, 2],
                              average='macro')
        logger.info('[total:{0}] [total predict wrong:{1}] [Neu2Neg:{2}] [Neu2Pos:{3}] [Neg2Pos:{4}] [Neg2Neu:{5}]'
                    '[Pos2Neu:{6}] [Pos2Neg:{7}]'.format(n_total, predict_wrong, Neu2Neg, Neu2Pos, Neg2Pos, Neg2Neu,
                                                         Pos2Neu, Pos2Neg))
        return acc, f1

    def run(self):
        # Loss and Optimizer
        CE = nn.CrossEntropyLoss()
        # SCL = SupConLoss(self.opt)
        TL = TripletLoss()

        _params = filter(lambda p: p.requires_grad, self.model.parameters())
        optimizer = self.opt.optimizer(_params, lr=self.opt.lr, weight_decay=self.opt.l2reg)

        train_data_loader = DataLoader(dataset=self.trainset, batch_size=self.opt.batch_size, shuffle=True)
        val_data_loader = DataLoader(dataset=self.valset, batch_size=self.opt.batch_size, shuffle=False)
        test_data_loader = DataLoader(dataset=self.testset, batch_size=self.opt.batch_size, shuffle=False)

        self._reset_params()
        # best_model_path = self._train_only_CE(CE, TL, optimizer, train_data_loader, val_data_loader)
        best_model_path = self._train(CE, TL, optimizer, train_data_loader, val_data_loader)
        self.model.load_state_dict(torch.load(best_model_path))
        print(best_model_path)
        test_acc, test_f1 = self._test_evaluate_acc_f1(test_data_loader)
        logger.info('>> test_acc: {:.4f}, test_f1: {:.4f}'.format(test_acc, test_f1))
        logger.info('>> best_model_path: {}'.format(best_model_path))


def main():
    # Hyper Parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='roberta_spc', type=str)
    parser.add_argument('--dataset', default='restaurant', type=str, help='twitter, restaurant, laptop, MAMS')
    parser.add_argument('--optimizer', default='adamw', type=str)
    parser.add_argument('--initializer', default='xavier_uniform_', type=str)
    parser.add_argument('--lr', default=2e-5, type=float, help='try 5e-5, 2e-5 for BERT, 1e-3 for others')
    parser.add_argument('--dropout', default=0.1, type=float)
    parser.add_argument('--l2reg', default=1e-3, type=float)
    parser.add_argument('--lamda', default=0.5, type=float)
    parser.add_argument('--num_epoch', default=50, type=int, help='try larger number for non-BERT models')
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--log_step', default=10, type=int)
    parser.add_argument('--embed_dim', default=300, type=int)
    parser.add_argument('--hidden_dim', default=300, type=int)
    parser.add_argument('--bert_dim', default=768, type=int)
    parser.add_argument('--pretrained_bert_name', default="roberta-base", type=str)
    parser.add_argument('--max_seq_len', default=85, type=int)
    parser.add_argument('--polarities_dim', default=3, type=int)
    parser.add_argument('--hops', default=3, type=int)
    parser.add_argument('--patience', default=8, type=int)
    parser.add_argument('--device', default='cuda:2', type=str, help='e.g. cuda:0')
    parser.add_argument('--seed', default=3152, type=int, help='set seed for reproducibility')
    parser.add_argument('--valset_ratio', default=0.2, type=float,
                        help='set ratio between 0 and 1 for validation support')
    # parser.add_argument('--cross_val_fold', default=10, type=int, help='k-fold cross validation')
    # The following parameters are only valid for the lcf-bert model
    parser.add_argument('--local_context_focus', default='cdw', type=str, help='local context focus mode, cdw or cdm')
    parser.add_argument('--SRD', default=3, type=int)
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
        'roberta_spc': Roberta_SPC,
        'lcf_roberta': LCF_Roberta,
        'lcf_roberta_attention': LCF_ROBERTA_Attention,
    }
    dataset_files = {
        'twitter': {
            'train': './datasets/acl-14-short-data/train.raw',
            'test': './datasets/acl-14-short-data/test.raw'
        },
        'restaurant': {
            'train': "/home/disk2/jye/ABSA/datasets/implicit/restaurant/txt_with_implicit_label4/train_total_text.txt",
            'test': "/home/disk2/jye/ABSA/datasets/implicit/restaurant/txt_with_implicit_label4/test_total_text.txt",
        },
        'restaurant_neu': {
            'train': "/home/disk2/jye/ABSA/datasets/implicit/restaurant/txt_selected_neu/train.txt",
            'test': "/home/disk2/jye/ABSA/datasets/implicit/restaurant/txt_selected_neu/test.txt",
        },
        'laptop': {
            'train': "/home/disk2/jye/ABSA/datasets/implicit/laptop/txt_with_implicit_label4/train_total_text.txt",
            'test': "/home/disk2/jye/ABSA/datasets/implicit/laptop/txt_with_implicit_label4/test_total_text.txt",

        },
        'MAMS': {
            'train': "/home/disk2/jye/ABSA/datasets/implicit/MAMS/sentence_sentiment_labeled/train.txt",
            'val': "/home/disk2/jye/ABSA/datasets/implicit/MAMS/sentence_sentiment_labeled/val.txt",
            'test': "/home/disk2/jye/ABSA/datasets/implicit/MAMS/sentence_sentiment_labeled/test.txt"
        },
    }
    input_colses = {
        'roberta_spc': ['concat_roberta_indices', 'concat_segments_indices'],
        'bert_NLI': ['NLI_bert_indices', 'NLI_segment_indices'],
        'aen_bert': ['text_bert_indices', 'aspect_bert_indices'],
        'lcf_roberta': ['concat_roberta_indices', 'concat_segments_indices',
                        'text_bert_indices', 'aspect_bert_indices'],
        'lcf_roberta_attention': ['concat_roberta_indices', 'concat_segments_indices',
                                  'text_bert_indices', 'aspect_bert_indices'],
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
        'adamw': torch.optim.AdamW
    }
    opt.model_class = model_classes[opt.model_name]
    opt.dataset_file = dataset_files[opt.dataset]
    opt.inputs_cols = input_colses[opt.model_name]
    opt.initializer = initializers[opt.initializer]
    opt.optimizer = optimizers[opt.optimizer]
    opt.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') \
        if opt.device is None else torch.device(opt.device)

    log_file = 'triplet loss: {}-{}-{}-{}-{}-{}-{}-train-{}.log' \
        .format(opt.model_name, opt.dataset, opt.lr, opt.l2reg, opt.batch_size,
                opt.seed, opt.lamda, strftime("%y%m%d-%H%M", localtime()))
    logger.addHandler(logging.FileHandler(log_file))

    ins = Instructor(opt)
    ins.run()


if __name__ == '__main__':
    main()
