# -*- coding: utf-8 -*-

import argparse
import logging
import math
import os
import random
import sys
import json
from time import strftime, localtime

import numpy
import torch
from sklearn import metrics
from torch.utils.data import DataLoader, random_split
from transformers import RobertaModel

from data_prepare import build_tokenizer, build_embedding_matrix, Tokenizer4Roberta, ABSADataset
from robert_spc import Roberta_SPC
from roberta_lcf import LCF_Roberta
from roberta_attention import LCF_ROBERTA_Attention

sys.path.append(os.pardir)  # 为了导入父目录的文件而进行的设定

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))


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
        self.testset = ABSADataset(opt.dataset_file['test'], tokenizer)

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
                t_outputs, t_features = self.model(t_inputs)

                dict1 = {
                    "features": t_features.tolist(),
                    "label": t_targets.tolist(),
                }

                save_path = "json/{0}/{1}/{2}/{3}/" \
                    .format(self.opt.model_name, self.opt.local_context_focus, self.opt.dataset, self.opt.seed)
                dirname = os.path.dirname(save_path)
                if not os.path.exists(dirname):
                    os.makedirs(dirname)
                path = save_path + 'test.json'
                json.dump(dict1, open(path, 'w'), indent=6)

                n_predict = torch.argmax(t_outputs, -1)
                n_correct += (n_predict == t_targets).sum().item()
                n_total += len(t_outputs)
                for i in range(len(t_batch['text'])):
                    if not n_predict[i] == t_targets[i]:
                        predict_wrong += 1
                        logger.info('============================Prediction Error=============================')
                        logger.info(
                            '{0} \n>>>aspect:{1}\n>>>ground_truth: {2}\n>>>predict：{3}'.format(t_batch['text'][i],
                                                                                               t_batch['aspect'][i],
                                                                                               t_batch['polarity'][
                                                                                                   i] - 1,
                                                                                               n_predict[i] - 1))
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
        test_data_loader = DataLoader(dataset=self.testset, batch_size=self.opt.batch_size, shuffle=False)
        best_model_path="/home/disk2/jye/ABSA_Curriculum_Learning/roberta-absa/state_dict/train/lcf_roberta_attention/restaurant/cdm/7321/epoch2_step130_acc_0.7986_f10.7084"
        self.model.load_state_dict(torch.load(best_model_path))
        test_acc, test_f1 = self._test_evaluate_acc_f1(test_data_loader)
        logger.info('>> test_acc: {:.4f}, test_f1: {:.4f}'.format(test_acc, test_f1))
        logger.info(best_model_path)


def main():
    # Hyper Parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_mode', default='CE+SCL', type=str, help='CE,CE+SCL,CE+TL,CE+SCL+TL')
    parser.add_argument('--model_name', default='lcf_roberta_attention', type=str)
    parser.add_argument('--dataset', default='restaurant', type=str, help='twitter, restaurant, laptop, MAMS')
    parser.add_argument('--initializer', default='xavier_uniform_', type=str)
    parser.add_argument('--lr', default=2e-5, type=float, help='try 5e-5, 2e-5 for BERT, 1e-3 for others')
    parser.add_argument('--dropout', default=0.1, type=float)
    parser.add_argument('--l2reg', default=1e-3, type=float)
    parser.add_argument('--margin', default=0.2, type=float, help='triplet loss margin')
    parser.add_argument('--beta', default=0.2, type=float, help='triplet loss weight')
    parser.add_argument('--lamda', default=1, type=float, help='SupConLoss weight')
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
    parser.add_argument('--device', default='cuda:0', type=str, help='e.g. cuda:0')
    parser.add_argument('--seed', default=41, type=int, help='set seed for reproducibility')
    parser.add_argument('--valset_ratio', default=0.2, type=float,
                        help='set ratio between 0 and 1 for validation support')
    # parser.add_argument('--cross_val_fold', default=10, type=int, help='k-fold cross validation')
    # The following parameters are only valid for the lcf-bert model
    parser.add_argument('--local_context_focus', default='cdm', type=str, help='local context focus mode, cdw or cdm')
    parser.add_argument('--SRD', default=0, type=int,
                        help='semantic-relative-distance, see the paper of LCF-BERT model')
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
            'train': "/home/disk2/jye/ABSA/datasets/implicit/twitter/sentence_sentiment_labeled/train.txt",
            'test': "/home/disk2/jye/ABSA/datasets/implicit/twitter/sentence_sentiment_labeled/test.txt"
        },
        'restaurant_total': {
            'test': "/home/disk2/jye/ABSA/datasets/implicit/restaurant/txt_with_implicit_label4/test_total_text.txt",
        },
        'restaurant_implicit': {
            'test': "/home/disk2/jye/ABSA/datasets/implicit/restaurant/txt_with_implicit_label4/test_implicit_text.txt",
        },
        'restaurant_explicit': {
            'test': "/home/disk2/jye/ABSA/datasets/implicit/restaurant/txt_with_implicit_label4/test_explicit_text.txt",
        },
        'laptop_total': {
            'test': "/home/disk2/jye/ABSA/datasets/implicit/laptop/txt_with_implicit_label4/test_total_text.txt",
        },
        'laptop_implicit': {
            'test': "/home/disk2/jye/ABSA/datasets/implicit/laptop/txt_with_implicit_label4/test_implicit_text.txt",
        },
        'laptop_explicit': {
            'test': "/home/disk2/jye/ABSA/datasets/implicit/laptop/txt_with_implicit_label4/test_explicit_text.txt",
        },
        'MAMS': {
            'train': './datasets/MAMS/train.txt',
            'val': './datasets/MAMS/val.txt',
            'test': './datasets/MAMS/test.txt'
        },
        'amazon': {
            'train': "/home/disk2/jye/ABSA/datasets/implicit/twitter/sentence_sentiment_labeled.txt",
            'test': "/home/disk2/jye/ABSA/datasets/implicit/twitter/sentence_sentiment_labeled.txt",
        },
        'yelp': {
            'train': "/home/disk2/jye/ABSA/datasets/implicit/yelp/yelp.txt",
            'test': "/home/disk2/jye/ABSA/datasets/implicit/yelp/yelp.txt",
        }

    }
    input_colses = {
        'roberta_spc': ['concat_roberta_indices', 'concat_segments_indices'],
        'lcf_roberta': ['concat_roberta_indices', 'concat_segments_indices', 'text_bert_indices',
                        'aspect_bert_indices'],
        'lcf_roberta_attention': ['concat_roberta_indices', 'concat_segments_indices', 'text_bert_indices',
                                  'aspect_bert_indices'],
    }
    initializers = {
        'xavier_uniform_': torch.nn.init.xavier_uniform_,
        'xavier_normal_': torch.nn.init.xavier_normal_,
        'orthogonal_': torch.nn.init.orthogonal_,
    }
    opt.model_class = model_classes[opt.model_name]
    opt.dataset_file = dataset_files[opt.dataset]
    opt.inputs_cols = input_colses[opt.model_name]
    opt.initializer = initializers[opt.initializer]
    opt.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') \
        if opt.device is None else torch.device(opt.device)

    log_file = 'test:{0}-{1}-{2}-{3}-{4}-{5}-{6}.log' \
        .format(opt.model_name, opt.dataset, opt.lr, opt.l2reg, opt.batch_size, opt.seed, strftime("%y%m%d-%H%M", localtime()))
    logger.addHandler(logging.FileHandler(log_file))

    ins = Instructor(opt)
    ins.run()


if __name__ == '__main__':
    main()
