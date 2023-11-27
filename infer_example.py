import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from transformers import BertModel

from data_utils import build_tokenizer, build_embedding_matrix, Tokenizer4Bert, pad_and_truncate
from dependency_graph import dependency_adj_matrix
from models import IAN, MemNet, RAM, TD_LSTM, TC_LSTM, Cabasc, TNet_LF, AOA, MGAN, ASGCN, RAM_BERT, RAM_BERT_1
from models import LCF_BERT, LCF_BERT_ONLY_LOCAL, LCF_BERT_ONLY_GLO, LCF_BERT_LOCAL_CLS, LCF_BERT_ONLY_Global_1
from models import LCF_BERT_xGlo_1_xLocal
from models import LCF_BERT_Attention
from models.aen import AEN_BERT, CrossEntropyLoss_LSR
from models.atae_lstm import ATAE_LSTM, ATAE_BLSTM, ATAE_LSTM_TANH
from models.bert_atae import ATAE_BERT
from models.bert_spc import BERT_SPC
from models.bert_NLI import BERT_NLI
from models.bert_simple import BERT_SIMPLE
from models.aspect_aware_bert import BERT_ASP
from models.lstm import LSTM, BiLSTM

sys.path.append(os.pardir)  # 为了导入父目录的文件而进行的设定


class Inferer:

    def __init__(self, opt):
        self.opt = opt
        print('building embedding ...')
        if 'bert' in opt.model_name:
            self.tokenizer = Tokenizer4Bert(opt.max_seq_len, opt.pretrained_bert_name)
            bert = BertModel.from_pretrained(opt.pretrained_bert_name, return_dict=False)
            self.model = opt.model_class(bert, opt).to(opt.device)
        else:
            self.tokenizer = build_tokenizer(
                fnames=[opt.dataset_file['train'], opt.dataset_file['test']],
                max_seq_len=opt.max_seq_len,
                dat_fname='{0}_tokenizer.dat'.format(opt.dataset))
            embedding_matrix = build_embedding_matrix(
                word2idx=self.tokenizer.word2idx,
                embed_dim=opt.embed_dim,
                dat_fname='{0}_{1}_embedding_matrix.dat'.format(str(opt.embed_dim), opt.dataset))
            self.model = opt.model_class(embedding_matrix, opt)
        print('loading model {0} ...'.format(opt.model_name))
        self.model.load_state_dict(torch.load(opt.state_dict_path))
        self.model = self.model.to(opt.device)
        # switch model to evaluation mode
        self.model.eval()
        torch.autograd.set_grad_enabled(False)

    def evaluate(self, text, aspect):
        aspect = aspect.lower().strip()
        text_left, _, text_right = [s.strip() for s in text.lower().partition(aspect)]

        text_indices = self.tokenizer.text_to_sequence(text_left + " " + aspect + " " + text_right)
        context_indices = self.tokenizer.text_to_sequence(text_left + " " + text_right)
        left_indices = self.tokenizer.text_to_sequence(text_left)
        left_with_aspect_indices = self.tokenizer.text_to_sequence(text_left + " " + aspect)
        right_indices = self.tokenizer.text_to_sequence(text_right, reverse=True)
        right_with_aspect_indices = self.tokenizer.text_to_sequence(aspect + " " + text_right, reverse=True)
        aspect_indices = self.tokenizer.text_to_sequence(aspect)
        left_len = np.sum(left_indices != 0)
        right_len = np.sum(right_indices != 0)
        aspect_len = np.sum(aspect_indices != 0)
        aspect_boundary = np.asarray([left_len, left_len + aspect_len - 1], dtype=np.int64)

        auxiliary_bert_seq = [0] * (left_len + 1) + [1] * aspect_len + [0] * (right_len + 1)
        auxiliary_bert_seq = pad_and_truncate(auxiliary_bert_seq, self.tokenizer.max_seq_len)

        text_len = np.sum(text_indices != 0)
        concat_bert_indices = self.tokenizer.text_to_sequence(
            '[CLS] ' + text_left + " " + aspect + " " + text_right + ' [SEP] ' + aspect + " [SEP]")
        concat_segments_indices = [0] * (text_len + 2) + [1] * (aspect_len + 1)
        concat_segments_indices = pad_and_truncate(concat_segments_indices, self.tokenizer.max_seq_len)

        text_bert_indices = self.tokenizer.text_to_sequence(
            "[CLS] " + text_left + " " + aspect + " " + text_right + " [SEP]")
        aspect_bert_indices = self.tokenizer.text_to_sequence("[CLS] " + aspect + " [SEP]")

        dependency_graph = dependency_adj_matrix(text)

        data = {
            'concat_bert_indices': concat_bert_indices,
            'concat_segments_indices': concat_segments_indices,
            'text_bert_indices': text_bert_indices,
            'aspect_bert_indices': aspect_bert_indices,
            'text_indices': text_indices,
            'context_indices': context_indices,
            'left_indices': left_indices,
            'left_with_aspect_indices': left_with_aspect_indices,
            'right_indices': right_indices,
            'right_with_aspect_indices': right_with_aspect_indices,
            'aspect_indices': aspect_indices,
            'aspect_boundary': aspect_boundary,
            'dependency_graph': dependency_graph,
            'auxiliary_bert_seq': auxiliary_bert_seq

        }

        t_inputs = [torch.tensor([data[col]], device=self.opt.device) for col in self.opt.inputs_cols]
        t_outputs,_ = self.model(t_inputs)
        t_probs = F.softmax(t_outputs, dim=-1).cpu().numpy()

        return t_probs


if __name__ == '__main__':
    model_classes = {
        'lstm': LSTM,
        'bilstm': BiLSTM,
        'td_lstm': TD_LSTM,
        'tc_lstm': TC_LSTM,
        'atae_lstm': ATAE_LSTM,
        'atae_bilstm': ATAE_BLSTM,
        'atae_lstm_tanh': ATAE_LSTM_TANH,
        'atae_bert': ATAE_BERT,
        'ian': IAN,
        'memnet': MemNet,
        'ram': RAM,
        'cabasc': Cabasc,
        'tnet_lf': TNet_LF,
        'aoa': AOA,
        'mgan': MGAN,
        'asgcn': ASGCN,
        'bert_spc': BERT_SPC,
        'bert_simple': BERT_SIMPLE,
        'aen_bert': AEN_BERT,
        'lcf_bert': LCF_BERT,
        'lcf_bert_only_lcf': LCF_BERT_ONLY_LOCAL,
        'lcf_bert_only_glo': LCF_BERT_ONLY_GLO,
        'lcf_bert_only_global': LCF_BERT_ONLY_Global_1,
        'lcf_bert_local_cls': LCF_BERT_LOCAL_CLS,
        'lcf_bert_xglo_1_xlocal': LCF_BERT_xGlo_1_xLocal,
        'lcf_bert_attention':LCF_BERT_Attention,
        'ram_bert': RAM_BERT,
        'ram_bert_1': RAM_BERT_1,
        'bert_NLI': BERT_NLI,
        'bert_asp':BERT_ASP,
    }
    dataset_files = {
        'twitter': {
            'train': './datasets/acl-14-short-data/train.raw',
            'test': './datasets/acl-14-short-data/test.raw'
        },
        'restaurant': {
            # 'train': "/home/disk2/jye/ABSA/datasets/implicit/restaurant/txt1/train_enhanced.txt",
            # 'train': "/home/disk2/jye/ABSA/datasets/semeval14/few-shot-16.txt",
            'val': "/home/disk2/jye/ABSA/datasets/implicit/restaurant/txt1/test_total_text.txt",
            'train': "/home/disk2/jye/ABSA/datasets/implicit/restaurant/txt1/train_total_text.txt",
            'test': "/home/disk2/jye/ABSA/datasets/implicit/restaurant/txt1/test_implicit_text.txt"
        },
        'laptop': {
            'train': "/home/disk2/jye/ABSA/datasets/implicit/laptop/txt1/train_total_text.txt",
            'val': "/home/disk2/jye/ABSA/datasets/implicit/laptop/txt1/test_total_text.txt",
            'test': "/home/disk2/jye/ABSA/datasets/implicit/laptop/txt1/test_implicit_text.txt",
        },
        'MAMS': {
            'train': './datasets/MAMS/train.txt',
            'val': './datasets/MAMS/val.txt',
            'test': './datasets/MAMS/test.txt'
        },
    }
    input_colses = {
        'lstm': ['text_indices'],
        'bilstm': ['text_indices'],
        'td_lstm': ['left_with_aspect_indices', 'right_with_aspect_indices'],
        'tc_lstm': ['left_with_aspect_indices', 'right_with_aspect_indices', 'aspect_indices'],
        'atae_lstm': ['text_indices', 'aspect_indices'],
        'atae_bilstm': ['text_indices', 'aspect_indices'],
        'atae_lstm_tanh': ['text_indices', 'aspect_indices'],
        'atae_bert': ['text_bert_indices', 'aspect_bert_indices'],
        'ian': ['text_indices', 'aspect_indices'],
        'memnet': ['context_indices', 'aspect_indices'],
        'ram': ['text_indices', 'aspect_indices', 'left_indices'],
        'ram_bert': ['text_bert_indices', 'aspect_bert_indices'],
        'ram_bert_1': ['text_bert_indices', 'auxiliary_bert_seq', 'aspect_indices'],
        'cabasc': ['text_indices', 'aspect_indices', 'left_with_aspect_indices', 'right_with_aspect_indices'],
        'tnet_lf': ['text_indices', 'aspect_indices', 'aspect_boundary'],
        'aoa': ['text_indices', 'aspect_indices'],
        'mgan': ['text_indices', 'aspect_indices', 'left_indices'],
        # 'asgcn': ['text_indices', 'aspect_indices', 'left_indices', 'dependency_graph'],
        'bert_spc': ['concat_bert_indices', 'concat_segments_indices'],
        'bert_simple': ['text_bert_indices'],
        'bert_asp': ['text_bert_indices', 'auxiliary_bert_seq'],
        'bert_NLI': ['NLI_bert_indices', 'NLI_segment_indices'],
        'aen_bert': ['text_bert_indices', 'aspect_bert_indices'],
        'lcf_bert': ['concat_bert_indices', 'concat_segments_indices', 'text_bert_indices', 'aspect_bert_indices'],
        'lcf_bert_only_lcf': ['concat_bert_indices', 'concat_segments_indices', 'text_bert_indices',
                              'aspect_bert_indices'],
        'lcf_bert_only_glo': ['concat_bert_indices', 'concat_segments_indices', 'text_bert_indices',
                              'aspect_bert_indices'],
        'lcf_bert_only_global': ['concat_bert_indices', 'concat_segments_indices', 'text_bert_indices',
                                 'aspect_bert_indices'],
        'lcf_bert_local_cls': ['concat_bert_indices', 'concat_segments_indices', 'text_bert_indices',
                               'aspect_bert_indices'],
        'lcf_bert_xglo_1_xlocal': ['concat_bert_indices', 'concat_segments_indices', 'text_bert_indices',
                                   'aspect_bert_indices'],
        'lcf_bert_attention': ['concat_bert_indices', 'concat_segments_indices', 'text_bert_indices',
                               'aspect_bert_indices'],
    }


    class Option(object): pass


    opt = Option()
    # set your trained models here
    opt.model_name = 'lcf_bert_attention'
    opt.model_class = model_classes[opt.model_name]
    opt.dataset = 'laptop'
    opt.dataset_file = dataset_files[opt.dataset]
    opt.inputs_cols = input_colses[opt.model_name]
    opt.state_dict_path = "/home/disk2/jye/ABSA/state_dict_2/lcf_bert_attention/laptop/cdm /seed_8745_epoch14_step10_acc_0.8056_f10.7689"
    opt.embed_dim = 300
    opt.hidden_dim = 300
    opt.max_seq_len = 20
    opt.bert_dim = 768
    opt.dropout = 0
    opt.pretrained_bert_name = 'bert-base-uncased'
    opt.polarities_dim = 3
    opt.hops = 1
    opt.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    opt.local_context_focus = 'cdw'
    opt.SRD = 3
    opt.alpha_g = 1.0
    opt.alpha_l = 1.0

    inf = Inferer(opt)
    # t_probs = inf.evaluate(
    #     "So far, I am very happy with this laptop, it is fast, lightweight, the screen is sharp & easy on the eyes.",
    #     'laptop')
    t_probs = inf.evaluate(
        "An excellent laptop with the i7 chip. Very thin and fast. Moving from a i4 Asus, I highly recommend this laptop",
        'i4 Asus')
    print(t_probs.argmax(axis=-1) - 1)
