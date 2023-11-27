# -*- coding: utf-8 -*-

import os
import sys
import pickle

import numpy as np
from torch.utils.data import Dataset
from transformers import RobertaTokenizer
sys.path.append(os.pardir)  # 为了导入父目录的文件而进行的设定


def build_tokenizer(fnames, max_seq_len, dat_fname):
    if os.path.exists(dat_fname):
        print('loading tokenizer:', dat_fname)
        tokenizer = pickle.load(open(dat_fname, 'rb'))
    else:
        text = ''
        for fname in fnames:
            fin = open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
            lines = fin.readlines()
            fin.close()
            for i in range(0, len(lines), 3):
                text_left, _, text_right = [s.lower().strip() for s in lines[i].partition("$T$")]
                aspect = lines[i + 1].lower().strip()
                text_raw = text_left + " " + aspect + " " + text_right
                text += text_raw + " "

        tokenizer = Tokenizer(max_seq_len)
        tokenizer.fit_on_text(text)
        pickle.dump(tokenizer, open(dat_fname, 'wb'))
    return tokenizer


def _load_word_vec(path, word2idx=None, embed_dim=300):
    fin = open(path, 'r', encoding='utf-8', newline='\n', errors='ignore')
    word_vec = {}
    for line in fin:
        tokens = line.rstrip().split()
        word, vec = ' '.join(tokens[:-embed_dim]), tokens[-embed_dim:]
        if word in word2idx.keys():
            word_vec[word] = np.asarray(vec, dtype='float32')
    return word_vec


def build_embedding_matrix(word2idx, embed_dim, dat_fname):
    if os.path.exists(dat_fname):
        print('loading embedding_matrix:', dat_fname)
        embedding_matrix = pickle.load(open(dat_fname, 'rb'))
    else:
        print('loading word vectors...')
        embedding_matrix = np.zeros((len(word2idx) + 2, embed_dim))  # idx 0 and len(word2idx)+1 are all-zeros
        fname = './glove.twitter.27B/glove.twitter.27B.' + str(embed_dim) + 'd.txt' \
            if embed_dim != 300 else './glove.840B.300d.txt'
        print(fname)
        word_vec = _load_word_vec(fname, word2idx=word2idx, embed_dim=embed_dim)
        print('building embedding_matrix:', dat_fname)
        for word, i in word2idx.items():
            vec = word_vec.get(word)
            if vec is not None:
                # words not found in embedding index will be all-zeros.
                embedding_matrix[i] = vec
        pickle.dump(embedding_matrix, open(dat_fname, 'wb'))
    return embedding_matrix


def pad_and_truncate(sequence, maxlen, dtype='int64', padding='post', truncating='post', value=0):
    x = (np.ones(maxlen) * value).astype(dtype)
    if truncating == 'pre':
        trunc = sequence[-maxlen:]
    else:
        trunc = sequence[:maxlen]
    trunc = np.asarray(trunc, dtype=dtype)
    if padding == 'post':
        x[:len(trunc)] = trunc
    else:
        x[-len(trunc):] = trunc
    return x


class Tokenizer(object):
    def __init__(self, max_seq_len, lower=True):
        self.lower = lower
        self.max_seq_len = max_seq_len
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 1

    def fit_on_text(self, text):
        if self.lower:
            text = text.lower()
        words = text.split()
        for word in words:
            if word not in self.word2idx:
                self.word2idx[word] = self.idx
                self.idx2word[self.idx] = word
                self.idx += 1

    def text_to_sequence(self, text, reverse=False, padding='post', truncating='post'):
        if self.lower:
            text = text.lower()
        words = text.split()
        unknownidx = len(self.word2idx) + 1
        sequence = [self.word2idx[w] if w in self.word2idx else unknownidx for w in words]
        if len(sequence) == 0:
            sequence = [0]
        if reverse:
            sequence = sequence[::-1]
        print(sequence)
        return pad_and_truncate(sequence, self.max_seq_len, padding=padding, truncating=truncating)


class Tokenizer4Roberta:
    def __init__(self, max_seq_len):
        self.tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
        self.max_seq_len = max_seq_len

    def text_to_sequence(self, text, reverse=False, padding='post', truncating='post'):
        sequence = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(text))
        if len(sequence) == 0:
            sequence = [0]
        if reverse:
            sequence = sequence[::-1]
        return pad_and_truncate(sequence, self.max_seq_len, padding=padding, truncating=truncating)


class ABSADataset(Dataset):
    def __init__(self, fname, tokenizer):
        fin = open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
        lines = fin.readlines()
        fin.close()
        # fin = open(fname + '.graph', 'rb')
        # idx2graph = pickle.load(fin)
        # fin.close()

        all_data = []
        for i in range(0, len(lines), 4):
            text_left, _, text_right = [s.lower().strip() for s in lines[i].partition("$T$")]
            aspect = lines[i + 1].lower().strip()
            polarity = lines[i + 2].strip()
            implicit_label = lines[i+3].strip()
            d = {
                'Positive': '1',
                'Negative': '-1',
                'Neutral': '0'
            }
            # polarity = int(d[polarity]) + 1
            polarity = int(polarity) + 1
            implicit_label = int(implicit_label) + 1
            text_indices = tokenizer.text_to_sequence(text_left + " " + aspect + " " + text_right)
            context_indices = tokenizer.text_to_sequence(text_left + " " + text_right)
            left_indices = tokenizer.text_to_sequence(text_left)
            left_with_aspect_indices = tokenizer.text_to_sequence(text_left + " " + aspect)
            right_indices = tokenizer.text_to_sequence(text_right, reverse=True)
            right_with_aspect_indices = tokenizer.text_to_sequence(aspect + " " + text_right, reverse=True)
            aspect_indices = tokenizer.text_to_sequence(aspect)
            left_len = np.sum(left_indices != 0)
            right_len = np.sum(right_indices != 0)
            aspect_len = np.sum(aspect_indices != 0)
            aspect_boundary = np.asarray([left_len, left_len + aspect_len - 1], dtype=np.int64)
            aspect_bert_boundary = np.asarray([left_len + 1, left_len + aspect_len], dtype=np.int64)


            text_len = np.sum(text_indices != 0)
            concat_roberta_indices = tokenizer.text_to_sequence('<s>' + text_left + " " + aspect + " " + text_right + '</s>'+ '</s>'+ aspect + '</s>')
            concat_segments_indices = [0] * (text_len + 2) + [1] * (aspect_len + 2)
            concat_segments_indices = pad_and_truncate(concat_segments_indices, tokenizer.max_seq_len)


            auxiliary_bert_seq = [0] * (left_len + 1) + [1] * aspect_len + [0] * (right_len + 1)
            auxiliary_bert_seq = pad_and_truncate(auxiliary_bert_seq, tokenizer.max_seq_len)

            text_bert_indices = tokenizer.text_to_sequence('<s>'+text_left + ' ' + aspect + ' ' + text_right+'</s>')
            aspect_bert_indices = tokenizer.text_to_sequence('<s>'+aspect+'</s>')
            left_bert_indices = tokenizer.text_to_sequence('<s>'+text_left+'</s>')
            left_aspect_bert_indices = tokenizer.text_to_sequence('<s>'+text_left + ' ' + aspect+'</s>')
            # dependency_graph = np.pad(idx2graph[i], \
            #                           ((0, tokenizer.max_seq_len - idx2graph[i].shape[0]),
            #                            (0, tokenizer.max_seq_len - idx2graph[i].shape[0])), 'constant')

            data = {
                'text': text_left + " " + aspect + " " + text_right,
                'aspect': aspect,
                'concat_roberta_indices': concat_roberta_indices,
                'concat_segments_indices': concat_segments_indices,
                'text_bert_indices': text_bert_indices,
                'aspect_bert_indices': aspect_bert_indices,
                'left_bert_indices': left_bert_indices,
                'left_aspect_bert_indices': left_aspect_bert_indices,
                'text_indices': text_indices,
                'context_indices': context_indices,
                'left_indices': left_indices,
                'left_with_aspect_indices': left_with_aspect_indices,
                'right_indices': right_indices,
                'right_with_aspect_indices': right_with_aspect_indices,
                'aspect_indices': aspect_indices,
                'aspect_boundary': aspect_boundary,
                # 'dependency_graph': dependency_graph,
                'polarity': polarity,
                'implicit_label':implicit_label,
                'aspect_bert_boundary': aspect_bert_boundary,
                'auxiliary_bert_seq': auxiliary_bert_seq,
            }

            all_data.append(data)
        self.data = all_data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)
