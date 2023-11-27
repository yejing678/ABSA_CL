# -*- coding: utf-8 -*-
# file: BERT_SPC.py
from transformers import AutoTokenizer, BertForMaskedLM
import torch
import torch.nn as nn
import json


class BERT_Mask(nn.Module):
    def __init__(self, bert, opt):
        super(BERT_Mask, self).__init__()
        self.bert = bert
        self.dropout = nn.Dropout(opt.dropout)
        self.dense = nn.Linear(opt.bert_dim, opt.polarities_dim)

    def forward(self, inputs):
        bert_indices, mask_index = inputs[0], inputs[1]
        print(mask_index)
        last_hidden_state, pooler_output = self.bert(bert_indices)
        mask_embeddings = last_hidden_state[0, mask_index, :]
        logits = self.dense(mask_embeddings)
        return logits, mask_embeddings





    # def forward(self, inputs):
    #     bert_indices = inputs[0]
    #     _, pooler_out = self.bert(bert_indices)
    #     pooled_output = self.dropout(pooler_out)
    #     logits = self.dense(pooled_output)
    #     return logits, pooler_out
