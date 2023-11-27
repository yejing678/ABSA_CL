# -*- coding: utf-8 -*-
# file: BERT_SPC.py

import torch.nn as nn
import json


class BERT_SPC(nn.Module):
    def __init__(self, bert, opt):
        super(BERT_SPC, self).__init__()
        self.bert = bert
        self.dropout = nn.Dropout(opt.dropout)
        self.dense = nn.Linear(opt.bert_dim, opt.polarities_dim)


    # def forward(self, inputs):
    #     bert_indices = inputs[0]
    #     outputs = self.bert(bert_indices)
    #     last_hidden_state = outputs.last_hidden_state  # 获取最后一层的编码结果
    #     pooled_output = self.dropout(last_hidden_state[:, 0, :])  # 提取最后一个token对应的向量
    #     logits = self.dense(pooled_output)
    #     return logits, pooled_output
    def forward(self, inputs):
        bert_indices = inputs[0]
        _, pooler_out = self.bert(bert_indices)
        pooled_output = self.dropout(pooler_out)
        logits = self.dense(pooled_output)
        return logits, pooler_out
