# -*- coding: utf-8 -*-

import torch.nn as nn


class Roberta_SPC(nn.Module):
    def __init__(self, roberta, opt):
        super(Roberta_SPC, self).__init__()
        self.roberta = roberta
        self.dropout = nn.Dropout(opt.dropout)
        self.dense = nn.Linear(opt.bert_dim, opt.polarities_dim)

    def forward(self, inputs):
        text_bert_indices, bert_segments_ids = inputs[0], inputs[1]
        outputs = self.roberta(input_ids=text_bert_indices)
        pooler_output = outputs.pooler_output
        pooler_output1 = self.dropout(pooler_output)
        logits = self.dense(pooler_output1)
        return logits, pooler_output