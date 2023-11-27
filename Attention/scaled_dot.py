from Attention.attention import Attention
from torch import nn
import math


class ScaledDotAttention(Attention):

    def __init__(self, dropout=0, dim=768):
        super(ScaledDotAttention, self).__init__(dropout, dim)

    def _score(self, query, key):
        """
        query: FloatTensor (batch_size, num_queries, query_size)
        key: FloatTensor (batch_size, time_step, key_size)
        """
        assert query.size(2) == key.size(2)
        return query.matmul(key.transpose(1, 2)) / math.sqrt(query.size(2))
