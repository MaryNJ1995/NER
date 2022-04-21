# -*- coding: utf-8 -*-
# pylint: disable-msg=import-error
# pylint: disable-msg=too-many-ancestors
# pylint: disable-msg=arguments-differ
# ========================================================
"""This module is written for write MT5 classifier."""
# ========================================================


# ========================================================
# Imports
# ========================================================

import torch

__author__ = "Ehsan Tavan"
__project__ = "NER_SemEval"
__version__ = "1.0.0"
__date__ = "2021/11/30"
__email__ = "tavan.ehsan@gmail.com"


class EncoderLayer(torch.nn.Module):
    def __init__(self, hid_dim, n_heads, pf_dim, dropout):
        super().__init__()

        self.self_attn_layer_norm = torch.nn.LayerNorm(hid_dim)
        self.ff_layer_norm = torch.nn.LayerNorm(hid_dim)
        self.self_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout)
        self.positionwise_feedforward = PositionwiseFeedforwardLayer(hid_dim, pf_dim, dropout)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, src, src_mask):
        # src = [batch_size, src_len, hid_dim]
        # src_mask = [batch_size, 1, 1, src_len]
        src_mask = src_mask.unsqueeze(1)
        src_mask = src_mask.unsqueeze(2)
        # src_mask = [batch_size, 1, 1, src_len]

        # self attention
        _src, _ = self.self_attention(src, src, src, src_mask)

        # dropout, residual connection and layer norm
        src = self.self_attn_layer_norm(src + self.dropout(_src))

        # src = [batch_size, src_len, hid_dim]

        # positionwise feedforward
        _src = self.positionwise_feedforward(src)

        # dropout, residual and layer norm
        src = self.ff_layer_norm(src + self.dropout(_src))

        # src = [batch size, src len, hid dim]

        return src


class MultiHeadAttentionLayer(torch.nn.Module):
    """
    MultiHeadAttentionLayer
    """

    def __init__(self, hid_dim, n_heads, dropout):
        super().__init__()

        assert hid_dim % n_heads == 0

        self.hid_dim = hid_dim
        self.n_heads = n_heads
        self.head_dim = hid_dim // n_heads

        self.fc_q = torch.nn.Linear(hid_dim, hid_dim)
        self.fc_k = torch.nn.Linear(hid_dim, hid_dim)
        self.fc_v = torch.nn.Linear(hid_dim, hid_dim)

        self.fc_o = torch.nn.Linear(hid_dim, hid_dim)

        self.dropout = torch.nn.Dropout(dropout)

        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim])).to("cuda:1")

    def forward(self, query, key, value, mask=None):
        batch_size = query.shape[0]

        # query = [batch_size, query_len, hid_dim]
        # key = [batch_size, key_len, hid_dim]
        # value = [batch_size, value_len, hid_dim]

        query = self.fc_q(query)
        key = self.fc_k(key)
        value = self.fc_v(value)

        # query = [batch_size, query_len, hid_dim]
        # key = [batch_size, key_len, hid_dim]
        # value = [batch_size, value_len, hid_dim]

        query = query.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        key = key.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        value = value.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)

        # query = [batch_size, n_heads, query_len, head_dim]
        # key = [batch_size, n_heads, key_len, head_dim]
        # value = [batch_size, n_heads, value_len, head_dim]

        energy = torch.matmul(query, key.permute(0, 1, 3, 2)) / self.scale

        # energy = [batch_size, n_heads, query_len, key_len]

        # if mask is not None:
        #     energy = energy.masked_fill(mask == 0, -1e10)

        attention = torch.softmax(energy, dim=-1)

        # attention = [batch_size, n_heads, query_len, key_len]

        context = torch.matmul(self.dropout(attention), value)

        # context = [batch_size, n_heads, query_len, head_dim]

        context = context.permute(0, 2, 1, 3).contiguous()

        # context = [batch_size, query_len, n_heads, head_dim]

        context = context.view(batch_size, -1, self.hid_dim)

        # context = [batch_size, query_len, hid_dim]

        context = self.fc_o(context)

        # context = [batch_size, query_len, hid_dim]

        return context, attention


class PositionwiseFeedforwardLayer(torch.nn.Module):
    """
    PositionwiseFeedforwardLayer
    """

    def __init__(self, hid_dim, pf_dim, dropout):
        super().__init__()

        self.fc_1 = torch.nn.Linear(hid_dim, pf_dim)
        self.fc_2 = torch.nn.Linear(pf_dim, hid_dim)

        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, inputs):
        # inputs = [batch_size, seq_len, hid_dim]

        inputs = self.dropout(torch.relu(self.fc_1(inputs)))

        # inputs = [batch_size, seq_len, pf_dim]

        inputs = self.fc_2(inputs)

        # inputs = [batch_size, seq_len, hid_dim]

        return inputs
