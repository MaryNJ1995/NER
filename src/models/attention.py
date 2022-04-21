# -*- coding: utf-8 -*-
# pylint: disable-msg=import-error
# pylint: disable-msg=too-many-ancestors
# pylint: disable-msg=arguments-differ
# ========================================================
"""This module is written for write attentions classes."""
# ========================================================


# ========================================================
# Imports
# ========================================================

import numpy as np
import torch
from typing import Optional, Tuple

__author__ = "Ehsan Tavan"
__project__ = "NER_SemEval"
__version__ = "1.0.0"
__date__ = "2021/11/21"
__email__ = "tavan.ehsan@gmail.com"


class ScaledDotProductAttention(torch.nn.Module):
    """
    Scaled Dot-Product Attention

    Inputs: query, key, value, mask
        - query (batch, q_len, d_model): tensor containing projection vector for decoder.
        - key (batch, k_len, d_model): tensor containing projection vector for encoder.
        - value (batch, v_len, d_model): tensor containing features of the encoded input sequence.
        - mask (-): tensor containing indices to be masked
    """

    def __init__(self, dim: int):
        super().__init__()
        self.sqrt_dim = np.sqrt(dim)
        self.wq = torch.nn.Linear(in_features=dim, out_features=dim)
        self.wk = torch.nn.Linear(in_features=dim, out_features=dim)
        self.wv = torch.nn.Linear(in_features=dim, out_features=dim)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        query = self.wq(query)
        key = self.wk(key)
        value = self.wv(value)

        score = torch.bmm(query, key.transpose(1, 2)) / self.sqrt_dim

        if mask is not None:
            mask = mask.unsqueeze(1).expand(-1, query.size(1), -1)
            score.masked_fill_(mask, -float("Inf"))

        attn = torch.nn.functional.softmax(score, dim=-1)

        context = torch.bmm(attn, value)
        return context, attn


class DotProductAttention(torch.nn.Module):
    """
    Compute the dot products of the query with all values and apply a softmax function to obtain the weights on the values
    """

    def __init__(self, hidden_dim):
        super(DotProductAttention, self).__init__()
        self.normalize = torch.nn.LayerNorm(hidden_dim)
        self.out_projection = torch.nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(self, query: torch.Tensor, value: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, hidden_dim, input_size = query.size(0), query.size(2), value.size(1)

        score = torch.bmm(query, value.transpose(1, 2))
        attn = torch.nn.functional.softmax(score.view(-1, input_size), dim=1).view(batch_size, -1, input_size)
        context = torch.bmm(attn, value)

        return context, attn


class SimpleAttention(torch.nn.Module):
    """
    In this class we implement Attention model for text classification
    """

    def __init__(self, rnn_size):
        super(SimpleAttention, self).__init__()
        self.att_fully_connected_layers1 = torch.nn.Linear(rnn_size, 350)
        self.att_fully_connected_layers2 = torch.nn.Linear(350, 30)

    '''
    input param: 
        lstm_output: output of bi-LSTM (batch_size, sent_len, hid_dim * num_directions)   
    return: 
        attention weights matrix, 
        attention_output(batch_size, num_directions * hidden_size): attention output vector for the batch
        with this informations:
        r=30 and da=350 and penalize_coeff = 1
    '''

    def forward(self, lstm_output):
        # work_flow: lstm_output>fc1>tanh>fc2>softmax
        # usage of softmax is to calculate the distribution probability through one sentence :)
        lstm_output = lstm_output.permute(1, 0, 2)
        hidden_matrix = self.att_fully_connected_layers2(torch.tanh(self.att_fully_connected_layers1(lstm_output)))
        # hidden_matrix.size() :(batch_size, sent_len, num_head_attention)===> torch.Size([64, 150, 30])
        # for each of this 150 word, we have 30 feature from 30 attention's head.
        # permute? because the softmax will apply in 3rd dimension and we want apply it on sent_len so:
        hidden_matrix = hidden_matrix.permute(0, 2, 1)
        # hidden_matrix.size() :(batch_size, num_head_attention, sent_len)===>torch.Size([64, 30, 150])
        hidden_matrix = torch.nn.functional.softmax(hidden_matrix, dim=2)
        # hidden_matrix.size() :(batch_size, num_head_attention, sent_len)===>torch.Size([64, 30, 150])
        # print(hidden_matrix.size())

        return hidden_matrix

