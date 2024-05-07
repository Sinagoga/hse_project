%%capture
import torch
import torch.nn as nn
from torch.nn import functional as nnf
from torch.amp import autocast
from torch import einsum
import torch.nn.functional as F

import open_clip

from transformers import GPT2LMHeadModel, AutoTokenizer

from typing import Optional

from transformers.optimization import Adafactor
import numpy as np
from torch.utils.data import DataLoader

from torchmetrics import BLEUScore
from evaluate import load
from statistics import mean

import math

class MultiHeadAttention(nn.Module):
  def __init__(self, input_dim, dim_embedds, num_heads):
    super(MultiHeadAttention, self).__init__()
    assert dim_embedds % num_heads == 0

    self.dim_embedds = dim_embedds
    self.num_heads = num_heads
    self.d_k = dim_embedds // num_heads

    self.W_qkv = nn.Linear(input_dim, 3 * dim_embedds)
    self.W_o = nn.Linear(dim_embedds, dim_embedds)
    self.dropout = nn.Dropout(0)

    self._reset_parameters()
  def _reset_parameters(self):
        nn.init.xavier_uniform_(self.W_qkv.weight)
        self.W_qkv.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.W_o.weight)
        self.W_o.bias.data.fill_(0)

  def scaled_dot_product_attention(self, Q, K, V, mask=None):
    attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
    if mask is not None:
        attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
    attention = torch.softmax(attn_scores, dim=-1)
    output = torch.matmul(self.dropout(attention), V)
    return output, attention

  def combine_heads(self, x, batch_size, seq_length):
    return x.permute(0, 2, 1, 3).reshape(batch_size, seq_length, self.dim_embedds)

  def forward(self, x, mask=None, return_attn=False):
    batch_size, seq_length, _ = x.size()

    if mask is not None:
      mask = expand_mask(mask)
    QKV = self.W_qkv(x)
    QKV = QKV.reshape(batch_size, seq_length, self.num_heads, 3 * self.d_k)
    QKV = QKV.permute(0, 2, 1, 3)
    q, k, v = QKV.chunk(3, dim=-1)

    attn_output, attention = self.scaled_dot_product_attention(q,k,v, mask)
    attn_output = attn_output.permute(0, 2, 1, 3)
    attn_output = attn_output.reshape(batch_size, seq_length, self.dim_embedds)

    output = self.W_o(self.combine_heads(attn_output, batch_size, seq_length))
    if return_attn:
       return output, attention
    return output

class FeedForward(nn.Module): #MLP
    def __init__(self, inp_shape, output_shape, act=nn.ReLU):
        super(FeedForward, self).__init__()
        self.seq = nn.Sequential(
            nn.Linear(inp_shape, inp_shape*2),
            act(),
            nn.Linear(inp_shape*2, output_shape*2)
        )
    @autocast("cuda")
    def forward(self, x):
        return self.seq(x)
class TextFeedForward(nn.Module):
    def __init__(self, text_emb_size, output_size, act=nn.ReLU):
        super(TextFeedForward, self).__init__()
        self.seq = nn.Sequential(
            nn.Linear(text_emb_size, text_emb_size*2),
            act(),
            nn.Linear(text_emb_size*2, text_emb_size*2),
            act(),
            nn.Linear(text_emb_size*2, output_size)
        )
    def forward(self, x):
        return self.seq(x)
