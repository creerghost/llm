import torch
from torch import nn, Tensor
from typing import Any


class SelfAttention(nn.Module):
    """
    Torch module that allow tokens to "attend" to each other, culminating
    in a MultiHeadAttention class.
    """
    def __init__(self, d_in: int, d_out: int, qkv_bias: bool = False) -> None:
        super().__init__()
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

    def _compute_scores_and_values(self, x: Tensor) -> tuple[Tensor, Tensor]:
        queries = self.W_query(x)
        keys = self.W_key(x)
        values = self.W_value(x)
        # calculate attention scores + scale them
        omega = queries @ keys.transpose(1, 2)
        omega /= keys.shape[-1] ** 0.5
        return omega, values

    def forward(self, x: Tensor) -> Tensor:
        omega, values = self._compute_scores_and_values(x)
        # apply softmax across the last dimension, then mult. by values
        attention_weights = torch.softmax(omega, dim=-1)
        return attention_weights @ values


class CausalAttention(SelfAttention):
    """
    Masking for attention layers: token should not be able to look at
    future tokens.
    """
    def __init__(self, d_in: int, d_out: int,
                 ctx_length: int, dropout: float,
                 qkv_bias: bool = False) -> None:
        super().__init__(d_in, d_out, qkv_bias)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("mask",
                             torch.tril(torch.ones(ctx_length, ctx_length)))

    def forward(self, x: Tensor) -> Tensor:
        omega, values = self._compute_scores_and_values(x)
        # apply mask to omega
        num_tokens = x.shape[1]
        omega.masked_fill_(self.mask[:num_tokens, :num_tokens] == 0,
                           -torch.inf)
        # apply softmax and dropout
        attention_weights = torch.softmax(omega, dim=-1)
        attention_weights = self.dropout(attention_weights)
        return attention_weights @ values


class MultiHeadAttention(nn.Module):
    def __init__(self, d_in: int, d_out: int, ctx_length: int, dropout: float,
                 num_heads: int, qkv_bias: bool = False) -> None:
        super().__init__()
        self.head_dim = d_out // num_heads
        self.heads = nn.ModuleList(
            [CausalAttention(d_in, self.head_dim, ctx_length, dropout, qkv_bias)
             for _ in range(num_heads)])
        # final projection layer
        self.out_proj = nn.Linear(d_out, d_out)

    def forward(self, x: Tensor) -> Tensor:
        tensors = [head(x) for head in self.heads]
        concatereted_tens = torch.cat(tensors, dim=-1)
        return self.out_proj(concatereted_tens)
