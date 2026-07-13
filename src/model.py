from .attention import MultiHeadAttention
import torch
from torch import nn, Tensor


class FeedForward(nn.Module):
    def __init__(self, d_in: int, d_out: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(d_in, 4 * d_in),
            nn.GELU(),
            nn.Linear(4 * d_in, d_out)
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.layers(x)


class Transformer(nn.Module):
    def __init__(self, d_in: int, d_out: int, ctx_length: int, dropout: float,
                 num_heads: int, qkv_bias: bool = False) -> None:
        super().__init__()
        self.att = MultiHeadAttention(d_in, d_in, ctx_length, dropout,
                                      num_heads, qkv_bias)
        self.ff = FeedForward(d_in, d_in)
        self.norm1 = nn.LayerNorm(d_in)
        self.norm2 = nn.LayerNorm(d_in)
        self.drop_shortcut = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.drop_shortcut(self.att(self.norm1(x)))
        x = x + self.drop_shortcut(self.ff(self.norm2(x)))
        return x

