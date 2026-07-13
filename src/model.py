from .attention import MultiHeadAttention
from torch import nn, Tensor
import torch
from typing import Any


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


class GPTModel(nn.Module):
    def __init__(self, cfg: dict[str, Any]) -> None:
        super().__init__()
        self.vocab_size = cfg["vocab_size"]
        self.emb_dim = cfg["emb_dim"]
        self.n_heads = cfg["n_heads"]
        self.n_layers = cfg["n_layers"]
        self.drop_rate = cfg["drop_rate"]
        self.qkv_bias = cfg["qkv_bias"]
        self.ctx_length = cfg["context_length"]

        self.tok_emb = nn.Embedding(self.vocab_size, self.emb_dim)
        self.pos_emb = nn.Embedding(self.ctx_length, self.emb_dim)
        self.drop_emb = nn.Dropout(self.drop_rate)

        self.trf_blocks = nn.Sequential(*[
            Transformer(
                d_in=self.emb_dim,
                d_out=self.emb_dim,
                ctx_length=self.ctx_length,
                dropout=self.drop_rate,
                num_heads=self.n_heads,
                qkv_bias=self.qkv_bias
            )
            for _ in range(self.n_layers)
        ])

        self.final_norm = nn.LayerNorm(self.emb_dim)
        self.out_head = nn.Linear(self.emb_dim, self.vocab_size, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        x = self.drop_emb(self.tok_emb(x) +
                          self.pos_emb(torch.arange(x.shape[1],
                                       device=x.device)))
        return self.out_head(self.final_norm(self.trf_blocks(x)))

    def generate(self, idx: Tensor, max_new_tokens: int,
                 ctx_size: int) -> Tensor:
        for _ in range(max_new_tokens):
            idx_cropped = idx[:, -ctx_size:]
            logits = self.forward(idx_cropped)
            # get logits for the very last token
            logits = logits[:, -1, :]
            # find a token with the highest probability
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)
            # append a token to idx
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

