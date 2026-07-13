import torch
from src.attention import SelfAttention, CausalAttention, MultiHeadAttention


def test_self_attention_shape() -> None:
    torch.manual_seed(123)

    # Batch size: 2, sequence length: 4, embedding dimension: 6
    batch = torch.rand((2, 4, 6))

    sa = SelfAttention(d_in=6, d_out=6)
    context_vectors = sa(batch)

    print("\nTesting SelfAttention...")
    print(f"Shape: {context_vectors.shape}\n")

    assert context_vectors.shape == (2, 4, 6)


def test_causal_attention_shape() -> None:
    torch.manual_seed(123)

    batch = torch.rand((2, 4, 6))

    ca = CausalAttention(d_in=6, d_out=6, ctx_length=4, dropout=0.0)
    context_vectors = ca(batch)

    print("\nTesting CausalAttention...")
    print(f"Shape: {context_vectors.shape}\n")

    assert context_vectors.shape == (2, 4, 6)


def test_multihead_attention_shape() -> None:
    torch.manual_seed(123)

    batch = torch.rand((2, 4, 6))

    mha = MultiHeadAttention(d_in=6, d_out=6, ctx_length=4, dropout=0.0,
                             num_heads=2)
    mha_vectors = mha(batch)

    print("\nTesting MultiHeadAttention...")
    print(f"Shape: {mha_vectors.shape}\n")

    assert mha_vectors.shape == (2, 4, 6)

