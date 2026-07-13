import torch
from src.model import GPTModel


GPT_CONFIG_124M = {
    "vocab_size": 50257,
    "context_length": 1024,
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.1,
    "qkv_bias": False
}


def test_gpt_model_forward() -> None:
    torch.manual_seed(123)

    model = GPTModel(GPT_CONFIG_124M)
    model.eval()

    # Batch size 2, Sequence length 4
    dummy_input = torch.tensor([[1, 5, 8, 2], [3, 9, 7, 4]])
    logits = model(dummy_input)

    print("\nTesting GPTModel Forward Pass...")
    print(f"Logits shape: {logits.shape}\n")

    assert logits.shape == (2, 4, 50257)


def test_gpt_model_generate() -> None:
    torch.manual_seed(123)

    model = GPTModel(GPT_CONFIG_124M)
    model.eval()

    start_context = torch.tensor([[1]])
    generated_ids = model.generate(
        idx=start_context, 
        max_new_tokens=10, 
        ctx_size=GPT_CONFIG_124M["context_length"]
    )

    print("\nTesting GPTModel Generate Function...")
    print(f"Generated sequence: {generated_ids}")
    print(f"Generated sequence shape: {generated_ids.shape}\n")

    # Starting token (1) + 10 new tokens = 11 tokens
    assert generated_ids.shape == (1, 11)
