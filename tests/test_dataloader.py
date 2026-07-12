import torch
from src.tokenizer import Tokenizer
from src.dataloader import GPTDataLoader


def test_dataloader_sliding_window():
    # 1. Create a dummy tokenizer and vocab
    text = "A B C D E F G H I"
    vocab = {token: i for i, token in enumerate(text.split())}
    vocab["<|unk|>"] = len(vocab)
    tokenizer = Tokenizer(vocab)

    print(f"\nvocab: {vocab}")
    print(f"text: {text}\n")

    # 2. Instantiate dataloader
    # The text has 9 tokens.
    # max_length = 3, stride = 2
    # Windows should be:
    # 1: A B C (target: B C D) -> IDs: 0 1 2 (target: 1 2 3)
    # 2: C D E (target: D E F) -> IDs: 2 3 4 (target: 3 4 5)
    # 3: E F G (target: F G H) -> IDs: 4 5 6 (target: 5 6 7)
    # Total 3 chunks.
    loader = GPTDataLoader(
        text=text,
        tokenizer=tokenizer,
        batch_size=2,
        max_length=3,
        stride=2,
        shuffle=False
    )

    # 3. Get batches
    iterator = iter(loader)

    # First batch (size 2)
    inputs, targets = next(iterator)
    print(f"batch 1 inputs:\n{inputs}")
    print(f"batch 1 targets:\n{targets}\n")
    assert inputs.shape == (2, 3)
    assert targets.shape == (2, 3)

    # Check the first window inside the batch
    assert torch.equal(inputs[0], torch.tensor([0, 1, 2]))
    assert torch.equal(targets[0], torch.tensor([1, 2, 3]))

    # Check the second window
    assert torch.equal(inputs[1], torch.tensor([2, 3, 4]))
    assert torch.equal(targets[1], torch.tensor([3, 4, 5]))

    # Second batch (size 1)
    inputs, targets = next(iterator)
    assert inputs.shape == (1, 3)

    # Check the third window
    assert torch.equal(inputs[0], torch.tensor([4, 5, 6]))
    assert torch.equal(targets[0], torch.tensor([5, 6, 7]))
