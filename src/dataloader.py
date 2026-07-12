"""
PyTorch Dataset that takes raw text, tokenizes it, and
creates (input, target) pairs using a sliding window.
Then wraps it in a PyTorch DataLoader.
"""

import torch
from torch.utils.data import Dataset, DataLoader
from src.tokenizer import Tokenizer


class GPTDataset(Dataset):
    def __init__(self, text: str, tokenizer: Tokenizer, max_length: int,
                 stride: int) -> None:
        self.input_ids: list[int] = []
        self.target_ids: list[int] = []
        token_ids = tokenizer.encode(text)
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1: i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]


class GPTDataLoader():
    def __init__(self, text: str, tokenizer: Tokenizer, batch_size: int = 4,
                 max_length: int = 256, stride: int = 128,
                 shuffle: bool = True) -> None:
        dataset = GPTDataset(text, tokenizer, max_length, stride)
        self.dataloader = DataLoader(dataset,
                                     batch_size=batch_size,
                                     shuffle=shuffle)

    def __iter__(self):
        return iter(self.dataloader)

    def __len__(self):
        return len(self.dataloader)
