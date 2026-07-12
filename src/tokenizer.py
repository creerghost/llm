import re


class Tokenizer:
    def __init__(self, vocab: dict[str, int]):
        self.str2int = vocab
        self.int2str = {i: s for s, i in vocab.items()}

    def encode(self, text: str) -> list[int]:
        splitted_text: list[str] = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        preprocessed = [s.strip() for s in splitted_text if s.strip()]

        preprocessed = [
            token if token in self.str2int else "<|unk|>"
            for token in preprocessed
        ]
        return [self.str2int[token] for token in preprocessed]

    def decode(self, ids: list[int]) -> str:
        return " ".join([self.int2str[i] for i in ids])
