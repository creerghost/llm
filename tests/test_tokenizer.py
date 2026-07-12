from src.tokenizer import Tokenizer
import pytest


def test_tokenizer_encode_known_words() -> None:
    vocab = {"Hello": 0, ",": 1, "world": 2, "!": 3, "<|unk|>": 4}
    tokenizer = Tokenizer(vocab)

    text = "Hello, world!"
    ids = tokenizer.encode(text)
    print(f"\nvocab: {vocab}")
    print(f"text: {text}")
    print(f"ids: {ids}\n")

    assert ids == [0, 1, 2, 3]


def test_tokenizer_encode_unknown_words() -> None:
    vocab = {"Hello": 0, "world": 1, "<|unk|>": 2}
    tokenizer = Tokenizer(vocab)

    # "everyone" is not in vocab, should become <|unk|> (2)
    text = "Hello everyone"
    ids = tokenizer.encode(text)
    print(f"\nvocab: {vocab}")
    print(f"text: {text}")
    print(f"ids: {ids}\n")

    assert ids == [0, 2]


def test_tokenizer_decode() -> None:
    vocab = {"Hello": 0, "world": 1}
    tokenizer = Tokenizer(vocab)

    ids = [0, 1]
    decoded_text = tokenizer.decode(ids)
    print(f"\nvocab: {vocab}")
    print(f"ids: {ids}")
    print(f"decoded text: {decoded_text}\n")

    assert decoded_text == "Hello world"


def test_tokenizer_encode_decode() -> None:
    vocab = {"I": 0, "love": 1, "Python": 2, ".": 3, "<|unk|>": 4}
    tokenizer = Tokenizer(vocab)

    original_text = "I love Python ."

    # Encode then decode should return the same space-separated tokens
    ids = tokenizer.encode(original_text)
    reconstructed_text = tokenizer.decode(ids)
    print(f"\nvocab: {vocab}")
    print(f"original_text: {original_text}")
    print(f"ids: {ids}")
    print(f"reconstructed text: {reconstructed_text}\n")

    assert reconstructed_text == original_text


if __name__ == "__main__":
    from pathlib import Path

    path_obj = Path(__file__)
    print(f"\n=== TESTING {path_obj.stem}.py ===")

    pytest.main(["-v", "-s", "--no-header", "-o",
                 "python_classes=*Suite", __file__])
