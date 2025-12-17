import json
from collections import Counter


class CharTokenizer:
    """
    Simple character-level tokenizer.
    Designed for from-scratch LLM training.
    """

    def __init__(self, text: str):
        chars = sorted(list(set(text)))
        self.vocab_size = len(chars)

        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}

    def encode(self, s: str):
        """
        Encode a string into a list of token IDs.
        """
        return [self.stoi[c] for c in s]

    def decode(self, tokens):
        """
        Decode a list of token IDs back into a string.
        """
        return "".join(self.itos[t] for t in tokens)

    def save(self, path: str):
        """
        Save tokenizer to disk.
        """
        with open(path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "stoi": self.stoi,
                    "itos": self.itos,
                    "vocab_size": self.vocab_size,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

    @classmethod
    def load(cls, path: str):
        """
        Load tokenizer from disk.
        """
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        tokenizer = cls.__new__(cls)
        tokenizer.stoi = data["stoi"]
        tokenizer.itos = {int(k): v for k, v in data["itos"].items()}
        tokenizer.vocab_size = data["vocab_size"]
        return tokenizer


def build_tokenizer_from_file(path: str):
    """
    Utility function to build a tokenizer from a text file.
    """
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    return CharTokenizer(text)
