import re


class WordTokenizer:
    """
    Simple word-level tokenizer.
    Splits on whitespace and punctuation.
    """

    def __init__(self, text: str):
        tokens = self._tokenize(text)
        self.vocab = sorted(set(tokens))
        self.stoi = {t: i for i, t in enumerate(self.vocab)}
        self.itos = {i: t for t, i in self.stoi.items()}

    def _tokenize(self, text: str):
        # keep punctuation as separate tokens
        return re.findall(r"\w+|[^\w\s]", text)

    def encode(self, text: str):
        tokens = self._tokenize(text)
        return [self.stoi[t] for t in tokens if t in self.stoi]

    def decode(self, ids):
        return " ".join(self.itos[i] for i in ids)

    @property
    def vocab_size(self):
        return len(self.vocab)
