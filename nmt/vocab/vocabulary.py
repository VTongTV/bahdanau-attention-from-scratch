"""frequency-counting vocabulary with shortlist support."""

from collections import Counter
from pathlib import Path

from nmt.config import BOS, EOS, UNK


class Vocab:
    """maps tokens to ids. keeps the most frequent tokens."""

    def __init__(self, size: int = 30000):
        self.size = size
        self.counter = Counter()
        self.itos = []
        self.stoi = {}
        self.frozen = False

    def count(self, tokens) -> None:
        """count tokens from an iterable of token lists."""
        for line in tokens:
            self.counter.update(line)

    def build(self) -> None:
        """cut the shortlist and assign ids. special tokens come first."""
        specials = [BOS, EOS, UNK]
        self.token = specials + [t for t, _ in self.counter.most_common(self.size - len(specials))]
        self.stoi = {t: i for i, t in enumerate(self.token)}
        self.frozen = True

    def __len__(self) -> int:
        return len(self.token)

    def __contains__(self, token: str) -> bool:
        return token in self.stoi

    def id(self, token: str) -> int:
        """return the id for a token, mapping unknowns to unk."""
        return self.stoi.get(token, self.stoi[UNK])

    def token_of(self, idx: int) -> str:
        return self.token[idx]

    def save(self, path) -> None:
        """write the token list to a text file."""
        Path(path).write_text("\n".join(self.token), encoding="utf-8")

    @classmethod
    def load(cls, path) -> "Vocab":
        """read a token list back into a frozen vocab."""
        tokens = Path(path).read_text(encoding="utf-8").splitlines()
        v = cls(len(tokens))
        v.token = tokens
        v.stoi = {t: i for i, t in enumerate(tokens)}
        v.frozen = True
        return v