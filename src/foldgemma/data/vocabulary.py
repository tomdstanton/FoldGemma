"""Custom character-level vocabulary for Protein AA and 3di structural tokens."""

from typing import Iterable, List

import typing

# 20 standard amino acid characters
AMINO_ACIDS: List[str] = list("ARNDCEGHILKMFPSTWYV")

# 20 3di structural tokens (lowercase)
THREE_DI_TOKENS: List[str] = list("acdefghiklmnpqrstvwy")

# Special tokens
PAD_TOKEN: str = "<pad>"
UNK_TOKEN: str = "<unk>"

PAD_ID: int = 0
UNK_ID: int = 1

# Total padded vocabulary size MUST be 64
VOCAB_SIZE: int = 64


class Protein3diVocabulary:
    """Character-level vocabulary for Protein AA and 3di structural tokens with vocab size 64."""

    def __init__(self) -> None:
        super().__init__()
        # Build token to ID mapping
        self._tokens: List[str] = [PAD_TOKEN, UNK_TOKEN] + AMINO_ACIDS + THREE_DI_TOKENS
        self._byte_to_id: dict[int, int] = {
            ord(c): i for i, c in enumerate(self._tokens) if len(c) == 1
        }
        self._id_to_byte: dict[int, bytes] = {
            i: c.encode("ascii") for i, c in enumerate(self._tokens) if len(c) == 1
        }
        self._char_to_id: dict[str, int] = {token: i for i, token in enumerate(self._tokens)}
        self._id_to_char: dict[int, str] = {i: token for i, token in enumerate(self._tokens)}


    @property
    def _base_vocab_size(self) -> int:
        """Base vocabulary size."""
        return VOCAB_SIZE

    @property
    def vocab_size(self) -> int:
        """Total padded vocabulary size."""
        return VOCAB_SIZE

    @property
    def pad_id(self) -> int:
        """Padding token ID."""
        return PAD_ID

    @property
    def unk_id(self) -> int:
        """Unknown token ID."""
        return UNK_ID

    @property
    def eos_id(self) -> int | None:
        """End of sequence token ID (None if unused)."""
        return None

    def encode(self, s: str | bytes) -> List[int]:
        """Encode Python string or bytes to token IDs."""
        if isinstance(s, bytes):
            return self.encode_bytes(s)
        return [self._char_to_id.get(char, UNK_ID) for char in s]

    def encode_bytes(self, s: bytes) -> List[int]:
        """Encode bytes to token IDs."""
        return [self._byte_to_id.get(b, UNK_ID) for b in s]

    def decode(self, ids: Iterable[int]) -> str:
        """Decode token IDs to Python string."""
        return "".join(self._id_to_char.get(int(i), UNK_TOKEN) for i in ids)

    def decode_bytes(self, ids: Iterable[int]) -> bytes:
        """Decode token IDs to bytes."""
        return b"".join(self._id_to_byte.get(int(i), b"<unk>") for i in ids)



    def __eq__(self, other: object) -> bool:
        return isinstance(other, Protein3diVocabulary) and self.vocab_size == other.vocab_size
