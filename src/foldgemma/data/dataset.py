"""PyTorch Dataset implementation for FoldGemma using binary memory mapping."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from foldgemma.data.vocabulary import Protein3diVocabulary

logger = logging.getLogger(__name__)


class FoldGemmaDataset(Dataset[dict[str, torch.Tensor]]):
    """Memory-mapped Structure-of-Arrays dataset for FoldGemma."""

    def __init__(
        self,
        data_dir: str | Path,
        vocabulary: Protein3diVocabulary | None = None,
        max_length: int = 2048,
    ) -> None:
        """Initialize dataset from a directory containing binary blobs and index.npz.

        Args:
            data_dir: Directory containing inputs.bin, targets.bin, plddt.bin, index.npz
            vocabulary: Protein3diVocabulary instance for tokenization.
            max_length: Maximum sequence length to truncate to.
        """
        self.data_dir = Path(data_dir)
        self.vocabulary = vocabulary or Protein3diVocabulary()
        self.max_length = max_length

        # Load indices
        self.index = np.load(self.data_dir / "index.npz")
        self.offsets = self.index["offsets"]
        self.lengths = self.index["lengths"]
        self.num_samples = len(self.offsets)

        # Lazy memmaps initialized per-worker
        self._inputs_mmap = None
        self._targets_mmap = None
        self._plddt_mmap = None

    def __len__(self) -> int:
        """Return number of samples."""
        return self.num_samples

    def _init_memmaps(self) -> None:
        if self._inputs_mmap is None:
            self._inputs_mmap = np.memmap(self.data_dir / "inputs.bin", dtype="uint8", mode="r")
            self._targets_mmap = np.memmap(self.data_dir / "targets.bin", dtype="uint8", mode="r")
            self._plddt_mmap = np.memmap(self.data_dir / "plddt.bin", dtype="float32", mode="r")

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        """Get a single sample by index."""
        self._init_memmaps()
        assert self._inputs_mmap is not None
        assert self._targets_mmap is not None
        assert self._plddt_mmap is not None

        start = self.offsets[index]
        length = self.lengths[index]
        end = start + length

        # Load raw bytes and convert to strings
        in_bytes = self._inputs_mmap[start:end].tobytes()
        tgt_bytes = self._targets_mmap[start:end].tobytes()

        # Load pLDDT array (copy to avoid non-writable warnings when converting to Tensor)
        plddt_arr = self._plddt_mmap[start:end].copy()

        inputs_str = in_bytes.decode("ascii")
        targets_str = tgt_bytes.decode("ascii")

        # Tokenize (CPU side)
        input_ids = self.vocabulary.encode(inputs_str)[: self.max_length]
        target_ids = self.vocabulary.encode(targets_str)[: self.max_length]
        plddt_tensor = torch.from_numpy(plddt_arr)[: self.max_length]

        input_ids_tensor = torch.tensor(input_ids, dtype=torch.long)
        target_ids_tensor = torch.tensor(target_ids, dtype=torch.long)

        return {
            "input_ids": input_ids_tensor,
            "target_ids": target_ids_tensor,
            "plddt": plddt_tensor,
        }


class DataCollatorForFoldGemma:
    """Collator that dynamically pads batches to the maximum length in the batch."""

    def __init__(self, vocabulary: Protein3diVocabulary | None = None) -> None:
        """Initialize collator."""
        self.vocabulary = vocabulary or Protein3diVocabulary()

    def __call__(self, features: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        """Pad and collate a batch."""
        batch_size = len(features)
        max_len = max(f["input_ids"].size(0) for f in features)

        input_ids = torch.full((batch_size, max_len), self.vocabulary.pad_id, dtype=torch.long)
        target_ids = torch.full((batch_size, max_len), self.vocabulary.pad_id, dtype=torch.long)
        plddt = torch.zeros((batch_size, max_len), dtype=torch.float32)

        for i, feature in enumerate(features):
            length = feature["input_ids"].size(0)
            input_ids[i, :length] = feature["input_ids"]
            target_ids[i, :length] = feature["target_ids"]
            plddt[i, :length] = feature["plddt"]

        return {
            "input_ids": input_ids,
            "target_ids": target_ids,
            "plddt": plddt,
        }
