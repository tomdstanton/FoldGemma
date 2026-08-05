"""PyTorch Dataset implementation for FoldGemma using binary memory mapping."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

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

        # Discover shards (either un-sharded data_dir, or sharded subdirs)
        index_files = sorted(list(self.data_dir.glob("*/index.npz")) + list(self.data_dir.glob("index.npz")))
        if not index_files:
            raise FileNotFoundError(f"No index.npz found in {self.data_dir} or its subdirectories.")

        self.shards: list[dict[str, Any]] = []
        for idx_file in index_files:
            shard_dir = idx_file.parent
            idx_data = np.load(idx_file)
            self.shards.append(
                {
                    "dir": shard_dir,
                    "offsets": idx_data["offsets"],
                    "lengths": idx_data["lengths"],
                    "num_samples": len(idx_data["offsets"]),
                }
            )

        self.num_samples = sum(s["num_samples"] for s in self.shards)

        # Build cumulative sums for fast bisect
        self.cumulative_samples = np.cumsum([s["num_samples"] for s in self.shards])

        # We will initialize mmaps lazily in a dictionary mapping shard_idx -> mmaps
        self._mmaps: dict[int, dict[str, Any]] = {}

    def __len__(self) -> int:
        """Return number of samples."""
        return self.num_samples

    def _init_shard_mmap(self, shard_idx: int) -> None:
        if shard_idx not in self._mmaps:
            shard_dir = self.shards[shard_idx]["dir"]
            self._mmaps[shard_idx] = {
                "inputs": np.memmap(shard_dir / "inputs.bin", dtype="uint8", mode="r"),
                "targets": np.memmap(shard_dir / "targets.bin", dtype="uint8", mode="r"),
                "plddt": np.memmap(shard_dir / "plddt.bin", dtype="float32", mode="r"),
            }

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        """Get a single sample by index."""
        shard_idx = int(np.searchsorted(self.cumulative_samples, index, side="right"))
        if shard_idx > 0:
            local_index = index - int(self.cumulative_samples[shard_idx - 1])
        else:
            local_index = index

        self._init_shard_mmap(shard_idx)
        shard = self.shards[shard_idx]
        mmaps = self._mmaps[shard_idx]

        start = shard["offsets"][local_index]
        length = shard["lengths"][local_index]
        end = start + length

        # Load raw bytes and convert to strings
        in_bytes = mmaps["inputs"][start:end].tobytes()
        tgt_bytes = mmaps["targets"][start:end].tobytes()

        # Load pLDDT array (copy to avoid non-writable warnings when converting to Tensor)
        plddt_arr = mmaps["plddt"][start:end].copy()

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
