"""Data preparation and ETL pipeline for FoldGemma.

Provides a PyTorch IterableDataset for parsing Foldcomp databases and writing TFRecords directly from workers.
"""

import logging
import os
import mmap
from typing import Iterator, Tuple
from abc import ABC, abstractmethod

import numpy as np
import tensorflow as tf
import torch
from torch.utils.data import IterableDataset

from foldgemma.data.generate_synthetic import serialize_example
from foldgemma.data.vocabulary import AMINO_ACIDS

logger = logging.getLogger(__name__)




class BaseTFRecordDataset(IterableDataset, ABC):
    """Abstract base dataset for writing TFRecord shards from background workers."""

    def __init__(self, out_dir: str, prefix: str):
        super().__init__()
        self.out_dir = out_dir
        self.prefix = prefix

    @abstractmethod
    def generate_records(self, worker_id: int, num_workers: int) -> Iterator[Tuple[bytes, bytes, np.ndarray]]:
        """Yields tuples of (inputs_aa, targets_3di, plddt_array)."""
        pass

    def __iter__(self) -> Iterator[int]:
        from pathlib import Path
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info is not None else 0
        num_workers = worker_info.num_workers if worker_info is not None else 1

        out_path = Path(self.out_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        shard_path = str(out_path / f"{self.prefix}_shard_{worker_id:05d}.tfrecord")
        
        writer = tf.io.TFRecordWriter(shard_path)
        try:
            for inputs_aa, targets_3di, plddt_array in self.generate_records(worker_id, num_workers):
                if len(inputs_aa) != len(targets_3di) or len(inputs_aa) != len(plddt_array):
                    logger.warning(
                        f"Length mismatch: AA={len(inputs_aa)}, "
                        f"3Di={len(targets_3di)}, pLDDT={len(plddt_array)}"
                    )
                    continue

                serialized = serialize_example(inputs_aa, targets_3di, plddt_array)
                writer.write(serialized)
                yield 1
        finally:
            writer.close()





class FoldseekDataset(BaseTFRecordDataset):
    """ETL Dataset that reads MMseqs2/Foldseek databases directly using mmap."""

    def __init__(self, db_prefix: str, out_dir: str, prefix: str = None):
        from pathlib import Path
        if prefix is None:
            prefix = Path(db_prefix).name
        super().__init__(out_dir, prefix)
        self.db_prefix = db_prefix

    def generate_records(self, worker_id: int, num_workers: int) -> Iterator[Tuple[bytes, bytes, np.ndarray]]:
        # In Foldseek, db_prefix (e.g. afdb50) is the amino acid sequence DB
        # db_prefix + "_ss" is the 3Di sequence DB
        aa_path = self.db_prefix
        ss_path = f"{self.db_prefix}_ss"
        aa_index_path = f"{self.db_prefix}.index"
        ss_index_path = f"{self.db_prefix}_ss.index"

        with open(aa_path, "rb") as f_aa, open(ss_path, "rb") as f_ss:
            mm_aa = mmap.mmap(f_aa.fileno(), 0, access=mmap.ACCESS_READ)
            mm_ss = mmap.mmap(f_ss.fileno(), 0, access=mmap.ACCESS_READ)

            with open(aa_index_path, "r", encoding="ascii") as f_aa_idx, \
                 open(ss_index_path, "r", encoding="ascii") as f_ss_idx:
                
                for i, (line_aa, line_ss) in enumerate(zip(f_aa_idx, f_ss_idx)):
                    if i % num_workers != worker_id:
                        continue
                        
                    key_aa, offset_aa_str, length_aa_str = line_aa.strip().split("\t")
                    key_ss, offset_ss_str, length_ss_str = line_ss.strip().split("\t")
                    
                    if key_aa != key_ss:
                        continue
                        
                    offset_aa, length_aa = int(offset_aa_str), int(length_aa_str)
                    offset_ss, length_ss = int(offset_ss_str), int(length_ss_str)
                    
                    # Strip trailing null bytes and newlines
                    aa_bytes = mm_aa[offset_aa:offset_aa+length_aa].strip(b'\x00\n')
                    ss_bytes = mm_ss[offset_ss:offset_ss+length_ss].strip(b'\x00\n')
                    
                    # Foldseek sequence DBs don't store pLDDT. Synthesize a 100.0 mask 
                    # so that it passes the standard >=70.0 threshold in FoldGemma's loss function.
                    plddt_arr = np.full(len(aa_bytes), 100.0, dtype=np.float32)
                    
                    yield aa_bytes, ss_bytes, plddt_arr





def write_tfrecords_from_foldseek(db_prefix: str, out_dir: str, num_workers: int = 4, prefix: str = None):
    """Executes the Foldseek dataset ETL pipeline."""
    dataset = FoldseekDataset(db_prefix=db_prefix, out_dir=out_dir, prefix=prefix)
    dataloader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=None,
        num_workers=num_workers,
        prefetch_factor=2 if num_workers > 0 else None,
    )
    return sum(count for count in dataloader)
