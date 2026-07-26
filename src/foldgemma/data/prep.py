"""Data preparation and ETL pipeline for FoldGemma.

Provides a PyTorch IterableDataset for parsing Foldcomp databases and writing TFRecords directly from workers.
"""

import logging
import os
from typing import Iterator, Tuple

import foldcomp
import numpy as np
import tensorflow as tf
import torch
from torch.utils.data import IterableDataset

from foldgemma.data.generate_synthetic import serialize_example
from foldgemma.data.vocabulary import AMINO_ACIDS

logger = logging.getLogger(__name__)

import numba as nb

@nb.njit
def _three_to_one(b1: int, b2: int, b3: int) -> int:
    if b1 == 65 and b2 == 76 and b3 == 65: return 65 # ALA -> A
    if b1 == 67 and b2 == 89 and b3 == 83: return 67 # CYS -> C
    if b1 == 65 and b2 == 83 and b3 == 80: return 68 # ASP -> D
    if b1 == 71 and b2 == 76 and b3 == 85: return 69 # GLU -> E
    if b1 == 80 and b2 == 72 and b3 == 69: return 70 # PHE -> F
    if b1 == 71 and b2 == 76 and b3 == 89: return 71 # GLY -> G
    if b1 == 72 and b2 == 73 and b3 == 83: return 72 # HIS -> H
    if b1 == 73 and b2 == 76 and b3 == 69: return 73 # ILE -> I
    if b1 == 76 and b2 == 89 and b3 == 83: return 75 # LYS -> K
    if b1 == 76 and b2 == 69 and b3 == 85: return 76 # LEU -> L
    if b1 == 77 and b2 == 69 and b3 == 84: return 77 # MET -> M
    if b1 == 65 and b2 == 83 and b3 == 78: return 78 # ASN -> N
    if b1 == 80 and b2 == 82 and b3 == 79: return 80 # PRO -> P
    if b1 == 71 and b2 == 76 and b3 == 78: return 81 # GLN -> Q
    if b1 == 65 and b2 == 82 and b3 == 71: return 82 # ARG -> R
    if b1 == 83 and b2 == 69 and b3 == 82: return 83 # SER -> S
    if b1 == 84 and b2 == 72 and b3 == 82: return 84 # THR -> T
    if b1 == 86 and b2 == 65 and b3 == 76: return 86 # VAL -> V
    if b1 == 84 and b2 == 82 and b3 == 80: return 87 # TRP -> W
    if b1 == 84 and b2 == 89 and b3 == 82: return 89 # TYR -> Y
    if b1 == 83 and b2 == 69 and b3 == 67: return 85 # SEC -> U
    if b1 == 80 and b2 == 89 and b3 == 76: return 79 # PYL -> O
    return 88 # X

@nb.njit
def _parse_pdb_bytes(data: np.ndarray):
    n = len(data)
    max_res = n // 80 + 1
    aa_out = np.empty(max_res, dtype=np.uint8)
    plddt_out = np.empty(max_res, dtype=np.float32)
    
    count = 0
    i = 0
    while i < n:
        if i + 66 <= n and data[i] == 65 and data[i+1] == 84 and data[i+2] == 79 and data[i+3] == 77 and data[i+4] == 32:
            if data[i+12] == 32 and data[i+13] == 67 and data[i+14] == 65 and data[i+15] == 32:
                b1 = data[i+17]
                b2 = data[i+18]
                b3 = data[i+19]
                
                aa = _three_to_one(b1, b2, b3)
                
                val = 0.0
                div = 1.0
                in_decimal = False
                for j in range(i+60, i+66):
                    c = data[j]
                    if c == 32: 
                        continue
                    if c == 46: 
                        in_decimal = True
                        continue
                    if 48 <= c <= 57:
                        if not in_decimal:
                            val = val * 10 + (c - 48)
                        else:
                            div *= 10
                            val = val + (c - 48) / div
                
                aa_out[count] = aa
                plddt_out[count] = val
                count += 1
                
        while i < n and data[i] != 10:
            i += 1
        i += 1
        
    return aa_out[:count], plddt_out[:count]


def parse_pdb_string(pdb_string: str) -> Tuple[bytes, np.ndarray]:
    """Extract AA sequence and pLDDT (B-factor) from CA atoms using high-speed numba JIT."""
    pdb_bytes = np.frombuffer(pdb_string.encode('ascii'), dtype=np.uint8)
    aa_arr, plddt_arr = _parse_pdb_bytes(pdb_bytes)
    return aa_arr.tobytes(), plddt_arr


class SteineggerLabDataset(IterableDataset):
    """ETL Dataset that reads Foldcomp FCZ and writes TFRecord shards from background workers."""

    def __init__(self, tsv_path: str, fcz_path: str, out_dir: str):
        super().__init__()
        self.tsv_path = tsv_path
        self.fcz_path = fcz_path
        self.out_dir = out_dir

    def __iter__(self) -> Iterator[int]:
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info is not None else 0
        num_workers = worker_info.num_workers if worker_info is not None else 1

        os.makedirs(self.out_dir, exist_ok=True)
        shard_path = os.path.join(self.out_dir, f"afdb_train_shard_{worker_id:05d}.tfrecord")
        
        # Count lines to figure out our chunk bounds (roughly)
        # Note: In a true massive dataset, you'd want to memory map or pre-index the TSV.
        # For simplicity, we step through the file and yield for our worker ID.
        
        with foldcomp.open(self.fcz_path) as db:
            writer = tf.io.TFRecordWriter(shard_path)
            try:
                with open(self.tsv_path, "r", encoding="utf-8") as f:
                    for i, line in enumerate(f):
                        if i % num_workers != worker_id:
                            continue
                            
                        parts = line.strip().split("\t")
                        if len(parts) != 2:
                            continue
                            
                        name, targets_3di_str = parts
                        targets_3di = targets_3di_str.encode('ascii')
                        
                        try:
                            pdb_string = db[name]
                        except KeyError:
                            continue

                        inputs_aa, plddt_array = parse_pdb_string(pdb_string)

                        if len(inputs_aa) != len(targets_3di) or len(inputs_aa) != len(plddt_array):
                            logger.warning(
                                f"Length mismatch for {name}: AA={len(inputs_aa)}, "
                                f"3Di={len(targets_3di)}, pLDDT={len(plddt_array)}"
                            )
                            continue

                        serialized = serialize_example(inputs_aa, targets_3di, plddt_array)
                        writer.write(serialized)
                        yield 1 # Signal that one record was successfully written
                        
            finally:
                writer.close()


def write_tfrecords_from_dataset(tsv_path: str, fcz_path: str, out_dir: str, num_workers: int = 4):
    """Executes the dataset ETL pipeline."""
    dataset = SteineggerLabDataset(tsv_path=tsv_path, fcz_path=fcz_path, out_dir=out_dir)
    dataloader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=None, # IterableDataset doesn't need batching here
        num_workers=num_workers,
        prefetch_factor=2 if num_workers > 0 else None,
    )
    
    total_written = 0
    for count in dataloader:
        total_written += count
        
    return total_written
