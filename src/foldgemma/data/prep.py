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

THREE_TO_ONE = {
    "ALA": "A", "CYS": "C", "ASP": "D", "GLU": "E", "PHE": "F",
    "GLY": "G", "HIS": "H", "ILE": "I", "LYS": "K", "LEU": "L",
    "MET": "M", "ASN": "N", "PRO": "P", "GLN": "Q", "ARG": "R",
    "SER": "S", "THR": "T", "VAL": "V", "TRP": "W", "TYR": "Y",
    "SEC": "U", "PYL": "O"
}

def parse_pdb_string(pdb_string: str) -> Tuple[str, np.ndarray]:
    """Extract AA sequence and pLDDT (B-factor) from CA atoms in PDB string."""
    aa_list = []
    plddts = []
    for line in pdb_string.splitlines():
        if line.startswith("ATOM ") and line[12:16] == " CA ":
            res_name = line[17:20].strip()
            aa = THREE_TO_ONE.get(res_name, "X")
            try:
                plddt = float(line[60:66].strip())
            except ValueError:
                plddt = 0.0
            aa_list.append(aa)
            plddts.append(plddt)
    return "".join(aa_list), np.array(plddts, dtype=np.float32)


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
                            
                        name, targets_3di = parts
                        
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
