"""Synthetic dummy protein data generator for FoldGemma."""

import random
from pathlib import Path
from typing import Sequence, Tuple

import numpy as np

from foldgemma.data.vocabulary import AMINO_ACIDS, THREE_DI_TOKENS


def generate_synthetic_protein(
    length: int,
    seed: int | None = None,
) -> Tuple[str, str, np.ndarray]:
    """Generate a synthetic protein sample with AA sequence, 3di sequence, and pLDDT array.

    Args:
        length: Sequence length (number of residues).
        seed: Optional random seed.

    Returns:
        Tuple of (inputs_aa, targets_3di, plddt_array).
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    inputs_aa = "".join(random.choices(AMINO_ACIDS, k=length))
    targets_3di = "".join(random.choices(THREE_DI_TOKENS, k=length))

    # Generate pLDDT values in range [30.0, 100.0]
    plddt = np.random.uniform(30.0, 100.0, size=(length,)).astype(np.float32)

    return inputs_aa, targets_3di, plddt


def write_synthetic_dataset(
    output_dir: str | Path,
    num_examples: int = 10,
    min_len: int = 100,
    max_len: int = 1500,
    seed: int = 42,
) -> None:
    """Generate synthetic dataset and write to binary SoA format.

    Args:
        output_dir: Output directory for binary files.
        num_examples: Number of protein samples to generate.
        min_len: Minimum sequence length.
        max_len: Maximum sequence length.
        seed: Random seed for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    offsets = []
    lengths = []
    current_offset = 0

    with open(out_path / "inputs.bin", "wb") as f_in, \
         open(out_path / "targets.bin", "wb") as f_tgt, \
         open(out_path / "plddt.bin", "wb") as f_plddt:
        
        for i in range(num_examples):
            length = random.randint(min_len, max_len)
            inputs, targets, plddt = generate_synthetic_protein(length, seed=seed + i)
            
            in_bytes = inputs.encode("ascii")
            tgt_bytes = targets.encode("ascii")
            
            f_in.write(in_bytes)
            f_tgt.write(tgt_bytes)
            f_plddt.write(plddt.tobytes())
            
            offsets.append(current_offset)
            lengths.append(length)
            current_offset += length

    # Save indices
    np.savez_compressed(
        out_path / "index.npz",
        offsets=np.array(offsets, dtype=np.int64),
        lengths=np.array(lengths, dtype=np.int32),
    )


def main() -> None:
    """Run synthetic data generation CLI."""
    output_dir = "synthetic_data"
    write_synthetic_dataset(output_dir, num_examples=20, min_len=100, max_len=1000, seed=42)
    print(f"Generated synthetic dataset written to {output_dir}")


if __name__ == "__main__":
    main()
