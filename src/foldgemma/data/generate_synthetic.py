"""Synthetic dummy protein data generator for FoldGemma."""

import random
from typing import Sequence, Tuple

import numpy as np
import tensorflow as tf

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


def serialize_example(inputs: str | bytes, targets: str | bytes, plddt: Sequence[float] | np.ndarray) -> bytes:
    """Serialize inputs, targets, and plddt array into a TFRecord Example binary string."""
    in_bytes = inputs if isinstance(inputs, bytes) else inputs.encode("utf-8")
    tgt_bytes = targets if isinstance(targets, bytes) else targets.encode("utf-8")
    
    feature = {
        "inputs": tf.train.Feature(bytes_list=tf.train.BytesList(value=[in_bytes])),
        "targets": tf.train.Feature(bytes_list=tf.train.BytesList(value=[tgt_bytes])),
        "plddt": tf.train.Feature(float_list=tf.train.FloatList(value=list(plddt))),
    }
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()


def write_synthetic_tfrecord(
    output_path: str,
    num_examples: int = 10,
    min_len: int = 100,
    max_len: int = 1500,
    seed: int = 42,
) -> None:
    """Generate synthetic dataset and write to a TFRecord file.

    Args:
        output_path: Output TFRecord file path.
        num_examples: Number of protein samples to generate.
        min_len: Minimum sequence length.
        max_len: Maximum sequence length.
        seed: Random seed for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)

    writer = tf.io.TFRecordWriter(output_path)
    try:
        for i in range(num_examples):
            length = random.randint(min_len, max_len)
            inputs, targets, plddt = generate_synthetic_protein(length, seed=seed + i)
            serialized = serialize_example(inputs, targets, plddt)
            writer.write(serialized)
    finally:
        writer.close()


def main() -> None:
    """Run synthetic data generation CLI."""
    output_file = "synthetic_data.tfrecord"
    write_synthetic_tfrecord(output_file, num_examples=20, min_len=100, max_len=1000, seed=42)
    print(f"Generated synthetic dataset written to {output_file}")


if __name__ == "__main__":
    main()
