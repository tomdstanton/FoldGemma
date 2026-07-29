"""Data pipeline module for FoldGemma.

Includes TFRecord deserialization, static length bucketing, and SeqIO task configuration.
"""

from typing import Any, Dict, Sequence, Tuple

import tensorflow as tf

from foldgemma.data.vocabulary import Protein3diVocabulary

STATIC_BUCKETS: Tuple[int, ...] = (512, 1024, 2048)


def deserialize_example(serialized_proto: tf.Tensor) -> Dict[str, tf.Tensor]:
    """Deserialize a TFRecord Example binary string tensor into input features.

    Args:
        serialized_proto: Scalar string tensor containing TFRecord Example.

    Returns:
        Dict with keys:
            'inputs': scalar string tensor (AA sequence)
            'targets': scalar string tensor (3di sequence)
            'plddt': 1D float32 tensor (pLDDT per residue)
    """
    feature_description = {
        "inputs": tf.io.FixedLenFeature([], tf.string),
        "targets": tf.io.FixedLenFeature([], tf.string),
        "plddt": tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
    }
    return tf.io.parse_single_example(serialized_proto, feature_description)


def pad_to_static_bucket(
    example: Dict[str, Any],
    vocabulary: Protein3diVocabulary | None = None,
    buckets: Sequence[int] = STATIC_BUCKETS,
) -> Dict[str, tf.Tensor]:
    """Tokenize and pad/truncate inputs, targets, and plddt to static length buckets.

    Args:
        example: Dict containing 'inputs', 'targets', and 'plddt'.
        vocabulary: Protein3diVocabulary instance. If None, instantiates default.
        buckets: Sequence of static length bucket sizes (default: (512, 1024, 2048)).

    Returns:
        Dict containing tokenized and padded 'inputs', 'targets', and 'plddt' tensors.
    """
    if vocabulary is None:
        vocabulary = Protein3diVocabulary()

    inputs_str = example["inputs"]
    targets_str = example["targets"]
    plddt = example["plddt"]

    # Tokenize strings to int32 ID tensors
    input_ids: Any = vocabulary.encode_tf(inputs_str)
    target_ids: Any = vocabulary.encode_tf(targets_str)
    plddt_tensor: Any = tf.cast(plddt, tf.float32)

    max_bucket = buckets[-1]
    input_ids = input_ids[:max_bucket]
    target_ids = target_ids[:max_bucket]
    plddt_tensor = plddt_tensor[:max_bucket]

    seq_len = tf.shape(input_ids)[0]

    # Determine static bucket length (512, 1024, 2048)
    bucket_size = tf.constant(buckets[-1], dtype=tf.int32)
    for b in reversed(buckets[:-1]):
        bucket_size = tf.where(seq_len <= b, tf.constant(b, dtype=tf.int32), bucket_size)

    pad_len = bucket_size - seq_len

    padded_inputs = tf.pad(input_ids, [[0, pad_len]], constant_values=vocabulary.pad_id)
    padded_targets = tf.pad(target_ids, [[0, pad_len]], constant_values=vocabulary.pad_id)
    padded_plddt = tf.pad(plddt_tensor, [[0, pad_len]], constant_values=0.0)

    return {
        "inputs": padded_inputs,
        "targets": padded_targets,
        "plddt": padded_plddt,
    }


def get_dataset_from_tfrecord(
    tfrecord_path: str | Sequence[str],
    vocabulary: Protein3diVocabulary | None = None,
    buckets: Sequence[int] = STATIC_BUCKETS,
) -> tf.data.Dataset:
    """Build a tf.data.Dataset from TFRecord file(s) with static bucket padding.

    Args:
        tfrecord_path: Path(s) to TFRecord file(s).
        vocabulary: Protein3diVocabulary instance.
        buckets: Sequence of static length bucket sizes.

    Returns:
        tf.data.Dataset yielding dicts of padded 'inputs', 'targets', and 'plddt'.
    """
    dataset = tf.data.TFRecordDataset(tfrecord_path, num_parallel_reads=tf.data.AUTOTUNE)
    dataset = dataset.map(deserialize_example, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.map(
        lambda ex: pad_to_static_bucket(ex, vocabulary=vocabulary, buckets=buckets),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    return dataset


def _dummy_dataset_fn(
    split: str,
    shuffle_files: bool = False,
    seed: int | None = None,
) -> tf.data.Dataset:
    return tf.data.Dataset.from_tensor_slices(
        {
            "inputs": ["MKTAY"],
            "targets": ["dpdpd"],
            "plddt": [[90.0, 90.0, 90.0, 90.0, 90.0]],
        }
    )


class FoldGemmaDataPipeline:
    """OOP encapsulation of the vocabulary and dataset loading logic."""

    def __init__(
        self,
        tfrecord_path: str | Sequence[str] = "dummy.tfrecord",
        seqio_mixture: str | None = None,
        buckets: Sequence[int] = STATIC_BUCKETS,
        vocabulary: Protein3diVocabulary | None = None,
        batch_size: int = 32,
    ) -> None:
        self.tfrecord_path = tfrecord_path
        self.seqio_mixture = seqio_mixture
        self.buckets = buckets
        self.vocabulary = vocabulary or Protein3diVocabulary()
        self.batch_size = batch_size

    def get_train_dataset(self) -> tf.data.Dataset:
        """Get batched dataset for training."""
        dataset = get_dataset_from_tfrecord(
            self.tfrecord_path, vocabulary=self.vocabulary, buckets=self.buckets
        )
        # Group identically padded sequences into the same batch
        return dataset.bucket_by_sequence_length(
            element_length_func=lambda ex: tf.shape(ex["inputs"])[0],
            bucket_boundaries=[self.buckets[0] + 1, self.buckets[1] + 1],
            bucket_batch_sizes=[self.batch_size] * len(self.buckets),
        ).prefetch(tf.data.AUTOTUNE)

