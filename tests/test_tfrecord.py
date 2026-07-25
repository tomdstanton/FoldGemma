"""Unit tests for TFRecord generation, deserialization, and static bucket dataset pipeline."""

import tempfile
from typing import Any, Dict

import tensorflow as tf

from foldgemma.data.generate_synthetic import (
    generate_synthetic_protein,
    serialize_example,
)
from foldgemma.data.pipeline import (
    STATIC_BUCKETS,
    get_dataset_from_tfrecord,
    pad_to_static_bucket,
)
from foldgemma.data.vocabulary import PAD_ID, Protein3diVocabulary


def test_vocabulary_basics() -> None:
    """Test Protein3diVocabulary size, pad/unk IDs, and encoding/decoding."""
    vocab = Protein3diVocabulary()
    assert vocab.vocab_size == 64
    assert vocab.pad_id == PAD_ID
    assert vocab.unk_id == 1

    # Test encoding/decoding AA and 3di tokens
    aa_str = "MKTAY"
    encoded_aa = vocab._encode(aa_str)
    decoded_aa = vocab._decode(encoded_aa)
    assert decoded_aa == aa_str

    three_di_str = "dpdpd"
    encoded_3di = vocab._encode(three_di_str)
    decoded_3di = vocab._decode(encoded_3di)
    assert decoded_3di == three_di_str


def test_tfrecord_static_length_buckets() -> None:
    """Generate synthetic data, write to .tfrecord, and verify static length shapes."""
    vocab = Protein3diVocabulary()

    # Create samples targeting each bucket: 512, 1024, 2048
    test_lengths = [200, 750, 1500]
    expected_bucket_shapes = [512, 1024, 2048]

    with tempfile.NamedTemporaryFile(suffix=".tfrecord", delete=True) as tmp_file:
        tmp_path = tmp_file.name

        writer = tf.io.TFRecordWriter(tmp_path)
        try:
            for i, length in enumerate(test_lengths):
                inputs, targets, plddt = generate_synthetic_protein(length, seed=42 + i)
                serialized = serialize_example(inputs, targets, plddt)
                writer.write(serialized)
        finally:
            writer.close()

        # Read back dataset through pipeline
        dataset = get_dataset_from_tfrecord(tmp_path, vocabulary=vocab, buckets=STATIC_BUCKETS)
        samples = list(dataset.as_numpy_iterator())

        assert len(samples) == len(test_lengths)

        for i, (sample, expected_shape) in enumerate(zip(samples, expected_bucket_shapes)):
            inputs_tensor = sample["inputs"]
            targets_tensor = sample["targets"]
            plddt_tensor = sample["plddt"]

            # Assert tensor shapes match static length buckets
            assert inputs_tensor.shape == (expected_shape,), (
                f"Sample {i}: inputs shape {inputs_tensor.shape} != ({expected_shape},)"
            )
            assert targets_tensor.shape == (expected_shape,), (
                f"Sample {i}: targets shape {targets_tensor.shape} != ({expected_shape},)"
            )
            assert plddt_tensor.shape == (expected_shape,), (
                f"Sample {i}: plddt shape {plddt_tensor.shape} != ({expected_shape},)"
            )

            # Check that padded positions contain PAD_ID / 0.0
            orig_len = test_lengths[i]
            assert (inputs_tensor[orig_len:] == vocab.pad_id).all()
            assert (targets_tensor[orig_len:] == vocab.pad_id).all()
            assert (plddt_tensor[orig_len:] == 0.0).all()


def test_pad_to_static_bucket_direct() -> None:
    """Direct test of pad_to_static_bucket function for a given example."""
    vocab = Protein3diVocabulary()
    example: Dict[str, Any] = {
        "inputs": tf.constant("ACDEF"),
        "targets": tf.constant("acdef"),
        "plddt": tf.constant([85.0, 90.0, 65.0, 75.0, 95.0], dtype=tf.float32),
    }

    padded = pad_to_static_bucket(example, vocabulary=vocab, buckets=(512, 1024, 2048))
    assert padded["inputs"].shape == (512,)
    assert padded["targets"].shape == (512,)
    assert padded["plddt"].shape == (512,)
