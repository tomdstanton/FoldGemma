"""Unit tests for composite loss function and loss masking behavior in FastProtT5."""

import jax.numpy as jnp
import optax

from foldgemma.data.vocabulary import PAD_ID, UNK_ID
from foldgemma.train.loss import compute_loss_mask, compute_masked_loss


def test_loss_mask_logic() -> None:
    """Verify that compute_loss_mask correctly identifies valid vs invalid tokens."""
    targets = jnp.array([PAD_ID, UNK_ID, 5, 6, 7, 8])
    plddt = jnp.array([90.0, 90.0, 80.0, 50.0, 70.0, 69.9])

    # Expected valid positions:
    # idx 0: target == PAD_ID (0) -> mask = 0
    # idx 1: target == UNK_ID (1) -> mask = 0
    # idx 2: target == 5, plddt == 80.0 >= 70.0 -> mask = 1
    # idx 3: target == 6, plddt == 50.0 < 70.0 -> mask = 0
    # idx 4: target == 7, plddt == 70.0 >= 70.0 -> mask = 1
    # idx 5: target == 8, plddt == 69.9 < 70.0 -> mask = 0
    expected_mask = jnp.array([0.0, 0.0, 1.0, 0.0, 1.0, 0.0])

    mask = compute_loss_mask(
        targets=targets,
        plddt=plddt,
        pad_id=PAD_ID,
        unk_id=UNK_ID,
        plddt_threshold=70.0,
    )
    assert jnp.allclose(mask, expected_mask)


def test_masked_tokens_contribute_zero_to_loss() -> None:
    """Verify that tokens with <pad>, <unk>, or plddt < 70.0 contribute EXACTLY 0.0 to loss sum."""
    vocab_size = 64
    seq_len = 6

    # Targets: [0 (pad), 1 (unk), 5 (valid), 6 (low plddt), 7 (valid border), 8 (low plddt)]
    targets = jnp.array([PAD_ID, UNK_ID, 5, 6, 7, 8])
    plddt = jnp.array([95.0, 95.0, 85.0, 40.0, 75.0, 65.0])

    # Construct two different logits matrices:
    # Logits 1 has moderate predictions for all tokens
    logits_1 = jnp.zeros((seq_len, vocab_size))

    # Logits 2 has identical predictions for valid positions (idx 2 and idx 4),
    # but wildly incorrect predictions for invalid positions (idx 0, 1, 3, 5)
    logits_2 = jnp.zeros((seq_len, vocab_size))
    # Make invalid positions have extreme negative logits for the target class
    logits_2 = logits_2.at[0, targets[0]].set(-100.0)
    logits_2 = logits_2.at[1, targets[1]].set(-100.0)
    logits_2 = logits_2.at[3, targets[3]].set(-100.0)
    logits_2 = logits_2.at[5, targets[5]].set(-100.0)

    loss_1 = compute_masked_loss(logits=logits_1, targets=targets, plddt=plddt)
    loss_2 = compute_masked_loss(logits=logits_2, targets=targets, plddt=plddt)

    # Invalid positions contribute EXACTLY 0.0 to loss sum, so loss_1 and loss_2 must be identical
    assert jnp.isclose(loss_1, loss_2, atol=1e-6)

    # Manually compute expected loss for valid positions (idx 2 and idx 4)
    raw_losses = optax.softmax_cross_entropy_with_integer_labels(logits=logits_1, labels=targets)
    expected_loss = (raw_losses[2] + raw_losses[4]) / 2.0

    assert jnp.isclose(loss_1, expected_loss, atol=1e-6)


def test_all_invalid_tokens_zero_loss() -> None:
    """Verify that when all tokens are masked, loss is 0.0 and does not raise NaN/Inf."""
    vocab_size = 64
    seq_len = 4
    logits = jnp.zeros((seq_len, vocab_size))

    # All targets are PAD or UNK or low pLDDT
    targets = jnp.array([PAD_ID, UNK_ID, PAD_ID, 10])
    plddt = jnp.array([90.0, 90.0, 90.0, 50.0])

    loss = compute_masked_loss(logits=logits, targets=targets, plddt=plddt)
    assert jnp.isclose(loss, 0.0)
    assert not jnp.isnan(loss)
    assert not jnp.isinf(loss)


def test_compute_loss_mask_plddt_none() -> None:
    """Verify that compute_loss_mask handles plddt=None gracefully."""
    targets = jnp.array([PAD_ID, UNK_ID, 5, 6, 7, 8])
    expected_mask = jnp.array([0.0, 0.0, 1.0, 1.0, 1.0, 1.0])
    mask = compute_loss_mask(targets=targets, plddt=None)
    assert jnp.allclose(mask, expected_mask)

    loss = compute_masked_loss(logits=jnp.zeros((6, 64)), targets=targets, plddt=None)
    assert not jnp.isnan(loss)

