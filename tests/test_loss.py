"""Unit tests for composite loss function and loss masking behavior in FastProtT5."""

import torch

from foldgemma.data.vocabulary import PAD_ID, UNK_ID
from foldgemma.loss import MaskedCrossEntropyLoss

def test_loss_mask_logic() -> None:
    """Verify that compute_loss_mask correctly identifies valid vs invalid tokens."""
    targets = torch.tensor([PAD_ID, UNK_ID, 5, 6, 7, 8])
    plddt = torch.tensor([90.0, 90.0, 80.0, 50.0, 70.0, 69.9])

    # Expected valid positions:
    # idx 0: target == PAD_ID (0) -> mask = 0
    # idx 1: target == UNK_ID (1) -> mask = 0
    # idx 2: target == 5, plddt == 80.0 >= 70.0 -> mask = 1
    # idx 3: target == 6, plddt == 50.0 < 70.0 -> mask = 0
    # idx 4: target == 7, plddt == 70.0 >= 70.0 -> mask = 1
    # idx 5: target == 8, plddt == 69.9 < 70.0 -> mask = 0
    expected_mask = torch.tensor([0, 0, 1, 0, 1, 0], dtype=torch.bool)

    loss_fn = MaskedCrossEntropyLoss(pad_id=PAD_ID, unk_id=UNK_ID, plddt_threshold=70.0)
    mask = loss_fn.compute_mask(
        targets=targets,
        plddt=plddt,
    )
    torch.testing.assert_close(mask, expected_mask)


def test_masked_tokens_contribute_zero_to_loss() -> None:
    """Verify that tokens with <pad>, <unk>, or plddt < 70.0 contribute EXACTLY 0.0 to loss sum."""
    vocab_size = 64
    seq_len = 6

    # Targets: [0 (pad), 1 (unk), 5 (valid), 6 (low plddt), 7 (valid border), 8 (low plddt)]
    targets = torch.tensor([PAD_ID, UNK_ID, 5, 6, 7, 8], dtype=torch.long)
    plddt = torch.tensor([95.0, 95.0, 85.0, 40.0, 75.0, 65.0])

    # Construct two different logits matrices:
    # Logits 1 has moderate predictions for all tokens
    logits_1 = torch.zeros((seq_len, vocab_size))

    # Logits 2 has identical predictions for valid positions (idx 2 and idx 4),
    # but wildly incorrect predictions for invalid positions (idx 0, 1, 3, 5)
    logits_2 = torch.zeros((seq_len, vocab_size))
    # Make invalid positions have extreme negative logits for the target class
    logits_2[0, targets[0]] = -100.0
    logits_2[1, targets[1]] = -100.0
    logits_2[3, targets[3]] = -100.0
    logits_2[5, targets[5]] = -100.0

    loss_fn = MaskedCrossEntropyLoss()
    loss_1 = loss_fn(logits=logits_1, targets=targets, plddt=plddt)
    loss_2 = loss_fn(logits=logits_2, targets=targets, plddt=plddt)

    # Invalid positions contribute EXACTLY 0.0 to loss sum, so loss_1 and loss_2 must be identical
    torch.testing.assert_close(loss_1, loss_2, rtol=1e-5, atol=1e-5)

    # Manually compute expected loss for valid positions (idx 2 and idx 4)
    raw_losses = torch.nn.functional.cross_entropy(logits_1, targets, reduction='none')
    expected_loss = (raw_losses[2] + raw_losses[4]) / 2.0

    torch.testing.assert_close(loss_1, expected_loss, rtol=1e-5, atol=1e-5)


def test_all_invalid_tokens_zero_loss() -> None:
    """Verify that when all tokens are masked, loss is 0.0 and does not raise NaN/Inf."""
    vocab_size = 64
    seq_len = 4
    logits = torch.zeros((seq_len, vocab_size))

    # All targets are PAD or UNK or low pLDDT
    targets = torch.tensor([PAD_ID, UNK_ID, PAD_ID, 10], dtype=torch.long)
    plddt = torch.tensor([90.0, 90.0, 90.0, 50.0])

    loss_fn = MaskedCrossEntropyLoss()
    loss = loss_fn(logits=logits, targets=targets, plddt=plddt)
    torch.testing.assert_close(loss, torch.tensor(0.0), rtol=1e-5, atol=1e-5)
    assert not torch.isnan(loss)
    assert not torch.isinf(loss)


def test_compute_loss_mask_plddt_none() -> None:
    """Verify that compute_loss_mask handles plddt=None gracefully."""
    targets = torch.tensor([PAD_ID, UNK_ID, 5, 6, 7, 8])
    expected_mask = torch.tensor([0, 0, 1, 1, 1, 1], dtype=torch.bool)
    loss_fn = MaskedCrossEntropyLoss()
    mask = loss_fn.compute_mask(targets=targets, plddt=None)
    torch.testing.assert_close(mask, expected_mask)

    loss = loss_fn(logits=torch.zeros((6, 64)), targets=targets, plddt=None)
    assert not torch.isnan(loss)
