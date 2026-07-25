"""Composite loss function with quality and token masking for FoldGemma."""

import jax
import jax.numpy as jnp
import optax

from foldgemma.data.vocabulary import PAD_ID, UNK_ID


def compute_loss_mask(
    targets: jax.Array,
    plddt: jax.Array | None = None,
    pad_id: int = PAD_ID,
    unk_id: int = UNK_ID,
    plddt_threshold: float = 70.0,
) -> jax.Array:
    """Construct composite binary mask: (targets != pad_id) & (targets != unk_id) & (plddt >= 70.0).

    Args:
        targets: Integer target token IDs tensor.
        plddt: Residue pLDDT confidence scores tensor (optional).
        pad_id: Padding token ID.
        unk_id: Unknown token ID.
        plddt_threshold: Minimum pLDDT score threshold (default: 70.0).

    Returns:
        Float32 binary mask tensor (1.0 for valid residues, 0.0 otherwise).
    """
    valid_target = (targets != pad_id) & (targets != unk_id)
    if plddt is not None:
        mask = valid_target & (plddt >= plddt_threshold)
    else:
        mask = valid_target
    return mask.astype(jnp.float32)


def compute_masked_loss(
    logits: jax.Array,
    targets: jax.Array,
    plddt: jax.Array | None = None,
    pad_id: int = PAD_ID,
    unk_id: int = UNK_ID,
    plddt_threshold: float = 70.0,
) -> jax.Array:
    """Compute composite masked cross entropy loss using optax.

    Args:
        logits: Unnormalized log probabilities of shape (..., vocab_size).
        targets: Integer target token IDs of shape (...).
        plddt: Residue pLDDT confidence scores of shape (...).
        pad_id: Padding token ID.
        unk_id: Unknown token ID.
        plddt_threshold: Minimum pLDDT score threshold (default: 70.0).

    Returns:
        Scalar average loss divided ONLY by sum of valid mask:
        sum(masked_loss) / max(1.0, sum(mask)).
    """
    raw_loss = optax.softmax_cross_entropy_with_integer_labels(logits=logits, labels=targets)
    mask = compute_loss_mask(
        targets=targets,
        plddt=plddt,
        pad_id=pad_id,
        unk_id=unk_id,
        plddt_threshold=plddt_threshold,
    )
    masked_loss = raw_loss * mask
    valid_count = jnp.sum(mask)
    loss = jnp.sum(masked_loss) / jnp.maximum(1.0, valid_count)
    return loss
