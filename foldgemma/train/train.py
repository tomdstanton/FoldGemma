"""Training utilities and step functions for FoldGemma."""

from typing import Dict

from flax import nnx
import jax

from foldgemma.train.loss import compute_masked_loss


@nnx.jit(static_argnames=("pad_id", "unk_id", "plddt_threshold"))
def train_step(
    model: nnx.Module,
    optimizer: nnx.Optimizer,
    batch: Dict[str, jax.Array],
    pad_id: int = 0,
    unk_id: int = 1,
    plddt_threshold: float = 70.0,
) -> jax.Array:
    """Perform a single training step.

    Args:
        model: The current model.
        optimizer: The optimizer holding optimizer state.
        batch: Dictionary with 'inputs', 'targets', and 'plddt'.
        pad_id: Padding token ID.
        unk_id: Unknown token ID.
        plddt_threshold: Minimum pLDDT score threshold.

    Returns:
        The scalar loss value.
    """

    def loss_fn(m: nnx.Module) -> jax.Array:
        if "decoder_input_ids" in batch:
            logits = m(
                batch["inputs"],
                decoder_input_ids=batch["decoder_input_ids"],
                plddt=batch.get("plddt"),
                plddt_threshold=plddt_threshold,
            )
        else:
            logits = m(
                batch["inputs"],
                plddt=batch.get("plddt"),
                plddt_threshold=plddt_threshold,
            )
        loss = compute_masked_loss(
            logits,
            batch["targets"],
            batch["plddt"],
            pad_id=pad_id,
            unk_id=unk_id,
            plddt_threshold=plddt_threshold,
        )
        return loss

    loss, grads = nnx.value_and_grad(loss_fn)(model)
    optimizer.update(grads)
    return loss


def main() -> None:
    """Run the training loop on synthetic data."""
    import os

    from foldgemma.api import FoldGemmaTrainer
    from foldgemma.data.generate_synthetic import write_synthetic_tfrecord
    from foldgemma.data.pipeline import FoldGemmaDataPipeline

    tfrecord_path = "dummy.tfrecord"
    if not os.path.exists(tfrecord_path):
        print(f"Generating synthetic dataset at {tfrecord_path}...")
        write_synthetic_tfrecord(tfrecord_path, num_examples=20, min_len=100, max_len=1000)

    print("Initializing Data Pipeline...")
    pipeline = FoldGemmaDataPipeline(tfrecord_path=tfrecord_path, batch_size=4)
    pipeline.register_task()

    print("Initializing Trainer...")
    trainer = FoldGemmaTrainer()

    # Orbax requires absolute paths
    checkpoint_dir = os.path.abspath("./checkpoints")

    # Run for 2 epochs, 5 steps each
    trainer.fit(pipeline, epochs=2, steps_per_epoch=5, checkpoint_dir=checkpoint_dir)


if __name__ == "__main__":
    main()
