"""Script to convert a JAX/Flax checkpoint to PyTorch SafeTensors."""

import argparse
import os

from foldgemma.api import FoldGemmaTrainer


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Convert FastProtT5 Flax checkpoint to PyTorch")
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="./checkpoints",
        help="Path to the Orbax checkpoint directory",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="./model.safetensors",
        help="Output path for the .safetensors file",
    )
    return parser.parse_args()


def main() -> None:
    """Run the conversion."""
    args = parse_args()

    checkpoint_dir = os.path.abspath(args.checkpoint_dir)
    if not os.path.exists(checkpoint_dir):
        raise FileNotFoundError(f"Checkpoint directory {checkpoint_dir} does not exist.")

    print(f"Loading Trainer state from {checkpoint_dir}...")
    trainer = FoldGemmaTrainer()
    trainer.load_checkpoint(checkpoint_dir)

    print(f"Exporting PyTorch weights to {args.output_path}...")
    trainer.export_to_pytorch(args.output_path)
    print("Export complete.")


if __name__ == "__main__":
    main()
