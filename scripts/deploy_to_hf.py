"""Deploy the model to Hugging Face Hub."""

import argparse
import os
import sys

from huggingface_hub import HfApi


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Deploy model to Hugging Face Hub")
    parser.add_argument(
        "--repo_id",
        type=str,
        required=True,
        help="Target Hugging Face repository ID (e.g. username/foldgemma)",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="./model.safetensors",
        help="Path to the model file",
    )
    return parser.parse_args()


def main() -> None:
    """Run deployment."""
    args = parse_args()
    token = os.environ.get("HF_TOKEN")
    
    if not token:
        print("Error: HF_TOKEN environment variable not set.", file=sys.stderr)
        sys.exit(1)
        
    if not os.path.exists(args.model_path):
        print(f"Error: Model file {args.model_path} does not exist.", file=sys.stderr)
        sys.exit(1)

    api = HfApi(token=token)
    
    print(f"Creating repository {args.repo_id} (if it doesn't exist)...")
    api.create_repo(repo_id=args.repo_id, exist_ok=True)
    
    print(f"Uploading {args.model_path} to {args.repo_id}...")
    api.upload_file(
        path_or_fileobj=args.model_path,
        path_in_repo=os.path.basename(args.model_path),
        repo_id=args.repo_id,
        commit_message="Deploy FastProtT5 model safetensors",
    )
    print("Deployment successful!")


if __name__ == "__main__":
    main()
