#!/bin/bash
#SBATCH --job-name=foldgemma_prep
#SBATCH --account=<YOUR_ACCOUNT_ID>
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16              # PyTorch dataloader will use 16 threads for parsing
#SBATCH --mem=64G                       # High RAM for holding chunk datasets
#SBATCH --partition=comp                # Use a standard CPU compute/data-mover node
#SBATCH --output=logs/prep_%j.out
#SBATCH --error=logs/prep_%j.err

module purge
module load python/3.10.10-gcc-11.2.0

DATA_DIR="/scratch/$USER/data/afdb50"
PROJECT_DIR="/scratch/$USER/FoldGemma"

echo "Running FoldGemma Prep via PyTorch DataLoader..."
cd $PROJECT_DIR

# Run the native CLI command to serialize directly to binary shards
uv run foldgemma prep \
    "${DATA_DIR}/afdb50" \
    "${DATA_DIR}/afdb_binary" \
    --num-workers 16

echo "Prep complete!"
