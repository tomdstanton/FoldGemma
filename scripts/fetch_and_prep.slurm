#!/bin/bash
#SBATCH --job-name=fetch_and_prep
#SBATCH --account=<YOUR_ACCOUNT_ID>
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16              # aria2c can use 16 threads, and PyTorch dataloader will use them for parsing
#SBATCH --mem=64G                       # High RAM for downloading and holding chunk datasets
#SBATCH --partition=comp                # Use a standard CPU compute/data-mover node (GPU not required for prep)
#SBATCH --output=logs/prep_%j.out
#SBATCH --error=logs/prep_%j.err

module purge
module load aria2
module load python/3.10.10-gcc-11.2.0

# 1. ALWAYS run from /scratch on M3 for heavy I/O workloads
DATA_DIR="/scratch/$USER/data/afdb50"
PROJECT_DIR="/scratch/$USER/FoldGemma"
mkdir -p $DATA_DIR
cd $DATA_DIR

echo "Downloading Steinegger AFDB50 database..."
aria2c -x 16 -s 16 -j 2 "https://steineggerlab.s3.amazonaws.com/foldseek/afdb50.tar.gz"

echo "Downloads complete. Extracting..."
tar -I pigz -xvf afdb50.tar.gz || tar -xvf afdb50.tar.gz

echo "Running FoldGemma Prep via PyTorch DataLoader..."
cd $PROJECT_DIR

# Use our new native CLI command which serializes directly to binary shards
uv run foldgemma prep \
    "${DATA_DIR}/afdb50" \
    "${DATA_DIR}/afdb_binary" \
    --num-workers 16

echo "Fetch and Prep complete!"
