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
DATA_DIR="/scratch/$USER/data/afdb"
PROJECT_DIR="/scratch/$USER/FoldGemma"
mkdir -p $DATA_DIR
cd $DATA_DIR

echo "Downloading Steinegger AFDB databases..."

# The base URL from the user's opendata server screenshot
BASE_URL="https://opendata.mmseqs.org/afdbv6"

# aria2c arguments:
# -x 16: use 16 concurrent connections per file
# -s 16: split the file into 16 parts for download
# -j 2: download up to 2 files concurrently
aria2c -x 16 -s 16 -j 2 \
    "${BASE_URL}/afdb241_ss.tar.gz" \
    "${BASE_URL}/afdb241_fcz.tar.gz"

echo "Downloads complete. Extracting..."
tar -I pigz -xvf afdb241_ss.tar.gz
tar -I pigz -xvf afdb241_fcz.tar.gz

echo "Creating MMseqs TSV dump for 3Di sequences..."
# Assuming mmseqs is installed/loaded. If not, load it:
# module load mmseqs2
mmseqs createtsv afdb241 afdb241_ss afdb241_ss.tsv

echo "Running FoldGemma Prep via PyTorch DataLoader..."
cd $PROJECT_DIR

# Use our new native CLI command which leverages torch.utils.data.IterableDataset
uv run foldgemma prep \
    --tsv "${DATA_DIR}/afdb241_ss.tsv" \
    --fcz "${DATA_DIR}/afdb241_fcz" \
    --out-dir "${DATA_DIR}/tfrecords" \
    --num-workers 16

echo "Fetch and Prep complete!"
