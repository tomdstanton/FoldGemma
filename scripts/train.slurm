#!/bin/bash
#SBATCH --job-name=foldgemma_train
#SBATCH --account=<YOUR_ACCOUNT_ID>
#SBATCH --time=48:00:00                 # Max walltime (models take time to train!)
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8               # 8 CPUs ensures SeqIO/TFRecords don't bottleneck the GPU
#SBATCH --mem=64G                       # High RAM to buffer the TFRecord pipeline
#SBATCH --partition=m3g                 # GPU partition
#SBATCH --gres=gpu:A100:1               # Explicitly request an A100 (Ampere architecture is insanely fast)
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err

module purge
module load python/3.10.10-gcc-11.2.0

# 1. ALWAYS run from /scratch on M3 for heavy I/O workloads!
# Home directories are too slow for TFRecord streaming and will bottleneck your GPU.
cd /scratch/$USER/FoldGemma

# 2. Run the training via the CLI
# Note: Because you used `uv sync`, `uv run` handles the virtualenv automatically.
uv run foldgemma train \
    --tfrecord-dir /scratch/$USER/datasets/afdb_tfrecords \
    --output-dir /scratch/$USER/FoldGemma/checkpoints \
    --model-type foldgemma \
    --batch-size 32 \
    --learning-rate 3e-4 \
    --epochs 10