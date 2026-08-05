#!/bin/bash
#SBATCH --job-name=foldgemma_train
#SBATCH --account=kr58
#SBATCH --time=48:00:00
#SBATCH --partition=m3g
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8

echo "Starting FoldGemma Training..."
just train 'data/tfrecords/*.tfrecord'
echo "Training complete!"
