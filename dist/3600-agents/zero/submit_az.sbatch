#!/bin/bash
#SBATCH --job-name=CarpetAZ
#SBATCH --account=your_pace_account_name
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:V100:1         # Requesting 1 V100 GPU (Adjust based on PACE availability)
#SBATCH --cpus-per-task=16       # Cores to rapidly manage the CPU-side Replay Buffer transfers
#SBATCH --mem=64G                # Necessary RAM for storing a massive game history array
#SBATCH --time=12:00:00          # 12-hour overnight run duration
#SBATCH --output=logs/az_%j.out  # Standard output trace
#SBATCH --error=logs/az_%j.err   # Standard error trace

echo "Starting AlphaZero Training on Host: $(hostname)"
echo "Date: $(date)"

# Purge any conflicting modules before loading Conda
module purge
module load anaconda3
module load cuda/12.1

# Activate the JAX computational environment
source activate carpet_env

# Guarantee output directories exist
mkdir -p checkpoints
mkdir -p logs

# Run the training pipeline orchestrator
python train.py

echo "Training Completed: $(date)"