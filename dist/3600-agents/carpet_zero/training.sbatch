#!/bin/bash
#SBATCH -J CarpetZeroTrain
#SBATCH -N 1
#SBATCH --cpus-per-task=16
#SBATCH -G H100:1
#SBATCH --mem=80G
#SBATCH -t 12:00:00                       # Increased time for AlphaZero trai
ning
#SBATCH --mail-type=BEGIN,END,FAIL # Mail preferences
#SBATCH --mail-user=skata3@gatech.edu # E-mail address for notifications
#SBATCH -o logs/train_%j.out
#SBATCH -e logs/train_%j.err
# 1. Environment Setup
module purge
module load anaconda3
module load cuda/12.1
# 2. Activate the environment containing your requirements
source activate /storage/ice1/2/2/skata3/dist/az_env
# 3. Pathing Configuration
# Ensure the root and engine are visible for imports
export PYTHONPATH=$PYTHONPATH:$(pwd):$(pwd)/engine
# 4. Verification (Optional)
# Uncomment the line below if you want to verify dependencies before starting
# pip install -r requirements.txt --quiet
# 5. Run Training
# Use the module flag to resolve your package structure correctly
srun python3 3600-agents/carpet_zero/self_play.py