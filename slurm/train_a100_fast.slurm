#!/bin/bash
#SBATCH --job-name=hrnet_bs128
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=A100
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=slurm_train_fast_%j.out
#SBATCH --error=slurm_train_fast_%j.err

echo '================== SLURM JOB INFO =================='
echo 'Node: '$SLURMD_NODENAME
echo 'Job ID: '$SLURM_JOB_ID
echo 'Allocated GPU: '$CUDA_VISIBLE_DEVICES
echo '==================================================='

# Load modules
module purge
module load 2023 slurm Python/3.11.3-GCCcore-12.3.0 CUDA/12.1.1

# Navigate to retraining folder
cd /home/WUR/kland001/retraining

# Activate virtualenv
source /home/WUR/kland001/retraining/venv_exact/bin/activate

# Print GPU info
nvidia-smi

echo 'Starting high-throughput training (bs=128, lr=0.002)...'
python3 train_fast.py

echo 'Training finished.'
