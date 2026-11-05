#!/bin/bash
#SBATCH --job-name=paper_hrnet_resume
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=slurm_resume_paper_%j.out
#SBATCH --error=slurm_resume_paper_%j.err

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

echo 'Resuming paper model training from Epoch 60...'
python3 resume_train.py

echo 'Training finished.'
