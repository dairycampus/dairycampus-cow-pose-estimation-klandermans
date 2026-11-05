#!/bin/bash
# ==============================================================================
# WUR Anunna HPC Reference Script: 210-Epoch HRNet-W48 Retraining Replay
# Job ID Reference: 43646012 | Node: gpun203 (NVIDIA A100-SXM4-80GB)
#
# NOTE FOR EXTERNAL RESEARCHERS:
# This file is an exact reference script for the Slurm scheduler on WUR Anunna.
# If running outside WUR HPC, execute the underlying Python training script directly:
#   python3 scripts/train_hrnet_baseline.py --epochs 210 --batch-size 2
# ==============================================================================

#SBATCH --job-name=paper_hrnet_replay
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=slurm_train_replay_%j.out
#SBATCH --error=slurm_train_replay_%j.err

echo "================== SLURM JOB INFO =================="
echo "Node: $SLURMD_NODENAME"
echo "Job ID: $SLURM_JOB_ID"
echo "Allocated GPU: $CUDA_VISIBLE_DEVICES"
echo "==================================================="

# Module environment on WUR Anunna
module purge
module load 2023 slurm Python/3.11.3-GCCcore-12.3.0 CUDA/12.1.1

# Activate environment
source /home/WUR/kland001/retraining/venv_exact/bin/activate
nvidia-smi

echo "Starting 210-epoch HRNet-W48 retraining replay..."
python3 scripts/train_hrnet_baseline.py \
    --config configs/td-hm_hrnet-w48_8xb64-210e_ap10k-256x256.py \
    --data-root data/cow_barn_kpt_it2_corr_300_20 \
    --work-dir output/hrnet_w48_ap10k_300_20_retrained \
    --epochs 210 \
    --batch-size 2 \
    --lr 5e-4

echo "Training completed."
