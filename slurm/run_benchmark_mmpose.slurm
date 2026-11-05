#!/bin/bash
#SBATCH --job-name=val_mmpose
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/lustre/scratch/WUR/ABGC/kland001/benchmark/slurm_val_mmpose_%j.out
#SBATCH --error=/lustre/scratch/WUR/ABGC/kland001/benchmark/slurm_val_mmpose_%j.err

echo "=== Running MMPose All Models Benchmark on Val (20) & Combined (320) ==="
module purge
module load 2023 slurm Python/3.11.3-GCCcore-12.3.0 CUDA/12.1.1

source /home/WUR/kland001/retraining/venv_exact/bin/activate
cd /lustre/scratch/WUR/ABGC/kland001/benchmark

python3 validate_all_mmpose.py
