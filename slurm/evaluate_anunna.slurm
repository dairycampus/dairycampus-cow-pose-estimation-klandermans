#!/bin/bash
# ==============================================================================
# Bovine Pose Estimation — Reproducibility Audit & Data Provenance Benchmark
# ==============================================================================
# File:         slurm/evaluate_anunna.slurm
# Description:  WUR Anunna HPC Reference Script: Provenance Benchmark Evaluator
# Job IDs:      43647113, 43649017 | Hardware: Node gpun203 (NVIDIA A100-SXM4-80GB)
# Author:       Bert Klandermans (Data Scientist)
#               ORCID: https://orcid.org/0009-0009-2173-9422
# Project Lead: Claudia Kamphuis
#               ORCID: https://orcid.org/0000-0001-5552-036X
# Affiliation:  Animal Breeding and Genomics, Wageningen Livestock Research (WUR)
# Facility:     Dairy Campus, Leeuwarden, the Netherlands
# Dataset DOI:  https://doi.org/10.5281/zenodo.23931569
# Repository:   https://github.com/dairycampus/dairycampus-cow-pose-estimation-klandermans
# License:      CC-BY-4.0
#
# NOTE FOR EXTERNAL RESEARCHERS:
# This file is an exact reference script for the Slurm scheduler on WUR Anunna.
# If running outside WUR HPC, execute the evaluation script directly:
#   python3 scripts/evaluate_provenance.py --checkpoint <path_to_weights.pth>
# ==============================================================================

#SBATCH --job-name=val_mmpose_provenance
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=slurm_val_provenance_%j.out
#SBATCH --error=slurm_val_provenance_%j.err

echo "=== Running Provenance Benchmark on Val (20) & Combined (320) ==="
module purge
module load 2023 slurm Python/3.11.3-GCCcore-12.3.0 CUDA/12.1.1

source /home/WUR/kland001/retraining/venv_exact/bin/activate
nvidia-smi

python3 scripts/evaluate_provenance.py \
    --checkpoint output/hrnet_w48_ap10k_300_20_retrained/best_coco_AP_epoch_180.pth \
    --model-name "Reproduced_ep180" \
    --split both \
    --val-data-root data/cow_barn_kpt_it2_corr_300_20 \
    --all-data-root data/cow_kpt_320 \
    --out-csv results/reproduced_ep180_metrics.csv

echo "Evaluation completed."
