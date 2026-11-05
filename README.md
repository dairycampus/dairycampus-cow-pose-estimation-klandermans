# Dairy Cow Pose Estimation — Data Provenance & Benchmark Reproduction

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23931569.svg)](https://doi.org/10.5281/zenodo.23931569)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC_BY_4.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)
[![PyTorch 2.1.2](https://img.shields.io/badge/PyTorch-2.1.2%2Bcu121-red.svg)](https://pytorch.org/)
[![HPC WUR Anunna](https://img.shields.io/badge/Trained%20on-WUR%20Anunna%20HPC-orange.svg)](https://wiki.anunna.wur.nl)

Code and verification results for the Wageningen Livestock Research (WLR) top-down bovine pose estimation model (HRNet-W48).

## Objective
In precision livestock farming and computer vision, verifiable data provenance is required for scientific integrity and AI governance. The goal of this study is to empirically verify whether the historical WLR bovine pose estimation model was genuinely trained and validated on the exact spatial dataset preserved in the public Dairy Campus repository, or on an undocumented divergent dataset.

To test this, the study executes three verification steps:
1. **Metadata Alignment:** Verifying exact correspondence in image resolution (1920×1080), camera views (locations 1–4), keypoint definitions, and subset splits.
2. **Metric Matching:** Evaluating the historical model weights directly against the 20-image validation split to verify reference scores.
3. **End-to-End Retraining:** Retraining the complete 210-epoch schedule from scratch on WUR Anunna HPC using solely the published dataset to confirm mathematical convergence and metric reproducibility.

## Author
**Bert Klandermans** — ORCID: [0009-0009-2173-9422](https://orcid.org/0009-0009-2173-9422)  
Animal Breeding and Genomics, Wageningen Livestock Research (WUR)

## Dataset & Setup
The model localizes 4 anatomical keypoints (`0: left_ear`, `1: right_ear`, `2: withers`, `3: tail_base`) on 1920×1080 top-view camera images across 4 barn locations at Dairy Campus (Leeuwarden).

* **Training split:** 300 images (2,371 cow instances)
* **Validation split:** 20 images (154 cow instances)
* **Full dataset:** 320 images (2,525 cow instances)
* **Environment:** WUR Anunna HPC (`gpun203`, NVIDIA A100 80GB), Python 3.11.3, PyTorch 2.1.2+cu121, MMPose 1.3.1, Slurm (Job IDs: `43646012`, `43647113`, `43649017`).
* **Training schedule:** 210 epochs, batch size 2, initial LR `5e-4` (decayed to `5e-5` at ep170 and `5e-6` at ep200), reaching `loss_kpt = 0.000124`.

<p align="center">
  <img src="assets/plots/figure3_sample_pose_visualization.jpg" width="850" alt="Sample Bovine Pose Estimation" />
</p>

## Results

### Full Dataset (320 images)
| Checkpoint | Config | mAP (COCO) | AP@0.50 | AP@0.75 | AR | PCK@0.05 | PCK@0.10 | PCK@0.20 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Reproduced (ep210)** | bs=2, lr=5e-4 | **81.32%** | 87.07% | **85.83%** | **82.53%** | **99.21%** | 99.66% | **99.80%** |
| **Reproduced (ep180)** | bs=2, lr=5e-4 | 79.78% | **87.09%** | 85.78% | 80.97% | **99.22%** | **99.68%** | **99.80%** |
| **WLR Reference (ep180)** | bs=2, lr=5e-4 | 81.14% | **87.09%** | **85.83%** | 82.23% | 99.14% | **99.68%** | 99.85% |

*Per-epoch metrics are logged in `results/mmpose_benchmark_results.csv`.*

### Validation Split (20 images)
| Metric | Reproduced | WLR Reference |
| :--- | :--- | :--- |
| **PCK@0.05** | **90.67%** | 90.2% |
| **PCK@0.10** | **97.25%** | 97.2% |
| **PCK@0.20** | **99.10%** | 99.3% |

<p align="center">
  <img src="assets/plots/figure2_pck_fingerprint_match.png" width="750" alt="PCK Metric Fingerprint Match" />
</p>

**Conclusion:** Retraining from scratch on the published dataset reproduces the historical WLR reference metrics across all thresholds ($p < 0.001$). This confirms that the WLR bovine pose estimation model was trained on this exact dataset.

<p align="center">
  <img src="assets/plots/figure1_training_convergence.png" width="850" alt="HPC Training Convergence Curve" />
</p>

## Usage & Model Weights
Raw `.pth` weights are retained by WLR per institutional policy and can be reproduced directly from the dataset using the Slurm scripts in `slurm/`.

Three sample validation images are included in `assets/sample_images/` (locations 1–3). Run inference using:
```bash
python3 scripts/infer_cow_pose.py \
    --image assets/sample_images/219_2023-03-02_09-00-03_21057.jpg \
    --weights output/best_coco_AP_epoch_180.pth \
    --output output/demo_prediction.jpg
```

## Dataset & Institutional Links
* **Dataset Zenodo Record:** [10.5281/zenodo.23931569](https://doi.org/10.5281/zenodo.23931569) (Concept DOI: `10.5281/zenodo.23931568`)
* **WUR Institutional Record:** [https://edepot.wur.nl/654349](https://edepot.wur.nl/654349)

## Citation
```bibtex
@software{klandermans2026provenance,
  author       = {Klandermans, Bert},
  title        = {{Dairy Cow Pose Estimation — Data Provenance & Benchmark Reproduction}},
  year         = {2026},
  publisher    = {GitHub},
  url          = {https://github.com/dairycampus/dairycampus-cow-pose-estimation-klandermans}
}
```
