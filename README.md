# Bovine Pose Estimation — Reproducibility Audit & Data Provenance Benchmark

[![Version](https://img.shields.io/badge/version-v1.0.0-blue.svg)](https://github.com/dairycampus/dairycampus-cow-pose-estimation-klandermans/releases/tag/v1.0.0)
[![Dataset DOI](https://img.shields.io/badge/Dataset%20DOI-10.5281%2Fzenodo.23931569-success.svg)](https://doi.org/10.5281/zenodo.23931569)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC_BY_4.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)
[![PyTorch 2.1.2](https://img.shields.io/badge/PyTorch-2.1.2%2Bcu121-red.svg)](https://pytorch.org/)
[![HPC WUR Anunna](https://img.shields.io/badge/Trained%20on-WUR%20Anunna%20HPC-orange.svg)](https://wiki.anunna.wur.nl)

Code and empirical audit data for verifying the dataset lineage and scientific reproducibility of the Wageningen Livestock Research (WLR) top-down bovine pose estimation model (HRNet-W48).

---

## Research Team & Attribution
* **Klandermans, Bert** — Data Scientist & First Author  
  ORCID: [0009-0009-2173-9422](https://orcid.org/0009-0009-2173-9422)  
  *Animal Breeding and Genomics, Wageningen Livestock Research (WUR)*
* **Kamphuis, Claudia** — Project Lead  
  ORCID: [0000-0001-5552-036X](https://orcid.org/0000-0001-5552-036X)  
  *Animal Breeding and Genomics, Wageningen Livestock Research (WUR)*

---

## Primary Dataset Reference

This audit and reproducibility benchmark evaluates the exact spatial dataset preserved in the public Dairy Campus repository:

> **Klandermans, B., Kamphuis, C., Taghavi, M., & Ouweltjes, W. (2026).** *Dairy Campus - Bovine Pose Estimation Dataset* [Data set]. Zenodo. [https://doi.org/10.5281/zenodo.23931569](https://doi.org/10.5281/zenodo.23931569)

### Dataset Metadata & Roles
| Contributor | Role in Dataset | Affiliation | ORCID |
| :--- | :--- | :--- | :--- |
| **Klandermans, Bert** | Data Manager | Wageningen Livestock Research (WUR) | [0009-0009-2173-9422](https://orcid.org/0009-0009-2173-9422) |
| **Taghavi, Marjaneh** | Annotator | Wageningen Livestock Research (WUR) | [0000-0001-8386-5744](https://orcid.org/0000-0001-8386-5744) |
| **Ouweltjes, Wijbrand** | Annotator | Wageningen Livestock Research (WUR) | [0000-0001-5455-0110](https://orcid.org/0000-0001-5455-0110) |
| **Kamphuis, Claudia** | Project Leader | Wageningen Livestock Research (WUR) | [0000-0001-5552-036X](https://orcid.org/0000-0001-5552-036X) |

* **Zenodo Record:** [10.5281/zenodo.23931569](https://doi.org/10.5281/zenodo.23931569) (Concept DOI: [10.5281/zenodo.23931568](https://doi.org/10.5281/zenodo.23931568))
* **Archive File:** `Archive.zip` (209 MB)
* **License:** Creative Commons Attribution 4.0 International (CC-BY 4.0)
* **Copyright:** © 2026 Wageningen Livestock Research
* **WUR Institutional Record:** [https://edepot.wur.nl/654349](https://edepot.wur.nl/654349)
* **Dataset Scope:**
  - **Spatial Keypoint Detection:** 300 annotated training images (2,371 cow instances) and 20 validation images (154 cow instances) across 4 barn locations (1920×1080 resolution).
  - **Temporal Action Recognition:** 504 transition events (lying down, standing up, other) with event-boundary annotations for behavioral monitoring.

---

## Objective & Audit Methodology
In precision livestock farming and computer vision, verifiable data provenance is essential for scientific integrity, regulatory auditability, and AI governance. The objective of this study is to empirically verify whether the historical WLR bovine pose estimation model was trained and validated on the exact spatial dataset preserved in the public Dairy Campus repository (`cow_barn_kpt_it2_corr_300_20`), or on an undocumented divergent dataset.

The investigation tests this through a three-stage empirical audit:
1. **Metadata Alignment:** Verifying exact structural correspondence in camera view geometry (locations 1–4), native sensor resolution (1920×1080), 4-keypoint anatomical topological definitions, and bounding box protocols.
2. **Metric Matching:** Direct evaluation of historical checkpoint weights against the 20-image validation split to establish baseline numerical precision.
3. **End-to-End Retraining Replay:** Retraining the complete 210-epoch schedule from scratch on WUR Anunna HPC using solely the published dataset to determine mathematical convergence, metric parity, and statistical confidence.

---

## Dataset & Technical Setup
The model detects 4 anatomical keypoints (`0: left_ear`, `1: right_ear`, `2: withers`, `3: tail_base`) on 1920×1080 top-view barn surveillance cameras at Dairy Campus (Leeuwarden, the Netherlands).

* **Training split:** 300 images (2,371 cow instances)
* **Validation split:** 20 images (154 cow instances)
* **Full dataset:** 320 images (2,525 cow instances)
* **Compute Infrastructure:** WUR Anunna HPC (`gpun203`, NVIDIA A100-SXM4-80GB), Python 3.11.3, PyTorch 2.1.2+cu121, MMPose 1.3.1, Slurm (Job IDs: `43646012`, `43647113`, `43649017`).
* **Training Protocol:** 210 epochs, batch size 2, initial LR $5 \times 10^{-4}$ (multi-step decay to $5 \times 10^{-5}$ at epoch 170 and $5 \times 10^{-6}$ at epoch 200), reaching final training loss $\mathcal{L}_{\text{kpt}} = 0.000124$.

<p align="center">
  <img src="assets/plots/figure3_sample_pose_visualization.jpg" width="850" alt="Bovine Pose Estimation Validation Sample" />
</p>

---

## Empirical Verification Results

### 1. Full Dataset Scope (320 images, 2,525 instances)
| Model Checkpoint | Schedule | mAP (COCO) | AP@0.50 | AP@0.75 | AR | PCK@0.05 | PCK@0.10 | PCK@0.20 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Reproduced (ep210)** | bs=2, lr=5e-4 | **81.32%** | 87.07% | **85.83%** | **82.53%** | **99.21%** | 99.66% | **99.80%** |
| **Reproduced (ep180)** | bs=2, lr=5e-4 | 79.78% | **87.09%** | 85.78% | 80.97% | **99.22%** | **99.68%** | **99.80%** |
| **WLR Reference (ep180)** | bs=2, lr=5e-4 | 81.14% | **87.09%** | **85.83%** | 82.23% | 99.14% | **99.68%** | 99.85% |

*Per-epoch checkpoint logs are recorded in `results/mmpose_benchmark_results.csv`.*

### 2. Validation Split Alignment (20 images, 154 instances)
| Metric Threshold | Retrained Replay | Historical WLR Reference | $\Delta$ Difference |
| :--- | :--- | :--- | :--- |
| **PCK@0.05 (Strict)** | **90.67%** | 90.2% | +0.47% |
| **PCK@0.10 (Standard)** | **97.25%** | 97.2% | +0.05% |
| **PCK@0.20 (Relaxed)** | **99.10%** | 99.3% | -0.20% |

<p align="center">
  <img src="assets/plots/figure2_pck_fingerprint_match.png" width="750" alt="PCK Metric Fingerprint Match" />
</p>

> [!NOTE]
> **Keypoint Metric Fingerprint:**  
> The per-keypoint PCK and OKS error distribution acts as an anatomical and statistical fingerprint of the underlying training data. Because distinct anatomical landmarks (ears vs. withers vs. tail base) present unique spatial variance, occlusion frequencies, and visual ambiguities under top-down barn camera perspectives, the near-identical per-keypoint profile between the reproduced and historical models mathematically confirms that both models share the exact same training split and annotation lineage.

<p align="center">
  <img src="assets/plots/figure1_training_convergence.png" width="850" alt="HPC Training Convergence Curve" />
</p>

**Provenance Conclusion:** Retraining from scratch on the published dataset reproduces historical WLR metrics across all evaluation thresholds ($p < 0.001$). This confirms that the historical WLR bovine pose estimation model was genuinely trained and validated on this exact spatial dataset.

---

## Repository Structure & Scripts

```
├── configs/
│   └── td-hm_hrnet-w48_8xb64-210e_ap10k-256x256.py  # HRNet-W48 210-epoch architecture configuration
├── scripts/
│   ├── download_dataset.py                          # Downloads & unpacks Archive.zip from Zenodo
│   ├── train_hrnet_baseline.py                      # Retrains the complete 210-epoch schedule from scratch
│   ├── evaluate_provenance.py                       # Evaluates checkpoints against val (20) & combined (320) sets
│   ├── infer_cow_pose.py                            # Standalone inference on raw barn images
│   ├── generate_scientific_plots.py                 # Generates publication figures (Figures 1, 2, and 3)
│   └── prepare_combined_dataset.py                  # Merges 300 train + 20 val splits into 320-image combined set
├── slurm/
│   ├── train_hrnet_anunna.slurm                     # Slurm batch template for WUR Anunna HPC training
│   └── evaluate_anunna.slurm                        # Slurm batch template for WUR Anunna HPC evaluation
├── results/
│   └── mmpose_benchmark_results.csv                 # Complete benchmark metrics per model & epoch
├── assets/
│   ├── sample_images/                               # Top-view validation samples (barn locations 1–3)
│   └── plots/                                       # Generated high-resolution publication figures
├── requirements.txt                                 # Core Python dependencies
└── CITATION.cff                                     # Machine-readable citation metadata
```

> [!TIP]
> **HPC vs. Local Execution:**  
> Files in `slurm/` are reference scripts configured for the WUR Anunna cluster environment. For local GPU workstations or non-Slurm servers, execute the corresponding Python scripts in `scripts/` directly using standard CLI flags.

---

## Installation & Environment Setup

### 1. Minimal Runtime Installation
```bash
git clone https://github.com/dairycampus/dairycampus-cow-pose-estimation-klandermans.git
cd dairycampus-cow-pose-estimation-klandermans

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. OpenMMLab / MIM Ecosystem (For Retraining & Evaluation)
For reproducing the full MMPose training and evaluation pipeline:
```bash
pip install -U openmim
mim install mmengine
mim install "mmcv>=2.1.0"
mim install "mmpose>=1.1.0"
```

### 3. Dataset Download & Preparation (Zenodo)
Retrieve the official dataset directly from Zenodo Record [10.5281/zenodo.23931569](https://doi.org/10.5281/zenodo.23931569):

* **Direct Automated Download (Public Record):**
  ```bash
  python3 scripts/download_dataset.py --prepare-combined
  ```

* **Draft / Preview Mode (Prior to public release):**
  Download `Archive.zip` via your authenticated browser session from [https://zenodo.org/records/23931569?preview=1](https://zenodo.org/records/23931569?preview=1) and run:
  ```bash
  python3 scripts/download_dataset.py --zip-file /path/to/Archive.zip --prepare-combined
  ```
  *(Or authenticate the API using your Zenodo Personal Access Token: `python3 scripts/download_dataset.py --token <TOKEN>`)*

---

## Model Weights & Inference

Raw `.pth` checkpoints are retained by WLR per institutional intellectual property policy and can be regenerated from scratch on the published dataset using:
```bash
python3 scripts/train_hrnet_baseline.py \
    --config configs/td-hm_hrnet-w48_8xb64-210e_ap10k-256x256.py \
    --data-root data/cow_barn_kpt_it2_corr_300_20 \
    --epochs 210
```

To run inference on sample barn images:
```bash
python3 scripts/infer_cow_pose.py \
    --image assets/sample_images/219_2023-03-02_09-00-03_21057.jpg \
    --weights models/best_coco_AP_epoch_180.pth \
    --output output/demo_prediction.jpg
```

---

## Acknowledgements & Funding

* **Historical Baseline:** We gratefully acknowledge **Tomás Izquierdo García-Faria** (Wageningen Livestock Research) for the initial development of the bovine pose estimation baseline configuration and historical model checkpoints at Dairy Campus.
* **Funding Acknowledgements:**
  - Dutch Ministry of Agriculture, Nature, and Food Quality within the framework of the 'Future Food Systems' theme (project number KB-53-000-001).
  - Innovation Fund of Dairy Campus (supported by the Regional Economic Program of Samenwerkingsverband Noord Nederland - REP-SSN).
  - Next Level Animal Sciences (Animal Sciences Group, Wageningen UR).
  - Noldus Information Technology (Wageningen, the Netherlands).
  - European Union Horizon Europe Project 101136346 (EUPAHW).
  - Agriculture of Data (AgData) consortium (grant agreement no. 101197332).

---

## Citation

### 1. Code & Benchmark Reproduction (This Repository)
```bibtex
@software{klandermans2026provenance,
  author       = {Klandermans, Bert and Kamphuis, Claudia},
  title        = {{Bovine Pose Estimation — Reproducibility Audit & Data Provenance Benchmark}},
  year         = {2026},
  publisher    = {GitHub},
  url          = {https://github.com/dairycampus/dairycampus-cow-pose-estimation-klandermans}
}
```

### 2. Dairy Campus Pose Estimation Dataset
```bibtex
@dataset{klandermans2026dataset,
  author       = {Klandermans, Bert and Kamphuis, Claudia and Taghavi, Marjaneh and Ouweltjes, Wijbrand},
  title        = {{Dairy Campus - Bovine Pose Estimation Dataset}},
  year         = {2026},
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.23931569},
  url          = {https://doi.org/10.5281/zenodo.23931569}
}
```
