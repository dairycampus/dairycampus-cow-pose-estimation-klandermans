# -*- coding: utf-8 -*-
"""
================================================================================
Bovine Pose Estimation — Reproducibility Audit & Data Provenance Benchmark
================================================================================
File:         scripts/train_hrnet_baseline.py
Description:  HRNet-W48 210-epoch baseline retraining script reproducing the
              historical Wageningen Livestock Research (WLR) model schedule.
Author:       Bert Klandermans (Data Scientist)
              ORCID: https://orcid.org/0009-0009-2173-9422
Project Lead: Dr. Claudia Kamphuis
              ORCID: https://orcid.org/0000-0001-5552-036X
Affiliation:  Animal Breeding and Genomics, Wageningen Livestock Research (WUR)
Facility:     Dairy Campus, Leeuwarden, the Netherlands
Dataset DOI:  https://doi.org/10.5281/zenodo.23931569
Repository:   https://github.com/dairycampus/
              dairycampus-cow-pose-estimation-klandermans
License:      CC-BY-4.0 / Apache-2.0
Standards:    Black-formatted, PEP 8 / Flake8 compliant
================================================================================
"""

import argparse
import os
import sys
from typing import Any

try:
    from mmengine.config import Config
    from mmengine.registry import init_default_scope
    from mmengine.runner import Runner
    from mmpose.datasets.datasets.base import BaseCocoStyleDataset
    from mmpose.registry import DATASETS

    init_default_scope("mmpose")
except ImportError:
    pass


def parse_args() -> argparse.Namespace:
    """Parse command line arguments for HRNet training replay."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(script_dir)

    parser = argparse.ArgumentParser(
        description="Retrain HRNet-W48 Bovine Pose Estimation Baseline"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=os.path.join(
            repo_root,
            "configs",
            "td-hm_hrnet-w48_8xb64-210e_ap10k-256x256.py",
        ),
        help="MMPose config file",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default="data/cow_barn_kpt_it2_corr_300_20",
        help="Dataset root path",
    )
    parser.add_argument(
        "--pretrained",
        type=str,
        default=("checkpoints/hrnet_w48_ap10k_256x256-d95ab412_20211029.pth"),
        help="AP-10K pretrained backbone checkpoint",
    )
    parser.add_argument(
        "--work-dir",
        type=str,
        default="output/hrnet_w48_ap10k_300_20_retrained",
        help="Training output directory",
    )
    parser.add_argument("--epochs", type=int, default=210, help="Total training epochs")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size per GPU")
    parser.add_argument("--lr", type=float, default=5e-4, help="Initial learning rate")
    return parser.parse_args()


def configure_runner(args: argparse.Namespace) -> Any:
    """Configure MMPose runner and custom dataset class."""
    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Config file not found: {args.config}")

    cfg = Config.fromfile(args.config)

    @DATASETS.register_module(name="AP10KCowDataset", force=True)
    class AP10KCowDataset(BaseCocoStyleDataset):
        """AP10K Cow Dataset based on BaseCocoStyleDataset."""

        METAINFO = cfg.dataset_info

    dataset_dir = args.data_root
    cfg.data_root = dataset_dir
    cfg.data_name = "cow_barn_kpt_it2_corr_300_20"

    cfg.train_dataloader.dataset.data_root = dataset_dir
    cfg.val_dataloader.dataset.data_root = dataset_dir
    cfg.test_dataloader.dataset.data_root = dataset_dir

    cfg.train_dataloader.dataset.metainfo = cfg.dataset_info
    cfg.val_dataloader.dataset.metainfo = cfg.dataset_info
    cfg.test_dataloader.dataset.metainfo = cfg.dataset_info

    val_ann = f"{dataset_dir}/val/annotations/instances_default.json"
    cfg.val_evaluator.ann_file = val_ann
    cfg.test_evaluator.ann_file = val_ann

    if os.path.exists(args.pretrained):
        cfg.load_from = args.pretrained
    else:
        print(
            f"Notice: Pretrained checkpoint not found at {args.pretrained}. "
            "Training without pretrained backbone."
        )
        cfg.load_from = None

    cfg.work_dir = args.work_dir
    cfg.train_cfg.max_epochs = args.epochs
    cfg.train_dataloader.batch_size = args.batch_size
    cfg.optim_wrapper.optimizer.lr = args.lr

    if "log_config" in cfg and "hooks" in cfg.log_config:
        cfg.log_config.hooks = [
            h for h in cfg.log_config.hooks if h.get("type") != "WandbLoggerHook"
        ]

    return cfg


def main() -> None:
    """Main execution function."""
    args = parse_args()

    try:
        cfg = configure_runner(args)
    except NameError as err:
        print(
            f"Error: OpenMMLab/MMPose framework is required ({err}).\n"
            "Install dependencies using: pip install -U openmim && "
            "mim install mmengine 'mmcv>=2.1.0' 'mmpose>=1.1.0'"
        )
        sys.exit(1)

    print("=" * 60)
    print("   HRNet-W48 COW POSE REPRODUCIBILITY RETRAINING RUN")
    print("=" * 60)
    print(f"  Config:        {args.config}")
    print(f"  Dataset:       {cfg.data_root}")
    print(f"  Work Dir:      {cfg.work_dir}")
    print(f"  Epochs:        {cfg.train_cfg.max_epochs}")
    print(f"  Batch size:    {cfg.train_dataloader.batch_size}")
    print(f"  Initial LR:    {cfg.optim_wrapper.optimizer.lr}")
    print("=" * 60)

    runner = Runner.from_cfg(cfg)
    runner.train()


if __name__ == "__main__":
    main()
