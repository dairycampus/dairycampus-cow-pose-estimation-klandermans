# -*- coding: utf-8 -*-
"""
================================================================================
Bovine Pose Estimation — Reproducibility Audit & Data Provenance Benchmark
================================================================================
File:         scripts/evaluate_provenance.py
Description:  Empirical benchmark and provenance verification evaluator.
              Evaluates HRNet-W48 pose checkpoints on validation split (20 imgs)
              and full combined dataset (320 imgs) to verify metric parity.
Author:       Bert Klandermans (Data Scientist)
              ORCID: https://orcid.org/0009-0009-2173-9422
Project Lead: Claudia Kamphuis
              ORCID: https://orcid.org/0000-0001-5552-036X
Affiliation:  Animal Breeding and Genomics, Wageningen Livestock Research (WUR)
Facility:     Dairy Campus, Leeuwarden, the Netherlands
Dataset DOI:  https://doi.org/10.5281/zenodo.23931569
Repository:   https://github.com/dairycampus/
              dairycampus-cow-pose-estimation-klandermans
License:      CC-BY-4.0
Standards:    Black-formatted, PEP 8 / Flake8 compliant
================================================================================
"""

import argparse
import os
import sys
from typing import Any, Dict

import pandas as pd

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
    """Parse CLI arguments for provenance evaluation."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(script_dir)

    parser = argparse.ArgumentParser(
        description="Evaluate Bovine Pose Model Lineage and Numerical Parity"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to .pth checkpoint file",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=os.path.join(
            repo_root,
            "configs",
            "td-hm_hrnet-w48_8xb64-210e_ap10k-256x256.py",
        ),
        help="MMPose model configuration file",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="EvaluationModel",
        help="Model label for reporting",
    )
    parser.add_argument(
        "--split",
        type=str,
        choices=["val", "all", "both"],
        default="both",
        help="Target split to evaluate ('val', 'all', or 'both')",
    )
    parser.add_argument(
        "--val-data-root",
        type=str,
        default="data/cow_barn_kpt_it2_corr_300_20",
        help="Path to 300_20 dataset directory",
    )
    parser.add_argument(
        "--all-data-root",
        type=str,
        default="data/cow_kpt_320",
        help="Path to 320-image combined dataset directory",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Evaluation inference batch size",
    )
    parser.add_argument(
        "--out-csv",
        type=str,
        default=os.path.join(repo_root, "results", "provenance_eval_results.csv"),
        help="Path to export benchmark results CSV",
    )
    return parser.parse_args()


def run_split_eval(
    cfg_file: str,
    checkpoint_path: str,
    model_name: str,
    eval_split: str,
    data_root: str,
    batch_size: int,
) -> Dict[str, Any]:
    """Execute evaluation on a single data split."""
    if not os.path.exists(cfg_file):
        raise FileNotFoundError(f"Config file not found: {cfg_file}")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    cfg = Config.fromfile(cfg_file)

    @DATASETS.register_module(name="AP10KCowDataset", force=True)
    class AP10KCowDataset(BaseCocoStyleDataset):
        METAINFO = cfg.dataset_info

    if eval_split == "val":
        ann_file = f"{data_root}/val/annotations/instances_default.json"
        img_prefix = f"{data_root}/val/images/"
        desc = "Val Split (20 images, 154 instances)"
    else:
        ann_file = f"{data_root}/annotations/instances_all_320.json"
        img_prefix = f"{data_root}/images/"
        desc = "Full Dataset (320 images, 2,525 instances)"

    if not os.path.exists(ann_file):
        raise FileNotFoundError(f"Annotation file not found: {ann_file}")

    cfg.data_root = data_root
    cfg.test_dataloader.dataset.data_root = data_root
    cfg.test_dataloader.dataset.ann_file = ann_file
    cfg.test_dataloader.dataset.data_prefix = dict(img=img_prefix)
    cfg.test_dataloader.dataset.metainfo = cfg.dataset_info
    cfg.test_dataloader.batch_size = batch_size
    cfg.test_dataloader.num_workers = 4

    cfg.test_evaluator = [
        dict(
            type="CocoMetric",
            ann_file=ann_file,
            score_mode="keypoint",
            keypoint_score_thr=0.0,
            prefix="coco",
        ),
        dict(type="PCKAccuracy", thr=0.05, norm_item=["bbox"], prefix="pck05"),
        dict(type="PCKAccuracy", thr=0.10, norm_item=["bbox"], prefix="pck10"),
        dict(type="PCKAccuracy", thr=0.20, norm_item=["bbox"], prefix="pck20"),
    ]

    cfg.load_from = checkpoint_path
    cfg.work_dir = f"work_dirs/eval_{eval_split}_{model_name}"

    print(f"\nEvaluating '{model_name}' on {desc}...")
    runner = Runner.from_cfg(cfg)
    metrics = runner.test()

    return {
        "model": model_name,
        "split": eval_split,
        "checkpoint": os.path.basename(checkpoint_path),
        "coco/AP": round(metrics.get("coco/AP", 0.0) * 100, 2),
        "coco/AP .50": round(metrics.get("coco/AP .50", 0.0) * 100, 2),
        "coco/AP .75": round(metrics.get("coco/AP .75", 0.0) * 100, 2),
        "coco/AR": round(metrics.get("coco/AR", 0.0) * 100, 2),
        "pck05/PCK": round(metrics.get("pck05/PCK", 0.0) * 100, 2),
        "pck10/PCK": round(metrics.get("pck10/PCK", 0.0) * 100, 2),
        "pck20/PCK": round(metrics.get("pck20/PCK", 0.0) * 100, 2),
    }


def main() -> None:
    """Main CLI entry point for evaluation."""
    args = parse_args()

    splits_to_eval = ["val", "all"] if args.split == "both" else [args.split]
    results = []

    for s in splits_to_eval:
        data_root = args.val_data_root if s == "val" else args.all_data_root
        try:
            row = run_split_eval(
                args.config,
                args.checkpoint,
                args.model_name,
                s,
                data_root,
                args.batch_size,
            )
            results.append(row)
        except NameError as err:
            print(
                f"Error: OpenMMLab/MMPose framework is required ({err}).\n"
                "Install dependencies using: pip install -U openmim && "
                "mim install mmengine 'mmcv>=2.1.0' 'mmpose>=1.1.0'"
            )
            sys.exit(1)

    df = pd.DataFrame(results)
    print("\nBenchmark Evaluation Results:")
    print(df.to_string(index=False))

    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    df.to_csv(args.out_csv, index=False)
    print(f"\nSaved metrics to: {args.out_csv}")


if __name__ == "__main__":
    main()
