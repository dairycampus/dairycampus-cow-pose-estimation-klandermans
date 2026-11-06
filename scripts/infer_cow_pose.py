# -*- coding: utf-8 -*-
"""
================================================================================
Bovine Pose Estimation — Reproducibility Audit & Data Provenance Benchmark
================================================================================
File:         scripts/infer_cow_pose.py
Description:  Bovine top-down pose estimation inference script.
              Localizes 4 anatomical keypoints (left ear, right ear, withers,
              tail base) on top-view barn surveillance frames.
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
from typing import List, Tuple

import cv2
import numpy as np
import torch

KEYPOINT_NAMES: List[str] = [
    "Left Ear",
    "Right Ear",
    "Withers",
    "Tail Base",
]
# Skeleton topology: ear-to-ear, ears-to-withers, withers-to-tail
SKELETON: List[Tuple[int, int]] = [(0, 1), (0, 2), (1, 2), (2, 3)]
COLORS: List[Tuple[int, int, int]] = [
    (0, 255, 0),  # 0: Left Ear (Green)
    (0, 255, 0),  # 1: Right Ear (Green)
    (0, 0, 255),  # 2: Withers (Red)
    (255, 0, 0),  # 3: Tail Base (Blue)
]


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for cow pose inference."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(script_dir)

    parser = argparse.ArgumentParser(
        description="Dairy Cow Top-Down Keypoint Pose Estimation"
    )
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument(
        "--weights",
        type=str,
        default=os.path.join(repo_root, "models", "best_coco_AP_epoch_180.pth"),
        help="MMPose HRNet checkpoint (.pth)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=os.path.join(
            repo_root,
            "configs",
            "td-hm_hrnet-w48_8xb64-210e_ap10k-256x256.py",
        ),
        help="MMPose HRNet model config file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=os.path.join(repo_root, "output", "demo_prediction.jpg"),
        help="Path to save output visualization",
    )
    parser.add_argument(
        "--yolo-weights",
        type=str,
        default=None,
        help="Optional YOLO cow detector weights (.pt)",
    )
    parser.add_argument(
        "--bbox",
        type=float,
        nargs=4,
        default=None,
        metavar=("X1", "Y1", "X2", "Y2"),
        help="Optional manual cow bounding box",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0" if torch.cuda.is_available() else "cpu",
        help="Inference device ('cuda:0' or 'cpu')",
    )
    return parser.parse_args()


def draw_cow_pose(
    image: np.ndarray,
    bboxes: np.ndarray,
    keypoints_list: List[Tuple[np.ndarray, np.ndarray]],
) -> np.ndarray:
    """Overlay detected bounding boxes and anatomical keypoints on frame."""
    vis_img = image.copy()
    for bbox, (kpts, scores) in zip(bboxes, keypoints_list):
        x1, y1, x2, y2 = map(int, bbox[:4])
        cv2.rectangle(vis_img, (x1, y1), (x2, y2), (0, 255, 255), 2)

        # Draw anatomical skeletal connections
        for p1, p2 in SKELETON:
            if scores[p1] > 0.2 and scores[p2] > 0.2:
                pt1 = (int(kpts[p1][0]), int(kpts[p1][1]))
                pt2 = (int(kpts[p2][0]), int(kpts[p2][1]))
                cv2.line(vis_img, pt1, pt2, (255, 255, 0), 2)

        # Draw landmark circles
        for idx, (pt, sc) in enumerate(zip(kpts, scores)):
            if sc > 0.2:
                center = (int(pt[0]), int(pt[1]))
                cv2.circle(vis_img, center, 5, COLORS[idx], -1)
                cv2.circle(vis_img, center, 6, (0, 0, 0), 1)
    return vis_img


def main() -> None:
    """Main execution function for inference."""
    args = parse_args()

    if not os.path.exists(args.image):
        raise FileNotFoundError(f"Input image not found: {args.image}")
    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Model config not found: {args.config}")

    img = cv2.imread(args.image)
    if img is None:
        raise ValueError(f"Unable to read or decode image: {args.image}")
    h, w = img.shape[:2]

    bboxes = []
    if args.yolo_weights and os.path.exists(args.yolo_weights):
        try:
            from ultralytics import YOLO

            print(f"Detecting cows with YOLO: {args.yolo_weights}")
            yolo_model = YOLO(args.yolo_weights)
            res = yolo_model(img, conf=0.35, device=args.device)[0]
            bboxes = res.boxes.xyxy.cpu().numpy()
        except ImportError:
            print(
                "Notice: ultralytics is not installed. "
                "Falling back to manual or full-frame bounding box."
            )

    if len(bboxes) == 0:
        if args.bbox:
            bboxes = np.array([args.bbox])
        else:
            # Fall back to full-frame detection
            bboxes = np.array([[0, 0, w, h]])

    print(f"Running HRNet-W48 pose estimation on {len(bboxes)} cow instance(s)...")
    try:
        from mmengine.config import Config
        from mmpose.apis import inference_topdown, init_model
        from mmpose.datasets.datasets.base import BaseCocoStyleDataset
        from mmpose.registry import DATASETS

        cfg = Config.fromfile(args.config)

        @DATASETS.register_module(name="AP10KCowDataset", force=True)
        class AP10KCowDataset(BaseCocoStyleDataset):
            METAINFO = cfg.dataset_info

        pose_model = init_model(args.config, args.weights, device=args.device)
        pose_results = inference_topdown(pose_model, img, bboxes)

        keypoints_list = []
        for res in pose_results:
            kpts = res.pred_instances.keypoints[0]
            scores = res.pred_instances.keypoint_scores[0]
            keypoints_list.append((kpts, scores))

        vis_img = draw_cow_pose(img, bboxes, keypoints_list)
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        cv2.imwrite(args.output, vis_img)
        print(f"Prediction successfully saved to: {args.output}")
    except ImportError as err:
        print(
            f"Error: MMPose framework is required for inference ({err}).\n"
            "Please install dependencies: pip install -U openmim && "
            "mim install mmengine 'mmcv>=2.1.0' 'mmpose>=1.1.0'"
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
