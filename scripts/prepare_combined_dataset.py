# -*- coding: utf-8 -*-
"""
================================================================================
Bovine Pose Estimation — Reproducibility Audit & Data Provenance Benchmark
================================================================================
File:         scripts/prepare_combined_dataset.py
Description:  Combines the 300-image training split and 20-image validation
              split into a unified 320-image COCO dataset for benchmark audits.
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
import json
import os
import shutil
from typing import Any, Dict


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for combining dataset splits."""
    parser = argparse.ArgumentParser(
        description="Combine Train and Val splits into a 320-image dataset"
    )
    parser.add_argument(
        "--dataset-root",
        type=str,
        default="data/cow_barn_kpt_it2_corr_300_20",
        help="Root directory of original 300_20 dataset",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/cow_kpt_320",
        help="Output directory for combined dataset",
    )
    parser.add_argument(
        "--copy-images",
        action="store_true",
        help="Copy images instead of creating symbolic links",
    )
    return parser.parse_args()


def link_or_copy_images(src_dir: str, dest_dir: str, copy_files: bool = False) -> None:
    """Link or copy images from source directory into destination directory."""
    if not os.path.exists(src_dir):
        raise FileNotFoundError(f"Source directory not found: {src_dir}")

    for fname in os.listdir(src_dir):
        if fname.startswith(".") or fname.startswith("._"):
            continue
        src = os.path.abspath(os.path.join(src_dir, fname))
        dst = os.path.join(dest_dir, fname)
        if not os.path.exists(dst):
            if copy_files:
                shutil.copy2(src, dst)
            else:
                os.symlink(src, dst)


def build_combined_coco(
    train_data: Dict[str, Any], val_data: Dict[str, Any]
) -> Dict[str, Any]:
    """Merge train and val COCO annotation dictionaries with offset IDs."""
    train_img_ids = {img["id"] for img in train_data["images"]}
    train_ann_ids = {ann["id"] for ann in train_data["annotations"]}

    combined: Dict[str, Any] = {
        "info": train_data.get("info", {}),
        "licenses": train_data.get("licenses", []),
        "categories": train_data["categories"],
        "images": list(train_data["images"]),
        "annotations": list(train_data["annotations"]),
    }

    img_id_offset = max(train_img_ids) + 1 if train_img_ids else 1000000
    ann_id_offset = max(train_ann_ids) + 1 if train_ann_ids else 1000000

    val_id_map: Dict[int, int] = {}
    for img in val_data["images"]:
        orig_id = img["id"]
        if orig_id in train_img_ids:
            new_id = img_id_offset
            img_id_offset += 1
            val_id_map[orig_id] = new_id
            new_img = dict(img)
            new_img["id"] = new_id
            combined["images"].append(new_img)
        else:
            val_id_map[orig_id] = orig_id
            combined["images"].append(img)

    for ann in val_data["annotations"]:
        new_ann = dict(ann)
        orig_img_id = ann["image_id"]
        new_ann["image_id"] = val_id_map.get(orig_img_id, orig_img_id)
        if new_ann["id"] in train_ann_ids:
            new_ann["id"] = ann_id_offset
            ann_id_offset += 1
        combined["annotations"].append(new_ann)

    return combined


def main() -> None:
    """Main execution function."""
    args = parse_args()
    train_root = os.path.join(args.dataset_root, "train")
    val_root = os.path.join(args.dataset_root, "val")

    train_ann_path = os.path.join(train_root, "annotations", "instances_default.json")
    val_ann_path = os.path.join(val_root, "annotations", "instances_default.json")

    if not os.path.exists(train_ann_path) or not os.path.exists(val_ann_path):
        raise FileNotFoundError(
            f"Missing train or val annotations in {args.dataset_root}"
        )

    out_images = os.path.join(args.output_dir, "images")
    out_ann_dir = os.path.join(args.output_dir, "annotations")
    os.makedirs(out_images, exist_ok=True)
    os.makedirs(out_ann_dir, exist_ok=True)

    with open(train_ann_path, "r", encoding="utf-8") as f:
        train_data = json.load(f)
    with open(val_ann_path, "r", encoding="utf-8") as f:
        val_data = json.load(f)

    link_or_copy_images(
        os.path.join(train_root, "images"), out_images, args.copy_images
    )
    link_or_copy_images(os.path.join(val_root, "images"), out_images, args.copy_images)

    combined = build_combined_coco(train_data, val_data)

    out_json = os.path.join(out_ann_dir, "instances_all_320.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(combined, f, indent=2)

    print(f"Successfully generated combined dataset at {args.output_dir}:")
    print(f"  Total Images: {len(combined['images'])} (300 train + 20 val)")
    print(f"  Total Annotations: {len(combined['annotations'])}")
    print(f"  Annotation file: {out_json}")


if __name__ == "__main__":
    main()
