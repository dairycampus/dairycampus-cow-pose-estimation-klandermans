"""
Prepare Combined 320-Image Dataset (Train 300 + Val 20)
Combines train and val COCO annotations into a unified evaluation dataset
to benchmark full-dataset performance metrics.
"""

import os
import json
import argparse
import shutil

def parse_args():
    parser = argparse.ArgumentParser(description="Combine Train and Val splits into a single 320-image dataset")
    parser.add_argument('--dataset-root', type=str, default='data/cow_barn_kpt_it2_corr_300_20', help='Root directory of original 300_20 dataset')
    parser.add_argument('--output-dir', type=str, default='data/cow_kpt_320', help='Output directory for combined dataset')
    parser.add_argument('--copy-images', action='store_true', help='Copy images instead of creating symlinks')
    return parser.parse_args()

def main():
    args = parse_args()
    train_root = os.path.join(args.dataset_root, 'train')
    val_root = os.path.join(args.dataset_root, 'val')
    
    train_ann_path = os.path.join(train_root, 'annotations', 'instances_default.json')
    val_ann_path = os.path.join(val_root, 'annotations', 'instances_default.json')
    
    if not os.path.exists(train_ann_path) or not os.path.exists(val_ann_path):
        raise FileNotFoundError(f"Missing train or val annotations in {args.dataset_root}")
        
    out_images = os.path.join(args.output_dir, 'images')
    out_ann_dir = os.path.join(args.output_dir, 'annotations')
    os.makedirs(out_images, exist_ok=True)
    os.makedirs(out_ann_dir, exist_ok=True)
    
    with open(train_ann_path) as f:
        train_data = json.load(f)
    with open(val_ann_path) as f:
        val_data = json.load(f)
        
    # Link/copy train images
    for fname in os.listdir(os.path.join(train_root, 'images')):
        src = os.path.abspath(os.path.join(train_root, 'images', fname))
        dst = os.path.join(out_images, fname)
        if not os.path.exists(dst):
            if args.copy_images:
                shutil.copy2(src, dst)
            else:
                os.symlink(src, dst)
                
    # Link/copy val images
    for fname in os.listdir(os.path.join(val_root, 'images')):
        src = os.path.abspath(os.path.join(val_root, 'images', fname))
        dst = os.path.join(out_images, fname)
        if not os.path.exists(dst):
            if args.copy_images:
                shutil.copy2(src, dst)
            else:
                os.symlink(src, dst)
                
    # Merge annotations
    train_img_ids = {img["id"] for img in train_data["images"]}
    train_ann_ids = {ann["id"] for ann in train_data["annotations"]}
    
    combined = {
        "info": train_data.get("info", {}),
        "licenses": train_data.get("licenses", []),
        "categories": train_data["categories"],
        "images": list(train_data["images"]),
        "annotations": list(train_data["annotations"])
    }
    
    img_id_offset = max(train_img_ids) + 1 if train_img_ids else 1000000
    ann_id_offset = max(train_ann_ids) + 1 if train_ann_ids else 1000000
    
    val_id_map = {}
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
        
    out_json = os.path.join(out_ann_dir, 'instances_all_320.json')
    with open(out_json, 'w') as f:
        json.dump(combined, f)
        
    print(f"Successfully generated combined dataset at {args.output_dir}:")
    print(f"  Total Images: {len(combined['images'])} (300 train + 20 val)")
    print(f"  Total Annotations: {len(combined['annotations'])}")
    print(f"  Annotation file: {out_json}")

if __name__ == '__main__':
    main()
