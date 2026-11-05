import os
import sys
import json
from ultralytics import YOLO

def main():
    print("==========================================================")
    print("      YOLOv8 VALIDATION ON COMBINED DATASET (400 IMAGES)  ")
    print("==========================================================")
    
    model_path = "/lustre/scratch/WUR/ABGC/kland001/benchmark/models/yolo/best_yolov8_390_10.pt"
    data_dir = "/lustre/scratch/WUR/ABGC/kland001/benchmark/data/yolo_bbox_400"
    
    # Create combined all.txt containing both train and val images
    train_txt = os.path.join(data_dir, "train.txt")
    val_txt = os.path.join(data_dir, "val.txt")
    all_txt = os.path.join(data_dir, "all_400.txt")
    
    lines = []
    if os.path.exists(train_txt):
        with open(train_txt) as f:
            lines.extend(f.readlines())
    if os.path.exists(val_txt):
        with open(val_txt) as f:
            lines.extend(f.readlines())
            
    # Normalize paths to absolute paths
    abs_lines = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if line.startswith("./"):
            line = os.path.join(data_dir, line[2:])
        abs_lines.append(line + "\n")
        
    with open(all_txt, "w") as f:
        f.writelines(abs_lines)
        
    print(f"Created all_400.txt with {len(abs_lines)} image paths.")
    
    # Create dedicated data_all.yaml
    yaml_path = os.path.join(data_dir, "data_all.yaml")
    yaml_content = f"""path: {data_dir}
train: {all_txt}
val: {all_txt}

names:
  0: cow
"""
    with open(yaml_path, "w") as f:
        f.write(yaml_content)
        
    print(f"Loading YOLOv8 model: {model_path}")
    model = YOLO(model_path)
    
    # 1. Validate on Val split (10 images, paper setup)
    print("\n--- 1. Evaluating on Paper Val Split (10 images) ---")
    val_10_yaml = os.path.join(data_dir, "data_val10.yaml")
    # rewrite val10 yaml with absolute paths
    abs_val_lines = []
    with open(val_txt) as f:
        for l in f:
            l = l.strip()
            if l.startswith("./"):
                l = os.path.join(data_dir, l[2:])
            abs_val_lines.append(l + "\n")
    val_10_txt = os.path.join(data_dir, "val_10.txt")
    with open(val_10_txt, "w") as f:
        f.writelines(abs_val_lines)
    with open(val_10_yaml, "w") as f:
        f.write(f"path: {data_dir}\ntrain: {val_10_txt}\nval: {val_10_txt}\nnames:\n  0: cow\n")
        
    metrics_val10 = model.val(data=val_10_yaml, split="val", imgsz=640, device=0)
    
    # 2. Validate on All 400 images (Train + Val combined)
    print("\n--- 2. Evaluating on All 400 Images (Train + Val Combined) ---")
    metrics_all = model.val(data=yaml_path, split="val", imgsz=640, device=0)
    
    print("\n==========================================================")
    print("                    YOLOv8 BENCHMARK RESULTS              ")
    print("==========================================================")
    print(f"Paper Val Split (10 images):")
    print(f"  Precision:    {metrics_val10.box.mp:.4f}")
    print(f"  Recall:       {metrics_val10.box.mr:.4f}")
    print(f"  mAP@0.50:     {metrics_val10.box.map50:.4f}")
    print(f"  mAP@0.50:0.95:{metrics_val10.box.map:.4f}")
    print("----------------------------------------------------------")
    print(f"All 400 Images (Train + Val Combined):")
    print(f"  Precision:    {metrics_all.box.mp:.4f}")
    print(f"  Recall:       {metrics_all.box.mr:.4f}")
    print(f"  mAP@0.50:     {metrics_all.box.map50:.4f}")
    print(f"  mAP@0.50:0.95:{metrics_all.box.map:.4f}")
    print("==========================================================")

if __name__ == "__main__":
    main()
