"""
End-to-End Cow Pose Estimation Demo Script
Detects cows using YOLOv8 and predicts 4 anatomical keypoints using HRNet-W48.
Keypoints:
  0: Left Ear
  1: Right Ear
  2: Withers (Schoft)
  3: Tail Base (Staartinplant)
"""

import os
import cv2
import torch
import argparse
import numpy as np
from ultralytics import YOLO
from mmengine.config import Config
from mmengine.runner import Runner
from mmpose.apis import init_model, inference_topdown

KEYPOINT_NAMES = ['Left Ear', 'Right Ear', 'Withers', 'Tail Base']
SKELETON = [(0, 1), (0, 2), (1, 2), (2, 3)]  # Ear-to-ear, ears-to-withers, withers-to-tail
COLORS = [(0, 255, 0), (0, 255, 0), (0, 0, 255), (255, 0, 0)]

def parse_args():
    parser = argparse.ArgumentParser(description="End-to-end Cow Pose Estimation")
    parser.add_argument('--input', type=str, required=True, help='Path to image or video file')
    parser.add_argument('--yolo-weights', type=str, default='models/best_yolov8_390_10.pt', help='YOLO weights')
    parser.add_argument('--pose-config', type=str, default='configs/td-hm_hrnet-w48_8xb64-210e_ap10k-256x256.py', help='Pose config')
    parser.add_argument('--pose-weights', type=str, default='models/best_coco_AP_epoch_180.pth', help='MMPose weights')
    parser.add_argument('--output', type=str, default='results/output_demo.jpg', help='Output visualization path')
    parser.add_argument('--device', type=str, default='cuda:0' if torch.cuda.is_available() else 'cpu')
    return parser.parse_args()

def draw_cow_pose(image, bboxes, keypoints_list):
    vis_img = image.copy()
    for bbox, (kpts, scores) in zip(bboxes, keypoints_list):
        x1, y1, x2, y2 = map(int, bbox[:4])
        cv2.rectangle(vis_img, (x1, y1), (x2, y2), (0, 255, 255), 2)
        
        # Draw skeleton lines
        for p1, p2 in SKELETON:
            if scores[p1] > 0.3 and scores[p2] > 0.3:
                pt1 = (int(kpts[p1][0]), int(kpts[p1][1]))
                pt2 = (int(kpts[p2][0]), int(kpts[p2][1]))
                cv2.line(vis_img, pt1, pt2, (255, 255, 0), 2)
                
        # Draw keypoints
        for idx, (pt, sc) in enumerate(zip(kpts, scores)):
            if sc > 0.3:
                center = (int(pt[0]), int(pt[1]))
                cv2.circle(vis_img, center, 5, COLORS[idx], -1)
                cv2.circle(vis_img, center, 6, (0, 0, 0), 1)
    return vis_img

def main():
    args = parse_args()
    print(f"Loading YOLOv8 from {args.yolo_weights} on {args.device}...")
    yolo_model = YOLO(args.yolo_weights)
    
    print(f"Loading MMPose HRNet from {args.pose_weights} on {args.device}...")
    pose_model = init_model(args.pose_config, args.pose_weights, device=args.device)
    
    img = cv2.imread(args.input)
    if img is None:
        raise FileNotFoundError(f"Could not load image: {args.input}")
        
    # Step 1: Detect cows
    yolo_results = yolo_model(img, conf=0.4, device=args.device)[0]
    bboxes = yolo_results.boxes.xyxy.cpu().numpy()
    print(f"Detected {len(bboxes)} cows.")
    
    # Step 2: Pose estimation per cow
    pose_results = inference_topdown(pose_model, img, bboxes)
    keypoints_list = []
    for res in pose_results:
        kpts = res.pred_instances.keypoints[0]
        scores = res.pred_instances.keypoint_scores[0]
        keypoints_list.append((kpts, scores))
        
    # Step 3: Visualize
    out_img = draw_cow_pose(img, bboxes, keypoints_list)
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    cv2.imwrite(args.output, out_img)
    print(f"Visualization saved to {args.output}")

if __name__ == '__main__':
    main()
