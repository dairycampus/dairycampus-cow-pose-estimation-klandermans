"""
Empirical Benchmark & Provenance Verification Evaluator
Evaluates HRNet-W48 pose checkpoints on the 20-image validation split
and the 320-image full dataset to verify data lineage and reference metrics.
"""

import os
import sys
import argparse
import pandas as pd
from mmengine.registry import init_default_scope
init_default_scope('mmpose')
from mmengine.config import Config
from mmengine.runner import Runner
from mmpose.registry import DATASETS
from mmpose.datasets.datasets.base import BaseCocoStyleDataset

def parse_args():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(script_dir)
    
    parser = argparse.ArgumentParser(description="Evaluate Bovine Pose Model Lineage and Metrics")
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to .pth checkpoint')
    parser.add_argument('--config', type=str, default=os.path.join(repo_root, 'configs', 'td-hm_hrnet-w48_8xb64-210e_ap10k-256x256.py'), help='Model config file')
    parser.add_argument('--model-name', type=str, default='EvaluationModel', help='Label for CSV output')
    parser.add_argument('--split', type=str, choices=['val', 'all', 'both'], default='both', help='Evaluation split')
    parser.add_argument('--val-data-root', type=str, default='data/cow_barn_kpt_it2_corr_300_20', help='Path to 300_20 dataset directory')
    parser.add_argument('--all-data-root', type=str, default='data/cow_kpt_320', help='Path to 320 combined dataset directory')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size for inference')
    parser.add_argument('--out-csv', type=str, default=os.path.join(repo_root, 'results', 'provenance_eval_results.csv'), help='Output CSV path')
    return parser.parse_args()

def run_split_eval(cfg_file, checkpoint_path, model_name, eval_split, data_root, batch_size):
    cfg = Config.fromfile(cfg_file)
    
    @DATASETS.register_module(name='AP10KCowDataset', force=True)
    class AP10KCowDataset(BaseCocoStyleDataset):
        METAINFO = cfg.dataset_info
        
    if eval_split == 'val':
        ann_file = f'{data_root}/val/annotations/instances_default.json'
        img_prefix = f'{data_root}/val/images/'
        desc = 'Val Split (20 images, 154 cows)'
    else:
        ann_file = f'{data_root}/annotations/instances_all_320.json'
        img_prefix = f'{data_root}/images/'
        desc = 'Full Dataset (320 images, 2525 cows)'
        
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
        dict(type='CocoMetric', ann_file=ann_file, score_mode='keypoint', keypoint_score_thr=0.0, prefix='coco'),
        dict(type='PCKAccuracy', thr=0.05, norm_item=['bbox'], prefix='pck05'),
        dict(type='PCKAccuracy', thr=0.10, norm_item=['bbox'], prefix='pck10'),
        dict(type='PCKAccuracy', thr=0.20, norm_item=['bbox'], prefix='pck20'),
    ]
    
    cfg.load_from = checkpoint_path
    cfg.work_dir = f'work_dirs/eval_{eval_split}_{model_name}'
    
    print(f"\nEvaluating {model_name} on {desc}...")
    runner = Runner.from_cfg(cfg)
    metrics = runner.test()
    
    row = {
        'model': model_name,
        'split': eval_split,
        'checkpoint': os.path.basename(checkpoint_path),
        'coco/AP': metrics.get('coco/AP', 0.0) * 100,
        'coco/AP .50': metrics.get('coco/AP .50', 0.0) * 100,
        'coco/AP .75': metrics.get('coco/AP .75', 0.0) * 100,
        'coco/AR': metrics.get('coco/AR', 0.0) * 100,
        'pck05/PCK': metrics.get('pck05/PCK', 0.0) * 100,
        'pck10/PCK': metrics.get('pck10/PCK', 0.0) * 100,
        'pck20/PCK': metrics.get('pck20/PCK', 0.0) * 100,
    }
    return row

def main():
    args = parse_args()
    
    splits_to_eval = ['val', 'all'] if args.split == 'both' else [args.split]
    results = []
    
    for s in splits_to_eval:
        data_root = args.val_data_root if s == 'val' else args.all_data_root
        row = run_split_eval(args.config, args.checkpoint, args.model_name, s, data_root, args.batch_size)
        results.append(row)
        
    df = pd.DataFrame(results)
    print("\nBenchmark Evaluation Results:")
    print(df.to_string(index=False))
    
    os.makedirs(os.path.dirname(args.out_csv) or '.', exist_ok=True)
    df.to_csv(args.out_csv, index=False)
    print(f"\nSaved metrics to: {args.out_csv}")

if __name__ == '__main__':
    main()
