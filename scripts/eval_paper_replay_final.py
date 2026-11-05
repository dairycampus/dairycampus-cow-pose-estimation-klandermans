import os
import sys
import torch
import pandas as pd
from mmengine.registry import init_default_scope
init_default_scope('mmpose')
from mmengine.config import Config
from mmengine.runner import Runner
from mmpose.registry import DATASETS
from mmpose.datasets.datasets.base import BaseCocoStyleDataset

cfg_file = '/home/WUR/kland001/retraining/td-hm_hrnet-w48_8xb64-210e_ap10k-256x256.py'
base_cfg = Config.fromfile(cfg_file)

@DATASETS.register_module(name='AP10KCowDataset', force=True)
class AP10KCowDataset(BaseCocoStyleDataset):
    METAINFO = base_cfg.dataset_info

def evaluate_model(checkpoint_path, model_name, eval_split='all'):
    cfg = Config.fromfile(cfg_file)
    
    if eval_split == 'val':
        data_root = '/home/WUR/kland001/retraining/data/cow_barn_kpt_it2_corr_300_20'
        ann_file = f'{data_root}/val/annotations/instances_default.json'
        img_prefix = f'{data_root}/val/images/'
        desc = 'Val (20 imgs)'
    else:
        data_root = '/lustre/scratch/WUR/ABGC/kland001/benchmark/data/cow_kpt_320'
        ann_file = f'{data_root}/annotations/instances_all_320.json'
        img_prefix = f'{data_root}/images/'
        desc = 'All 320 imgs (Train+Val)'
        
    cfg.data_root = data_root
    cfg.test_dataloader.dataset.data_root = data_root
    cfg.test_dataloader.dataset.ann_file = ann_file
    cfg.test_dataloader.dataset.data_prefix = dict(img=img_prefix)
    cfg.test_dataloader.dataset.metainfo = cfg.dataset_info
    cfg.test_dataloader.batch_size = 32
    cfg.test_dataloader.num_workers = 4
    
    cfg.test_evaluator = [
        dict(
            type='CocoMetric',
            ann_file=ann_file,
            score_mode='keypoint',
            keypoint_score_thr=0.0,
            prefix='coco'
        ),
        dict(
            type='PCKAccuracy',
            thr=0.05,
            norm_item=['bbox'],
            prefix='pck05'
        ),
        dict(
            type='PCKAccuracy',
            thr=0.10,
            norm_item=['bbox'],
            prefix='pck10'
        ),
        dict(
            type='PCKAccuracy',
            thr=0.20,
            norm_item=['bbox'],
            prefix='pck20'
        )
    ]
    
    cfg.load_from = checkpoint_path
    cfg.resume = False
    cfg.work_dir = f'/lustre/scratch/WUR/ABGC/kland001/benchmark/eval_logs/{model_name}_{eval_split}'
    cfg.default_hooks.checkpoint = None
    cfg.default_hooks.logger = dict(type='LoggerHook', interval=50)
    
    runner = Runner.from_cfg(cfg)
    metrics = runner.test()
    
    return {
        'Model': model_name,
        'Dataset': desc,
        'mAP': round(metrics.get('coco/AP', 0.0) * 100, 2),
        'AP@0.50': round(metrics.get('coco/AP .50', 0.0) * 100, 2),
        'AP@0.75': round(metrics.get('coco/AP .75', 0.0) * 100, 2),
        'AR': round(metrics.get('coco/AR', 0.0) * 100, 2),
        'PCK@0.05': round(metrics.get('pck05/PCK', 0.0) * 100, 2),
        'PCK@0.10': round(metrics.get('pck10/PCK', 0.0) * 100, 2),
        'PCK@0.20': round(metrics.get('pck20/PCK', 0.0) * 100, 2),
    }

def main():
    models = [
        ('/lustre/scratch/WUR/ABGC/kland001/output/hrnet_w48_ap10k_300_20_retrained/best_coco_AP_epoch_180.pth', 'Paper_Replay_bs2_BEST_ep180'),
        ('/lustre/scratch/WUR/ABGC/kland001/output/hrnet_w48_ap10k_300_20_retrained/epoch_210.pth', 'Paper_Replay_bs2_FINAL_ep210'),
    ]
    
    out_csv = '/lustre/scratch/WUR/ABGC/kland001/benchmark/mmpose_benchmark_results.csv'
    if os.path.exists(out_csv):
        df_existing = pd.read_csv(out_csv)
    else:
        df_existing = pd.DataFrame()
        
    new_results = []
    for ckpt_path, name in models:
        for split in ['val', 'all']:
            print(f"Evaluating {name} on {split}...")
            res = evaluate_model(ckpt_path, name, eval_split=split)
            new_results.append(res)
            
    df_new = pd.DataFrame(new_results)
    df_combined = pd.concat([df_existing, df_new], ignore_index=True)
    df_combined.drop_duplicates(subset=['Model', 'Dataset'], keep='last', inplace=True)
    df_combined.to_csv(out_csv, index=False)
    
    print("\n================ UPDATED BENCHMARK TABLE ================")
    try:
        print(df_combined.to_markdown(index=False))
    except Exception:
        print(df_combined.to_string(index=False))

if __name__ == '__main__':
    main()
