"""
HRNet-W48 210-Epoch Exact Baseline Retraining Script
Replays the exact training schedule of the historical WLR bovine pose estimation model.
"""

import os
import sys
import argparse
from mmengine.registry import init_default_scope
init_default_scope('mmpose')
from mmengine.config import Config
from mmengine.runner import Runner
from mmpose.registry import DATASETS
from mmpose.datasets.datasets.base import BaseCocoStyleDataset

def parse_args():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(script_dir)
    
    parser = argparse.ArgumentParser(description="Retrain HRNet-W48 Pose Estimation Baseline")
    parser.add_argument('--config', type=str, default=os.path.join(repo_root, 'configs', 'td-hm_hrnet-w48_8xb64-210e_ap10k-256x256.py'), help='MMPose config file')
    parser.add_argument('--data-root', type=str, default='data/cow_barn_kpt_it2_corr_300_20', help='Dataset root path')
    parser.add_argument('--pretrained', type=str, default='checkpoints/hrnet_w48_ap10k_256x256-d95ab412_20211029.pth', help='AP-10K pretrained backbone checkpoint')
    parser.add_argument('--work-dir', type=str, default='output/hrnet_w48_ap10k_300_20_retrained', help='Training output directory')
    parser.add_argument('--epochs', type=int, default=210, help='Total training epochs')
    parser.add_argument('--batch-size', type=int, default=2, help='Batch size per GPU')
    parser.add_argument('--lr', type=float, default=5e-4, help='Initial learning rate')
    return parser.parse_args()

def main():
    args = parse_args()
    
    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Config file not found: {args.config}")
        
    cfg = Config.fromfile(args.config)
    
    @DATASETS.register_module(name='AP10KCowDataset', force=True)
    class AP10KCowDataset(BaseCocoStyleDataset):
        """AP10K Cow Dataset based on BaseCocoStyleDataset."""
        METAINFO = cfg.dataset_info

    # Dataset path adjustment
    dataset_dir = args.data_root
    cfg.data_root = dataset_dir
    cfg.data_name = 'cow_barn_kpt_it2_corr_300_20'
    
    cfg.train_dataloader.dataset.data_root = dataset_dir
    cfg.val_dataloader.dataset.data_root = dataset_dir
    cfg.test_dataloader.dataset.data_root = dataset_dir
    
    cfg.train_dataloader.dataset.metainfo = cfg.dataset_info
    cfg.val_dataloader.dataset.metainfo = cfg.dataset_info
    cfg.test_dataloader.dataset.metainfo = cfg.dataset_info
    
    cfg.val_evaluator.ann_file = f'{dataset_dir}/val/annotations/instances_default.json'
    cfg.test_evaluator.ann_file = f'{dataset_dir}/val/annotations/instances_default.json'
    
    # Pretrained checkpoint path
    if os.path.exists(args.pretrained):
        cfg.load_from = args.pretrained
    else:
        print(f"Notice: Pretrained checkpoint not found at {args.pretrained}. Training without pretrained weights.")
        cfg.load_from = None
        
    # Schedule & Optimizer
    cfg.work_dir = args.work_dir
    cfg.train_cfg.max_epochs = args.epochs
    cfg.train_dataloader.batch_size = args.batch_size
    cfg.optim_wrapper.optimizer.lr = args.lr
    
    # Disable WandB hook if offline / headless
    if 'log_config' in cfg and 'hooks' in cfg.log_config:
        cfg.log_config.hooks = [h for h in cfg.log_config.hooks if h.get('type') != 'WandbLoggerHook']
        
    print("==========================================================")
    print("   HRNet-W48 COW POSE REPRODUCIBILITY RETRAINING RUN       ")
    print("==========================================================")
    print(f"  Config:        {args.config}")
    print(f"  Dataset:       {cfg.data_root}")
    print(f"  Work Dir:      {cfg.work_dir}")
    print(f"  Epochs:        {cfg.train_cfg.max_epochs}")
    print(f"  Batch size:    {cfg.train_dataloader.batch_size}")
    print(f"  Initial LR:    {cfg.optim_wrapper.optimizer.lr}")
    print("==========================================================")
    
    runner = Runner.from_cfg(cfg)
    runner.train()

if __name__ == '__main__':
    main()
