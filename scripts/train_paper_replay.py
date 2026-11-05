import os
import sys
from mmengine.registry import init_default_scope
init_default_scope('mmpose')
from mmengine.config import Config
from mmengine.runner import Runner
from mmpose.registry import DATASETS
from mmpose.datasets.datasets.base import BaseCocoStyleDataset

# Load config
cfg_file = 'td-hm_hrnet-w48_8xb64-210e_ap10k-256x256.py'
cfg = Config.fromfile(cfg_file)

# Register custom dataset AP10KCowDataset with METAINFO from config
@DATASETS.register_module(name='AP10KCowDataset', force=True)
class AP10KCowDataset(BaseCocoStyleDataset):
    """AP10K Cow Dataset based on BaseCocoStyleDataset."""
    METAINFO = cfg.dataset_info

def main():
    # Dataset path adjustment
    dataset_dir = 'data/cow_barn_kpt_it2_corr_300_20'
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
    cfg.load_from = './checkpoints/hrnet_w48_ap10k_256x256-d95ab412_20211029.pth'
    
    # Output directory for the retraining run
    cfg.work_dir = 'output/hrnet_w48_ap10k_300_20_retrained'
    
    # Disable WandB hook if not logged in to prevent blocking in headless batch jobs
    if 'log_config' in cfg and 'hooks' in cfg.log_config:
        cfg.log_config.hooks = [h for h in cfg.log_config.hooks if h.get('type') != 'WandbLoggerHook']
    
    print('Starting MMPose runner with config:')
    print(f'  Dataset: {cfg.data_root}')
    print(f'  Pretrained weights: {cfg.load_from}')
    print(f'  Output work_dir: {cfg.work_dir}')
    print(f'  Total epochs: {cfg.train_cfg.max_epochs}')
    
    runner = Runner.from_cfg(cfg)
    runner.train()

if __name__ == '__main__':
    main()
