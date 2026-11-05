import os
import sys
from mmengine.registry import init_default_scope
init_default_scope('mmpose')
from mmengine.config import Config
from mmengine.runner import Runner
from mmpose.registry import DATASETS
from mmpose.datasets.datasets.base import BaseCocoStyleDataset

# Load base config
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
    cfg.load_from = None
    
    # ==========================================================
    # SCALED BATCH SIZE & LEARNING RATE FOR A100 80GB
    # ==========================================================
    cfg.train_dataloader.batch_size = 128
    cfg.train_dataloader.num_workers = 8
    cfg.val_dataloader.batch_size = 64
    cfg.val_dataloader.num_workers = 4
    
    # Scaled Learning Rate (0.002 instead of 0.0005)
    cfg.optim_wrapper.optimizer.lr = 0.002
    
    # Warmup scheduler: warmup over first 5 epochs (~95 iterations with bs=128)
    cfg.param_scheduler = [
        dict(
            type='LinearLR', begin=0, end=95, start_factor=0.001, by_epoch=False),
        dict(
            type='MultiStepLR',
            begin=0,
            end=210,
            milestones=[170, 200],
            gamma=0.1,
            by_epoch=True)
    ]
    
    # Output directory
    cfg.work_dir = 'output/hrnet_w48_ap10k_300_20_bs128_lr002'
    
    # RESUME FROM EPOCH 100
    cfg.resume = True
    
    # Keep only the last 3 checkpoints to keep clean
    cfg.default_hooks.checkpoint = dict(
        type='CheckpointHook',
        interval=10,
        save_best='coco/AP',
        rule='greater',
        max_keep_ckpts=3
    )
    
    # Disable WandB hook if present
    if 'log_config' in cfg and 'hooks' in cfg.log_config:
        cfg.log_config.hooks = [h for h in cfg.log_config.hooks if h.get('type') != 'WandbLoggerHook']
    
    print('========================================================')
    print('MMPose Runner: Resuming High Throughput from Epoch 100')
    print(f'  Dataset: {cfg.data_root}')
    print(f'  Batch size (Train): {cfg.train_dataloader.batch_size}')
    print(f'  Output work_dir: {cfg.work_dir}')
    print(f'  Resume: {cfg.resume}')
    print(f'  Total epochs: {cfg.train_cfg.max_epochs}')
    print('========================================================')
    
    runner = Runner.from_cfg(cfg)
    runner.train()

if __name__ == '__main__':
    main()
