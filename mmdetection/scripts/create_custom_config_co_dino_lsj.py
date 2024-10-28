from mmengine import Config
from mmengine.runner import set_random_seed

import sys
sys.path.append('..')

# 기본 설정 파일 로드
config_name = 'co_dino_5scale_swin_l_lsj_16xb1_1x_coco_nomask'
cfg = Config.fromfile(f'../projects/CO-DETR/configs/codino/{config_name}.py')

# 사용자 정의 설정
cfg.load_from = '../checkpoints/co_dino_5scale_lsj_swin_large_1x_coco-3af73af2.pth'

cfg.train_dataloader.batch_size = 1
cfg.train_dataloader.num_workers = 8

cfg.model.backbone.frozen_stages = -1

cfg.max_epochs = 12
cfg.train_cfg.max_epochs = cfg.max_epochs

cfg.metainfo = {
    'classes': ("veh_go", "veh_goLeft", "veh_noSign", "veh_stop", "veh_stopLeft",
                "veh_stopWarning", "veh_warning", "ped_go", "ped_noSign", "ped_stop",
                "bus_go", "bus_noSign", "bus_stop", "bus_warning")
}

cfg.data_root = '../../tld_db'
cfg.train_dataloader.dataset.dataset.ann_file = 'json/train_coco.json'
cfg.train_dataloader.dataset.dataset.data_root = cfg.data_root
cfg.train_dataloader.dataset.dataset.data_prefix.img = ''
cfg.train_dataloader.dataset.dataset.metainfo = cfg.metainfo

cfg.val_dataloader.dataset.ann_file = 'json/val_coco.json'
cfg.val_dataloader.dataset.data_root = cfg.data_root
cfg.val_dataloader.dataset.data_prefix.img = ''
cfg.val_dataloader.dataset.metainfo = cfg.metainfo

cfg.test_dataloader = cfg.val_dataloader

cfg.val_evaluator.ann_file = cfg.data_root + '/' + 'json/val_coco.json'
cfg.test_evaluator = cfg.val_evaluator

# cfg.default_hooks.checkpoint.interval = 4

# model weights are saved every 10 intervals, up to two weights are saved at the same time, and the saving strategy is auto
cfg.default_hooks.checkpoint = dict(type='CheckpointHook', interval=1, max_keep_ckpts=1, save_best='auto')

# Interval of reporting indicators
cfg.default_hooks.logger.interval = 1

cfg.device = 'cuda'

set_random_seed(42, deterministic=False)

cfg.visualizer.vis_backends = [
    dict(type='LocalVisBackend'),
    dict(type='WandbVisBackend'), # wandb 시각화, 사용시 wandb 계정 필요, 사용하지 않을 시 주석처리
]

custom_config_name = config_name + '_'
cfg.work_dir = f'../work_dirs/{custom_config_name}'

# 설정 파일 저장
config_path = f'../custom_configs/{custom_config_name}.py'
cfg.dump(config_path)

print(f"Custom config saved to {config_path}")