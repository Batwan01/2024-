from mmengine import ConfigDict

auto_scale_lr = dict(base_batch_size=16)
backend_args = None
batch_augments = [
    dict(pad_mask=True, size=(
        2048,
        2048,
    ), type='BatchFixedSizePad'),
]
custom_imports = dict(
    allow_failed_imports=False, imports=[
        'projects.CO-DETR.codetr',
    ])
data_root = '../../tld_db'
dataset_type = 'CocoDataset'
default_hooks = dict(
    checkpoint=dict(
        interval=1, max_keep_ckpts=1, save_best='auto', type='CheckpointHook'),
    logger=dict(_scope_='mmdet', interval=1, type='LoggerHook'),
    param_scheduler=dict(_scope_='mmdet', type='ParamSchedulerHook'),
    sampler_seed=dict(_scope_='mmdet', type='DistSamplerSeedHook'),
    timer=dict(_scope_='mmdet', type='IterTimerHook'),
    visualization=dict(_scope_='mmdet', type='DetVisualizationHook'))
default_scope = 'mmdet'
device = 'cuda'
env_cfg = dict(
    cudnn_benchmark=False,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
image_size = (
    2048,
    2048,
)
load_from = '../checkpoints/cascade_1ep.pth'

load_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(
        keep_ratio=True,
        ratio_range=(
            0.1,
            2.0,
        ),
        scale=(
            1024,
            1024,
        ),
        type='RandomResize'),
    dict(
        allow_negative_crop=True,
        crop_size=(
            1024,
            1024,
        ),
        crop_type='absolute_range',
        recompute_bbox=True,
        type='RandomCrop'),
    dict(min_gt_bbox_wh=(
        0.01,
        0.01,
    ), type='FilterAnnotations'),
    dict(prob=0.5, type='RandomFlip'),
    dict(pad_val=dict(img=(
        114,
        114,
        114,
    )), size=(
        1024,
        1024,
    ), type='Pad'),
]
log_level = 'INFO'
log_processor = dict(
    _scope_='mmdet', by_epoch=True, type='LogProcessor', window_size=50)
loss_lambda = 2.0
max_epochs = 1
max_iters = 270000
metainfo = dict(
    classes=(
        'veh_go',
        'veh_goLeft',
        'veh_noSign',
        'veh_stop',
        'veh_stopLeft',
        'veh_stopWarning',
        'veh_warning',
        'ped_go',
        'ped_noSign',
        'ped_stop',
        'bus_go',
        'bus_noSign',
        'bus_stop',
        'bus_warning',
    ))

pretrained ='https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_large_patch4_window12_384_22k.pth'

model = dict(
    type='CascadeRCNN',
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_size_divisor=1),
    backbone=dict(
        type='SwinTransformer',
        pretrain_img_size=384,
        embed_dims=192,
        depths=[2, 2, 18, 2],
        num_heads=[6, 12, 24, 48],
        window_size=12,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.2,
        patch_norm=True,
        out_indices=(0, 1, 2, 3),
        with_cp=False,
        convert_weights=True,
        init_cfg=dict(type='Pretrained', checkpoint=pretrained)),
    neck=dict(
        type='FPN',
        in_channels=[192, 384, 768, 1536],
        out_channels=256,
        num_outs=5),
    rpn_head=dict(
        type='RPNHead',
        in_channels=256,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[8],
            ratios=[0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32, 64]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0)),
    roi_head=dict(
        type='CascadeRoIHead',
        num_stages=3,
        stage_loss_weights=[1, 0.5, 0.25],
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        bbox_head=[
            dict(
                type='Shared2FCBBoxHead',
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=14
                ,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.1, 0.1, 0.2, 0.2]),
                reg_class_agnostic=True,
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.0),
                loss_bbox=dict(type='SmoothL1Loss', beta=1.0,
                               loss_weight=1.0)),
            dict(
                type='Shared2FCBBoxHead',
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=14
                ,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.05, 0.05, 0.1, 0.1]),
                reg_class_agnostic=True,
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.0),
                loss_bbox=dict(type='SmoothL1Loss', beta=1.0,
                               loss_weight=1.0)),
            dict(
                type='Shared2FCBBoxHead',
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=14
                
                
                
                ,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.033, 0.033, 0.067, 0.067]),
                reg_class_agnostic=True,
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.0),
                loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0))
        ]),
    # model training and testing settings
    train_cfg=dict(
        rpn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.7,
                neg_iou_thr=0.3,
                min_pos_iou=0.3,
                match_low_quality=True,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=256,
                pos_fraction=0.5,
                neg_pos_ub=-1,
                add_gt_as_proposals=False),
            allowed_border=0,
            pos_weight=-1,
            debug=False),
        rpn_proposal=dict(
            nms_pre=2000,
            max_per_img=2000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=[
            dict(
                assigner=dict(
                    type='MaxIoUAssigner',
                    pos_iou_thr=0.5,
                    neg_iou_thr=0.5,
                    min_pos_iou=0.5,
                    match_low_quality=False,
                    ignore_iof_thr=-1),
                sampler=dict(
                    type='RandomSampler',
                    num=512,
                    pos_fraction=0.25,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=True),
                pos_weight=-1,
                debug=False),
            dict(
                assigner=dict(
                    type='MaxIoUAssigner',
                    pos_iou_thr=0.6,
                    neg_iou_thr=0.6,
                    min_pos_iou=0.6,
                    match_low_quality=False,
                    ignore_iof_thr=-1),
                sampler=dict(
                    type='RandomSampler',
                    num=512,
                    pos_fraction=0.25,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=True),
                pos_weight=-1,
                debug=False),
            dict(
                assigner=dict(
                    type='MaxIoUAssigner',
                    pos_iou_thr=0.7,
                    neg_iou_thr=0.7,
                    min_pos_iou=0.7,
                    match_low_quality=False,
                    ignore_iof_thr=-1),
                sampler=dict(
                    type='RandomSampler',
                    num=512,
                    pos_fraction=0.25,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=True),
                pos_weight=-1,
                debug=False)
        ]),
    test_cfg=dict(
        rpn=dict(
            nms_pre=1000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
            score_thr=0.00,
            nms=dict(type='nms', iou_threshold=0.5),
            max_per_img=100)))


num_classes = 14
num_dec_layer = 6
optim_wrapper = dict(
    clip_grad=dict(max_norm=0.1, norm_type=2),
    optimizer=dict(lr=0.0001, type='AdamW', weight_decay=0.0001),
    paramwise_cfg=dict(custom_keys=dict(backbone=dict(lr_mult=0.1))),
    type='OptimWrapper')
param_scheduler = [
    dict(
        begin=0,
        by_epoch=True,
        end=16,
        gamma=0.1,
        milestones=[
            8,
        ],
        type='MultiStepLR'),
]
pretrained = 'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_large_patch4_window12_384_22k.pth'
resume = False
test_cfg = dict(_scope_='mmdet', type='TestLoop')
test_dataloader = dict(
    batch_size=1,
    dataset=dict(
        _scope_='mmdet',
        ann_file='json/val_coco.json',
        backend_args=None,
        data_prefix=dict(img=''),
        data_root='../../tld_db',
        metainfo=dict(
            classes=(
                'veh_go',
                'veh_goLeft',
                'veh_noSign',
                'veh_stop',
                'veh_stopLeft',
                'veh_stopWarning',
                'veh_warning',
                'ped_go',
                'ped_noSign',
                'ped_stop',
                'bus_go',
                'bus_noSign',
                'bus_stop',
                'bus_warning',
            )),
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(keep_ratio=True, scale=(
                2048,
                1280,
            ), type='Resize'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(
                meta_keys=(
                    'img_id',
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'scale_factor',
                ),
                type='PackDetInputs'),
        ],
        test_mode=True,
        type='CocoDataset'),
    drop_last=False,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(_scope_='mmdet', shuffle=False, type='DefaultSampler'))
test_evaluator = dict(
    _scope_='mmdet',
    ann_file='../../tld_db/json/val_coco.json',
    backend_args=None,
    format_only=False,
    metric='bbox',
    type='CocoMetric')
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(keep_ratio=True, scale=(
        2048,
        1280,
    ), type='Resize'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        meta_keys=(
            'img_id',
            'img_path',
            'ori_shape',
            'img_shape',
            'scale_factor',
        ),
        type='PackDetInputs'),
]
train_cfg = dict(max_epochs=1, type='EpochBasedTrainLoop', val_interval=1)
train_dataloader = dict(
    batch_size=1,
    dataset=dict(
        ann_file='json/train_coco.json',
        backend_args=None,
        data_prefix=dict(img=''),
        data_root='../../tld_db',
        filter_cfg=dict(filter_empty_gt=False, min_size=32),
        metainfo=dict(
            classes=(
                'veh_go',
                'veh_goLeft',
                'veh_noSign',
                'veh_stop',
                'veh_stopLeft',
                'veh_stopWarning',
                'veh_warning',
                'ped_go',
                'ped_noSign',
                'ped_stop',
                'bus_go',
                'bus_noSign',
                'bus_stop',
                'bus_warning',
            )),
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(prob=0.5, type='RandomFlip'),
            dict(
                transforms=[
                    [
                        dict(
                            keep_ratio=True,
                            scales=[
                                (
                                    480,
                                    2048,
                                ),
                                (
                                    512,
                                    2048,
                                ),
                                (
                                    544,
                                    2048,
                                ),
                                (
                                    576,
                                    2048,
                                ),
                                (
                                    608,
                                    2048,
                                ),
                                (
                                    640,
                                    2048,
                                ),
                                (
                                    672,
                                    2048,
                                ),
                                (
                                    704,
                                    2048,
                                ),
                                (
                                    736,
                                    2048,
                                ),
                                (
                                    768,
                                    2048,
                                ),
                                (
                                    800,
                                    2048,
                                ),
                                (
                                    832,
                                    2048,
                                ),
                                (
                                    864,
                                    2048,
                                ),
                                (
                                    896,
                                    2048,
                                ),
                                (
                                    928,
                                    2048,
                                ),
                                (
                                    960,
                                    2048,
                                ),
                                (
                                    992,
                                    2048,
                                ),
                                (
                                    1024,
                                    2048,
                                ),
                                (
                                    1056,
                                    2048,
                                ),
                                (
                                    1088,
                                    2048,
                                ),
                                (
                                    1120,
                                    2048,
                                ),
                                (
                                    1152,
                                    2048,
                                ),
                                (
                                    1184,
                                    2048,
                                ),
                                (
                                    1216,
                                    2048,
                                ),
                                (
                                    1248,
                                    2048,
                                ),
                                (
                                    1280,
                                    2048,
                                ),
                                (
                                    1312,
                                    2048,
                                ),
                                (
                                    1344,
                                    2048,
                                ),
                                (
                                    1376,
                                    2048,
                                ),
                                (
                                    1408,
                                    2048,
                                ),
                                (
                                    1440,
                                    2048,
                                ),
                                (
                                    1472,
                                    2048,
                                ),
                                (
                                    1504,
                                    2048,
                                ),
                                (
                                    1536,
                                    2048,
                                ),
                            ],
                            type='RandomChoiceResize'),
                    ],
                    [
                        dict(
                            keep_ratio=True,
                            scales=[
                                (
                                    400,
                                    4200,
                                ),
                                (
                                    500,
                                    4200,
                                ),
                                (
                                    600,
                                    4200,
                                ),
                            ],
                            type='RandomChoiceResize'),
                        dict(
                            allow_negative_crop=True,
                            crop_size=(
                                384,
                                600,
                            ),
                            crop_type='absolute_range',
                            type='RandomCrop'),
                        dict(
                            keep_ratio=True,
                            scales=[
                                (
                                    480,
                                    2048,
                                ),
                                (
                                    512,
                                    2048,
                                ),
                                (
                                    544,
                                    2048,
                                ),
                                (
                                    576,
                                    2048,
                                ),
                                (
                                    608,
                                    2048,
                                ),
                                (
                                    640,
                                    2048,
                                ),
                                (
                                    672,
                                    2048,
                                ),
                                (
                                    704,
                                    2048,
                                ),
                                (
                                    736,
                                    2048,
                                ),
                                (
                                    768,
                                    2048,
                                ),
                                (
                                    800,
                                    2048,
                                ),
                                (
                                    832,
                                    2048,
                                ),
                                (
                                    864,
                                    2048,
                                ),
                                (
                                    896,
                                    2048,
                                ),
                                (
                                    928,
                                    2048,
                                ),
                                (
                                    960,
                                    2048,
                                ),
                                (
                                    992,
                                    2048,
                                ),
                                (
                                    1024,
                                    2048,
                                ),
                                (
                                    1056,
                                    2048,
                                ),
                                (
                                    1088,
                                    2048,
                                ),
                                (
                                    1120,
                                    2048,
                                ),
                                (
                                    1152,
                                    2048,
                                ),
                                (
                                    1184,
                                    2048,
                                ),
                                (
                                    1216,
                                    2048,
                                ),
                                (
                                    1248,
                                    2048,
                                ),
                                (
                                    1280,
                                    2048,
                                ),
                                (
                                    1312,
                                    2048,
                                ),
                                (
                                    1344,
                                    2048,
                                ),
                                (
                                    1376,
                                    2048,
                                ),
                                (
                                    1408,
                                    2048,
                                ),
                                (
                                    1440,
                                    2048,
                                ),
                                (
                                    1472,
                                    2048,
                                ),
                                (
                                    1504,
                                    2048,
                                ),
                                (
                                    1536,
                                    2048,
                                ),
                            ],
                            type='RandomChoiceResize'),
                    ],
                ],
                type='RandomChoice'),
            dict(type='PackDetInputs'),
        ],
        type='CocoDataset'),
    num_workers=8,
    persistent_workers=True,
    sampler=dict(_scope_='mmdet', shuffle=True, type='DefaultSampler'))
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(prob=0.5, type='RandomFlip'),
    dict(
        transforms=[
            [
                dict(
                    keep_ratio=True,
                    scales=[
                        (
                            480,
                            2048,
                        ),
                        (
                            512,
                            2048,
                        ),
                        (
                            544,
                            2048,
                        ),
                        (
                            576,
                            2048,
                        ),
                        (
                            608,
                            2048,
                        ),
                        (
                            640,
                            2048,
                        ),
                        (
                            672,
                            2048,
                        ),
                        (
                            704,
                            2048,
                        ),
                        (
                            736,
                            2048,
                        ),
                        (
                            768,
                            2048,
                        ),
                        (
                            800,
                            2048,
                        ),
                        (
                            832,
                            2048,
                        ),
                        (
                            864,
                            2048,
                        ),
                        (
                            896,
                            2048,
                        ),
                        (
                            928,
                            2048,
                        ),
                        (
                            960,
                            2048,
                        ),
                        (
                            992,
                            2048,
                        ),
                        (
                            1024,
                            2048,
                        ),
                        (
                            1056,
                            2048,
                        ),
                        (
                            1088,
                            2048,
                        ),
                        (
                            1120,
                            2048,
                        ),
                        (
                            1152,
                            2048,
                        ),
                        (
                            1184,
                            2048,
                        ),
                        (
                            1216,
                            2048,
                        ),
                        (
                            1248,
                            2048,
                        ),
                        (
                            1280,
                            2048,
                        ),
                        (
                            1312,
                            2048,
                        ),
                        (
                            1344,
                            2048,
                        ),
                        (
                            1376,
                            2048,
                        ),
                        (
                            1408,
                            2048,
                        ),
                        (
                            1440,
                            2048,
                        ),
                        (
                            1472,
                            2048,
                        ),
                        (
                            1504,
                            2048,
                        ),
                        (
                            1536,
                            2048,
                        ),
                    ],
                    type='RandomChoiceResize'),
            ],
            [
                dict(
                    keep_ratio=True,
                    scales=[
                        (
                            400,
                            4200,
                        ),
                        (
                            500,
                            4200,
                        ),
                        (
                            600,
                            4200,
                        ),
                    ],
                    type='RandomChoiceResize'),
                dict(
                    allow_negative_crop=True,
                    crop_size=(
                        384,
                        600,
                    ),
                    crop_type='absolute_range',
                    type='RandomCrop'),
                dict(
                    keep_ratio=True,
                    scales=[
                        (
                            480,
                            2048,
                        ),
                        (
                            512,
                            2048,
                        ),
                        (
                            544,
                            2048,
                        ),
                        (
                            576,
                            2048,
                        ),
                        (
                            608,
                            2048,
                        ),
                        (
                            640,
                            2048,
                        ),
                        (
                            672,
                            2048,
                        ),
                        (
                            704,
                            2048,
                        ),
                        (
                            736,
                            2048,
                        ),
                        (
                            768,
                            2048,
                        ),
                        (
                            800,
                            2048,
                        ),
                        (
                            832,
                            2048,
                        ),
                        (
                            864,
                            2048,
                        ),
                        (
                            896,
                            2048,
                        ),
                        (
                            928,
                            2048,
                        ),
                        (
                            960,
                            2048,
                        ),
                        (
                            992,
                            2048,
                        ),
                        (
                            1024,
                            2048,
                        ),
                        (
                            1056,
                            2048,
                        ),
                        (
                            1088,
                            2048,
                        ),
                        (
                            1120,
                            2048,
                        ),
                        (
                            1152,
                            2048,
                        ),
                        (
                            1184,
                            2048,
                        ),
                        (
                            1216,
                            2048,
                        ),
                        (
                            1248,
                            2048,
                        ),
                        (
                            1280,
                            2048,
                        ),
                        (
                            1312,
                            2048,
                        ),
                        (
                            1344,
                            2048,
                        ),
                        (
                            1376,
                            2048,
                        ),
                        (
                            1408,
                            2048,
                        ),
                        (
                            1440,
                            2048,
                        ),
                        (
                            1472,
                            2048,
                        ),
                        (
                            1504,
                            2048,
                        ),
                        (
                            1536,
                            2048,
                        ),
                    ],
                    type='RandomChoiceResize'),
            ],
        ],
        type='RandomChoice'),
    dict(type='PackDetInputs'),
]
val_cfg = dict(_scope_='mmdet', type='ValLoop')
val_dataloader = dict(
    batch_size=1,
    dataset=dict(
        _scope_='mmdet',
        ann_file='json/val_coco.json',
        backend_args=None,
        data_prefix=dict(img=''),
        data_root='../../tld_db',
        metainfo=dict(
            classes=(
                'veh_go',
                'veh_goLeft',
                'veh_noSign',
                'veh_stop',
                'veh_stopLeft',
                'veh_stopWarning',
                'veh_warning',
                'ped_go',
                'ped_noSign',
                'ped_stop',
                'bus_go',
                'bus_noSign',
                'bus_stop',
                'bus_warning',
            )),
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(keep_ratio=True, scale=(
                2048,
                1280,
            ), type='Resize'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(
                meta_keys=(
                    'img_id',
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'scale_factor',
                ),
                type='PackDetInputs'),
        ],
        test_mode=True,
        type='CocoDataset'),
    drop_last=False,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(_scope_='mmdet', shuffle=False, type='DefaultSampler'))
val_evaluator = dict(
    _scope_='mmdet',
    ann_file='../../tld_db/json/val_coco.json',
    backend_args=None,
    format_only=False,
    metric='bbox',
    type='CocoMetric')
vis_backends = [
    dict(_scope_='mmdet', type='LocalVisBackend'),
]
visualizer = dict(
    _scope_='mmdet',
    name='visualizer',
    type='DetLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
        dict(type='WandbVisBackend'),
    ])
work_dir = '../work_dirs/cascade_2ep'

tta_model = dict(
    type='DetTTAModel',
    tta_cfg=dict(nms=dict(
                   type='nms',
                   iou_threshold=0.5),
                   max_per_img=300))

img_scales = [(1333, 800), (666, 400), (2000, 1200)]
tta_pipeline = [
    dict(type='LoadImageFromFile',
         backend_args=None),
    dict(
        type='TestTimeAug',
        transforms=[[
            dict(type='Resize', scale=s, keep_ratio=True) for s in img_scales
        ], [
            dict(type='RandomFlip', prob=1.),
            dict(type='RandomFlip', prob=0.)
        ], [
            dict(
               type='PackDetInputs',
               meta_keys=('img_id', 'img_path', 'ori_shape',
                       'img_shape', 'scale_factor', 'flip',
                       'flip_direction'))
       ]])]

# model = ConfigDict(tta_model, module=model)
# test_dataloader.dataset.pipeline = tta_pipeline