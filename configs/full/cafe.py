model = dict(
    type='CAFE',
    freeze_neck=False,
    freeze_head=False,
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='caffe',
        init_cfg=[]),
    neck=dict(
        type='CAFENeck',
        in_channels=[256, 512, 1024, 2048],
        plv_channels=512,
        out_channels=256,
        num_outs=5,
        mil_classifier=dict(
            type='DyHeadClassifier',
            kappa=35,
            logits_weight=True,
            tau=0.07,
            loss_mil=dict(type='FocalWithLogitsLoss', weight=32),
            loss_image_kd=dict(type='L1Loss', weight=256)),
        glip_refine_level=2,
        glip_refine_layers=3,
        norm_cfg=dict(type='SyncBN', requires_grad=True)),
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
            target_means=[0.0, 0.0, 0.0, 0.0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', loss_weight=1.0)),
    roi_head=dict(
        type='ViLDEnsembleRoIHead',
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        bbox_head=dict(
            type='ViLDTextBBoxHead',
            in_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=1203,
            norm_cfg=dict(type='SyncBN', requires_grad=True),
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0.0, 0.0, 0.0, 0.0],
                target_stds=[0.1, 0.1, 0.2, 0.2]),
            reg_class_agnostic=True,
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            loss_bbox=dict(type='L1Loss', loss_weight=1.0),
            bg_class_embedding=True,
            class_embeddings='data/lvis_v1/prompt/detpro_ViT-B-32.pt'),
        mask_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=14, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        mask_head=dict(
            type='FCNMaskHead',
            num_convs=4,
            in_channels=256,
            conv_out_channels=256,
            num_classes=1203,
            class_agnostic=True,
            loss_mask=dict(
                type='CrossEntropyLoss', use_mask=True, loss_weight=1.0)),
        ensemble_head=dict(
            type='ViLDImageBBoxHead',
            distiller=dict(
                losses=dict(
                    bbox_kd=dict(
                        type='L1Loss',
                        tensor_names=['preds', 'targets'],
                        weight=256)),
                schedulers=[
                    dict(
                        type='WarmupScheduler',
                        tensor_names=['loss_bbox_kd'],
                        iter_=200)
                ]))),
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
            allowed_border=-1,
            pos_weight=-1,
            debug=False),
        rpn_proposal=dict(
            nms_pre=2000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0.0),
        rcnn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.5,
                neg_iou_thr=0.5,
                min_pos_iou=0.5,
                match_low_quality=True,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=512,
                pos_fraction=0.25,
                neg_pos_ub=-1,
                add_gt_as_proposals=True),
            pos_weight=-1,
            debug=False,
            mask_size=28)),
    test_cfg=dict(
        rpn=dict(
            nms_pre=1000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0.0),
        rcnn=dict(
            score_thr=0.0001,
            nms=dict(type='nms', iou_threshold=0.5),
            max_per_img=300,
            mask_thr_binary=0.5)),
    init_cfg=dict(
        type='Pretrained',
        checkpoint='data/ckpts/detpro_mask_rcnn_r50_fpn_20e_lvis_v1.pth'))
dataset_type = 'LVISV1Dataset'
data_root = 'data/lvis_v1/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='LoadImageEmbeddingFromFile',
        data_root='data/lvis_v1/image_embeddings1'),
    dict(
        type='LoadPthEmbeddings',
        data_root='data/lvis_v1/proposal_embeddings10/',
        min_bbox_area=1024,
        detpro=True),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(
        type='Resize',
        img_scale=[(1333, 640), (1333, 800)],
        multiscale_mode='range',
        keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='ToTensor', keys=['bboxes', 'bbox_embeddings']),
    dict(
        type='ToDataContainer',
        fields=[dict(key='bboxes'),
                dict(key='bbox_embeddings')]),
    dict(
        type='Collect',
        keys=[
            'img', 'image_embeddings', 'gt_bboxes', 'gt_labels', 'gt_masks',
            'bboxes', 'bbox_embeddings'
        ])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=10,
    train=dict(
        type='ClassBalancedDataset',
        oversample_thr=0.001,
        dataset=dict(
            type='LVISV1ZSLSeenDataset',
            ann_file='data/lvis_v1/annotations/lvis_v1_train_866_337_4.json',
            img_prefix='data/coco/',
            pipeline=[
                dict(type='LoadImageFromFile'),
                dict(
                    type='LoadImageEmbeddingFromFile',
                    data_root='data/lvis_v1/image_embeddings1'),
                dict(
                    type='LoadPthEmbeddings',
                    data_root='data/lvis_v1/proposal_embeddings10/',
                    min_bbox_area=1024,
                    detpro=True),
                dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
                dict(
                    type='Resize',
                    img_scale=[(1333, 640), (1333, 800)],
                    multiscale_mode='range',
                    keep_ratio=True),
                dict(type='RandomFlip', flip_ratio=0.5),
                dict(
                    type='Normalize',
                    mean=[123.675, 116.28, 103.53],
                    std=[58.395, 57.12, 57.375],
                    to_rgb=True),
                dict(type='Pad', size_divisor=32),
                dict(type='DefaultFormatBundle'),
                dict(type='ToTensor', keys=['bboxes', 'bbox_embeddings']),
                dict(
                    type='ToDataContainer',
                    fields=[dict(key='bboxes'),
                            dict(key='bbox_embeddings')]),
                dict(
                    type='Collect',
                    keys=[
                        'img', 'image_embeddings', 'gt_bboxes', 'gt_labels',
                        'gt_masks', 'bboxes', 'bbox_embeddings'
                    ])
            ],
            proposal_file=
            'data/lvis_v1/proposals/rpn_r101_fpn_lvis_v1_train.pkl')),
    val=dict(
        type='LVISV1GZSLDataset',
        ann_file='data/lvis_v1/annotations/lvis_v1_val.json',
        img_prefix='data/coco/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(1333, 800),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='Pad', size_divisor=32),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ]),
    test=dict(
        type='LVISV1GZSLDataset',
        ann_file='data/lvis_v1/annotations/lvis_v1_val.json',
        img_prefix='data/coco/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(1333, 800),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='Pad', size_divisor=32),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ]))
evaluation = dict(interval=4, metric='bbox', tmpdir='work_dirs/tmp')
img_prefix = 'data/coco/'
optimizer = dict(type='SGD', lr=0.002, momentum=0.9, weight_decay=0.00003)
optimizer_config = dict(grad_clip=None)
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[18, 22])
runner = dict(type='EpochBasedRunner', max_epochs=24)
checkpoint_config = dict(interval=1, create_symlink=False)
log_config = dict(interval=50, hooks=[dict(type='TextLoggerHook')])
custom_hooks = []
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
opencv_num_threads = 0
mp_start_method = 'fork'
fp16 = dict(loss_scale=dict(init_scale=512.0))
