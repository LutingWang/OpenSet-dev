_base_ = [
    'detpro_faster_rcnn_r50_fpn_1x_lvis_v1.py',
]

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadRawImageFromFile', n_px=224),
    # dict(type='LoadPthEmbeddings', data_root='data/lvis_v1/proposal_embeddings4/'),
    # dict(type='LoadZipEmbeddings', data_root='data/lvis_v1/proposal_embeddings.zip/data/lvis_clip_image_embedding/', task_name='train2017'),
    dict(type='LoadPthEmbeddings', data_root='data/lvis_v1/proposal_embeddings10/', min_bbox_area=32*32, detpro=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='Resize',
        img_scale=[(1333, 480), (1333, 800)],
        multiscale_mode='range',
        keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='ToTensor', keys=['bboxes', 'bbox_embeddings']),
    dict(
        type='ToDataContainer', 
        fields=[dict(key='bboxes'), dict(key='bbox_embeddings')]),
    dict(type='Collect', keys=['img', 'raw_image', 'gt_bboxes', 'gt_labels', 'bboxes', 'bbox_embeddings']),
]
data = dict(train=dict(dataset=dict(pipeline=train_pipeline)))
model = dict(
    type='GLIPFasterRCNN',
    glip_neck= dict(
        type='GLIP',
        in_channels=256,
        num_levels=5,
        refine_level=2,
    ),
    distiller=dict(
        teacher_cfg=dict(
            pretrained='pretrained/clip/ViT-B-32.pt',
            image_only=True,
            with_preprocess=False,
            vpe_hook=False),
        student_hooks=dict(image_features=dict(
            type='StandardHook', path='_glip_neck.refine._adapter')),
        losses=dict(image_kd=dict(
            type='L1Loss', 
            tensor_names=['image_features', 'clip_image_features'], 
            weight=256.0,
            norm=True)),
        schedulers=[dict(
            type='WarmupScheduler',
            tensor_names=['loss_image_kd'],
            iter_=200)],
    )
)
