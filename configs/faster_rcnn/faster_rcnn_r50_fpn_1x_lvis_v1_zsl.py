_base_ = [
    '../_base_/models/faster_rcnn_r50_fpn.py',
    '../_base_/datasets/lvis_v1_detection_zsl.py',
    '../_base_/schedules/schedule_20e.py', 
    '../_base_/default_runtime.py',
]

data_root = 'data/lvis_v1/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], 
    std=[58.395, 57.12, 57.375], 
    to_rgb=True,
)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    # dict(type='LoadProposals', num_max_proposals=None),
    dict(type='LoadProposalEmbeddings', data_root='data/lvis_v1/proposal_embeddings4/'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    # dict(type='Collect', keys=['img', 'proposals', 'gt_bboxes', 'gt_labels', 'bboxes', 'bbox_embeddings']),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'bboxes', 'bbox_embeddings']),
]
# test_pipeline = [
#     dict(type='LoadImageFromFile'),
#     dict(type='LoadProposals', num_max_proposals=None),
#     dict(
#         type='MultiScaleFlipAug',
#         img_scale=(1333, 800),
#         flip=False,
#         transforms=[
#             dict(type='Resize', keep_ratio=True),
#             dict(type='RandomFlip'),
#             dict(type='Normalize', **img_norm_cfg),
#             dict(type='Pad', size_divisor=32),
#             dict(type='ImageToTensor', keys=['img']),
#             dict(type='ToTensor', keys=['proposals']),
#             dict(
#                 type='ToDataContainer', 
#                 fields=[
#                     dict(key='proposals', stack=False),
#                 ],
#             ),
#             dict(type='Collect', keys=['img', 'proposals']),
#         ],
#     ),
# ]
data = dict(
    # samples_per_gpu=2,
    train=dict(
        dataset=dict(
            # proposal_file=data_root + 'proposals/rpn_r101_fpn_lvis_v1_train.pkl',
            pipeline=train_pipeline,
        ),
    ),
    # val=dict(
    #     proposal_file=data_root + 'proposals/rpn_r101_fpn_lvis_v1_val.pkl',
    #     pipeline=test_pipeline,
    # ),
    # test=dict(
    #     proposal_file=data_root + 'proposals/rpn_r101_fpn_lvis_v1_val.pkl',
    #     pipeline=test_pipeline,
    # ),
)
model = dict(
    roi_head=dict(
        type='ViLDEnsembleRoIHead',
        bbox_head=dict(
            type='ViLDTextBBoxHead',
            # class_embeddings=data_root + 'prompt/vild_ViT-B-32.pt',
            class_embeddings=data_root + 'prompt/detpro_vild_ViT-B-32.pt',
            bg_class_embedding=True,
            num_classes=1203,
        ),
        ensemble_head=dict(
            type='ViLDImageBBoxHead',
            distiller=dict(
                # student_hooks=dict(
                #     preds=dict(type='StandardHook', path='fc_cls[0]'),
                # ),
                losses=dict(image=dict(
                    type='L1Loss',
                    tensor_names=['preds', 'targets'],
                    weight=256,
                )),
                schedulers=[dict(
                    type='WarmupScheduler',
                    tensor_names=['loss_image'],
                    iter_=200,
                )],
            ),
        ),
    ),
    test_cfg=dict(
        rcnn=dict(
            score_thr=0.0001,
            max_per_img=300,
        ),
    ),
    init_cfg=dict(type='Pretrained', checkpoint='data/ckpts/soco_star_mask_rcnn_r50_fpn_400e.pth'),
)
# load_from = 'data/ckpts/soco_mask_rcnn_r50_fpn_star_400e.pth'
optimizer = dict(lr=0.005, weight_decay=0.000025)
evaluation = dict(interval=1)
custom_hooks = []
