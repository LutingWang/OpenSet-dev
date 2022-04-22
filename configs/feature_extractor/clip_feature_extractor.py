_base_ = [
    '../_base_/schedules/schedule_1x.py',
    '../_base_/default_runtime.py',
]

model = dict(
    type='CLIPFeatureExtractor',
    pretrained='pretrained/clip/ViT-B-32.pt',
    data_root='data/lvis_v1/proposal_embeddings4/',
)
img_norm_cfg = dict(
    mean=[v * 255 for v in (0.48145466, 0.4578275, 0.40821073)], 
    std=[v * 255 for v in (0.26862954, 0.26130258, 0.27577711)], 
    to_rgb=True,
)
pipeline = [
    dict(type='LoadProposals', num_max_proposals=None),
    # dict(type='LoadAnnotations', with_bbox=True),
    dict(type='LoadImageFromRegions', n_px=224),
    dict(type='Collect', keys=['img', 'bboxes'], meta_keys=('id',)),
]
dataset_type = 'LVISV1GZSLDataset'
data_root = 'data/lvis_v1/'
img_prefix = 'data/coco/'
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=6,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/lvis_v1_train.json',
        img_prefix=img_prefix,
        proposal_file=data_root + 'proposals/rpn_r101_fpn_lvis_v1_train.pkl',
        pipeline=pipeline,
    ),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/lvis_v1_val.json',
        img_prefix=img_prefix,
        proposal_file=data_root + 'proposals/rpn_r101_fpn_lvis_v1_val.pkl',
        pipeline=pipeline,
    ),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/lvis_v1_val.json',
        img_prefix=img_prefix,
        proposal_file=data_root + 'proposals/rpn_r101_fpn_lvis_v1_val.pkl',
        pipeline=pipeline,
    ),
)
runner = dict(max_epochs=1)
