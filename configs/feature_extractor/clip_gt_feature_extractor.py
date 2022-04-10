_base_ = [
    '../_base_/schedules/schedule_1x.py',
    '../_base_/default_runtime.py',
]

model = dict(
    type='CLIPGTFeatureExtractor',
    lmdb_filepath='tmp.lmdb',
)
img_norm_cfg = dict(
    mean=[v * 255 for v in (0.48145466, 0.4578275, 0.40821073)], 
    std=[v * 255 for v in (0.26862954, 0.26130258, 0.27577711)], 
    to_rgb=True,
)
pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'], meta_keys=('img_info',)),
]
dataset_type = 'CocoFeatureExtractionDataset'
data_root = 'data/coco/'
data = dict(
    samples_per_gpu=8,
    workers_per_gpu=6,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_train2017.json',
        img_prefix=data_root + 'train2017/',
        pipeline=pipeline,
    ),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_val2017.json',
        img_prefix=data_root + 'val2017/',
        pipeline=pipeline,
    ),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_val2017.json',
        img_prefix=data_root + 'val2017/',
        pipeline=pipeline,
    ),
)
runner = dict(max_epochs=1)
custom_hooks = [
    dict(type='SaveLmdbToOSS', priority='LOWEST', lmdb_filepath='data/coco/embeddings5.lmdb'),
]
