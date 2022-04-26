_base_ = [
    '../_base_/schedules/schedule_1x.py',
    '../_base_/default_runtime.py',
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
    ),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/lvis_v1_val.json',
        img_prefix=img_prefix,
    ),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/lvis_v1_val.json',
        img_prefix=img_prefix,
    ),
)
model = dict(
    type='CLIPFeatureExtractor',
)
runner = dict(max_epochs=1)
