_base_ = [
    '../_base_/datasets/lvis_v1_instance_zsl.py',
    '../_base_/models/vild_ens_mask_rcnn_r50_fpn.py',
    '../_base_/schedules/schedule_2x.py',
    '../_base_/default_runtime.py',
]

data = dict(
    train=dict(
        dataset=dict(
            type='LVISV1WithBugDataset',
            ann_file='data/lvis_v1/annotations/lvis_v1_train.json',
        ),
    ),
)
optimizer = dict(weight_decay=2.5e-5)
optimizer_config = dict(grad_clip=None)