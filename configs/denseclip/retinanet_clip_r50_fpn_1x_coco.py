_base_ = [
    '../_base_/models/retinanet_r50_fpn.py',
    '../_base_/datasets/coco_detection_clip.py',
    '../_base_/schedules/schedule_1x.py',
    '../_base_/default_runtime.py'
]

model = dict(
    backbone=dict(
        type='CLIPResNet',
        layers=[3, 4, 6, 3],
        output_dim=1024,
        input_resolution=1344,
        style='pytorch',
        pretrained='pretrained/RN50.pt',
        init_cfg=dict(type='Pretrained', checkpoint='pretrained/RN50.pt')))

# optimizer
optimizer = dict(
    _delete_=True,
    type='AdamW', lr=0.0001, weight_decay=0.0001,
    paramwise_cfg=dict(custom_keys={
        'backbone': dict(lr_mult=0.1),
        'text_encoder': dict(lr_mult=0.0),
        'norm': dict(decay_mult=0.),
    }),
)
optimizer_config = dict(
    _delete_=True,
    grad_clip=dict(max_norm=0.1, norm_type=2),
)
