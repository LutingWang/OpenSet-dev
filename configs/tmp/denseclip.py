_base_ = [
    '../_base_/models/retinanet_r50_fpn.py',
    '../_base_/datasets/coco_detection_clip_zsl.py',
    '../_base_/schedules/schedule_1x.py',
    '../_base_/default_runtime.py',
]

model = dict(
    type='DenseCLIP_RetinaNet',
    distiller=dict(
        teacher_cfg=dict(
            pretrained='pretrained/RN50.pt',
            context_length=13,  # including sot(1), prompt(8), text(3), and eot(1)
            prompt_length=8,
            input_resolution=1344,
        ),
        weight_transfer={
            'student.backbone': 'teacher.visual',
        }
    ),
    backbone=dict(
        _delete_=True,
        type='CLIPResNetWithAttention',
        layers=[3, 4, 6, 3],
        output_dim=1024,
        heads=32,
        input_resolution=1344),
    context_decoder=dict(
        in_features=1024,
        hidden_features=256,
        num_heads=4,
        num_layers=3,
        dropout=0.1),
    bbox_head=dict(
        num_classes=48,
        anchor_generator=dict(
            octave_base_scale=8,
            scales_per_octave=1,
            ratios=[1.0],
        ),
        init_cfg=dict(
            type='Normal',
            layer='Conv2d',
            std=0.01,
        ),
    ),
)
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
