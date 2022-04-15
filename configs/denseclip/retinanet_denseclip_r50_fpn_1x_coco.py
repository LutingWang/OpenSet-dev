_base_ = [
    '../_base_/models/retinanet_r50_fpn.py',
    '../_base_/datasets/coco_detection_clip.py',
    '../_base_/schedules/schedule_1x.py',
    '../_base_/default_runtime.py',
]

model = dict(
    type='DenseCLIP_RetinaNet',
    context_length=5,
    clip_head=False,
    seg_loss=True,
    backbone=dict(
        type='CLIPResNetWithAttention',
        layers=[3, 4, 6, 3],
        output_dim=1024,
        input_resolution=1344,
        style='pytorch',
        pretrained='pretrained/clip/RN50.pt',
        init_cfg=dict(type='Pretrained', checkpoint='pretrained/clip/RN50.pt')),
    text_encoder=dict(
        type='CLIPTextContextEncoder',
        context_length=13,
        embed_dim=1024,
        transformer_width=512,
        transformer_heads=8,
        transformer_layers=12,
        pretrained='pretrained/clip/RN50.pt',
        style='pytorch'),
    context_decoder=dict(
        type='ContextDecoder',
        transformer_width=256,
        transformer_heads=4,
        transformer_layers=3,
        visual_dim=1024,
        dropout=0.1,
        outdim=1024,
        style='pytorch'),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048 + 80],
        out_channels=256,
        start_level=1,
        add_extra_convs='on_input',
        num_outs=5))
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
