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
            pretrained='pretrained/clip/RN50.pt',
            context_length=13,  # including sot(1), prompt(8), text(3), and eot(1)
            prompt_length=8,
            input_resolution=1344,
        ),
        weight_transfer={
            'student.backbone': 'teacher.visual',
        },
        student_hooks=dict(
            cls=dict(
                type='MultiCallsHook',
                path='bbox_head.retina_cls',
            ),
        ),
        adapts=dict(
            seg_mask=dict(
                type='DenseCLIPMask',
                tensor_names=['batch_input_shape', 'gt_bboxes', 'gt_labels'],
                num_classes=48,
            ),
            cls_rearranged=dict(
                type='Rearrange',
                tensor_names=['cls'],
                multilevel=True,
                pattern='n c h w -> n h w c',
            ),
            crop_features=dict(
                type='Index',
                tensor_names=['cls_rearranged', 'crop_indices'],
            ),
        ),
        losses=dict(
            seg=dict(
                type='BCEWithLogitsLoss',
                tensor_names=['seg_map', 'seg_mask'],
            ),
            image=dict(
                type='MSELoss',
                tensor_names=['image_features', 'teacher_image_features'],
                weight=1000,
            ),
            crop=dict(
                type='MSELoss',
                tensor_names=['crop_features', 'teacher_crop_features'],
                weight=1000,
            ),
        ),
        schedulers=dict(
            warmup=dict(
                type='WarmupScheduler',
                tensor_names=[
                    # 'loss_cls', 
                    'loss_seg',
                    # 'loss_image',
                    'loss_crop',
                ],
                iter_=2000,
            ),
        ),
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
        type='RetinaRPNHead',
        num_classes=48,
        anchor_generator=dict(
            type='AnchorGeneratorWithPos',
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
    # paramwise_cfg=dict(custom_keys={
    #     'backbone': dict(lr_mult=0.1),
    #     'text_encoder': dict(lr_mult=0.0),
    #     'norm': dict(decay_mult=0.),
    # }),
)
optimizer_config = dict(
    _delete_=True,
    grad_clip=dict(max_norm=20),
)
