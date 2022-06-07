_base_ = [
    '../_base_/models/retinanet_r50_fpn.py',
    '../_base_/datasets/coco_detection_zsl.py',
    '../_base_/schedules/schedule_1x.py',
    '../_base_/default_runtime.py',
]

model = dict(
    bbox_head=dict(
        type='RetinaHeadZSL',
        class_embeddings='data/prompt/CocoDataset.pth',
        num_classes=48,
        anchor_generator=dict(
            octave_base_scale=8,
            scales_per_octave=1,
            ratios=[1.0],
        ),
    ),
)
optimizer_config = dict(
    _delete_=True,
    grad_clip=dict(max_norm=20),
)
