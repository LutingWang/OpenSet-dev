_base_ = [
    '../_base_/datasets/coco_detection_zsl.py',
    '../_base_/models/vild_ens_faster_rcnn_r50_fpn.py',
    '../_base_/schedules/schedule_1x.py',
    '../_base_/default_runtime.py',
]

model = dict(
    roi_head=dict(
        bbox_head=dict(
            num_classes=65,
            class_embeddings='data/coco/prompt/vild_ViT-B-32.pt',
        ),
    ),
)
optimizer = dict(weight_decay=2.5e-5)
optimizer_config = dict(grad_clip=None)