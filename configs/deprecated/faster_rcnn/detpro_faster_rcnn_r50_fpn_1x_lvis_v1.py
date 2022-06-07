_base_ = [
    '../_base_/models/lvis_faster_rcnn_r50_fpn.py',
    '../_base_/datasets/lvis_v1_detection_zsl.py',
    '../_base_/schedules/schedule_1x_cos.py', 
    '../_base_/default_runtime.py',
]

model = dict(
    backbone=dict(
        frozen_stages=4,
    ),
    roi_head=dict(
        bbox_head=dict(
            class_embeddings='data/lvis_v1/prompt/detpro_ViT-B-32.pt',
            num_classes=1203,
            # loss_cls=dict(
            #     _delete_=True, type='KLDivLossZSL', T=1, loss_weight=1000,
            # ),
        ),
    ),
)
# optimizer = dict(lr=0.005, weight_decay=0.000025)
