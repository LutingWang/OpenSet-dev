_base_ = [
    '../_base_/models/lvis_mask_rcnn_r50_fpn.py',
    '../_base_/datasets/lvis_v1_instance_zsl.py',
    '../_base_/schedules/schedule_20e.py', 
    '../_base_/default_runtime.py',
]

model = dict(
    backbone=dict(
        # frozen_stages=-1,
        norm_cfg=dict(type='SyncBN'),
    ),
    neck=dict(
        norm_cfg=dict(type='SyncBN'),
    ),
    roi_head=dict(
        bbox_head=dict(
            norm_cfg=dict(type='SyncBN'),
            class_embeddings='data/lvis_v1/prompt/detpro_ViT-B-32.pt',
            num_classes=1203,
            # loss_cls=dict(
            #     _delete_=True, type='KLDivLossZSL', T=1, loss_weight=1000,
            # ),
        ),
        mask_head=dict(num_classes=1203),
    ),
)
# optimizer = dict(lr=0.005, weight_decay=0.000025)
