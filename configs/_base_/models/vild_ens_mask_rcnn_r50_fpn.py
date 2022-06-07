_base_ = [
    'mask_rcnn_r50_fpn.py',
]

model = dict(
    roi_head=dict(
        type='ViLDRoIHead',
        bbox_head=dict(
            _delete_=True,
            type='ViLDTextBBoxHead',
            num_classes=1203,
            norm_cfg=dict(type='SyncBN', requires_grad=True),
            reg_class_agnostic=True,
            loss_bbox=dict(type='L1Loss', loss_weight=1.0),
            class_embeddings='data/lvis_v1/prompt/detpro_ViT-B-32.pt'),
        distiller=dict(),
        init_cfg=[
            dict(type='Xavier', layer='Linear'),
            dict(type='Normal', layer='Conv2d', std=0.01),
        ]
    ),
    test_cfg=dict(
        rcnn=dict(
            score_thr=0.0001,
            max_per_img=300,
        ),
    ),
    init_cfg=dict(type='Pretrained', checkpoint='data/ckpts/soco_star_mask_rcnn_r50_fpn_400e.pth'),
)
