_base_ = [
    'faster_rcnn_r50_fpn.py',
]

model = dict(
    roi_head=dict(
        _delete_=True,
        type='ViLDEnsembleRoIHead',
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        text_bbox_head=dict(
            type='ViLDTextBBoxHead',
            num_classes=1203,
            norm_cfg=dict(type='SyncBN', requires_grad=True),
            reg_class_agnostic=True,
            loss_bbox=dict(type='L1Loss', loss_weight=1.0),
            class_embeddings='data/lvis_v1/prompt/detpro_ViT-B-32.pt'),
        image_bbox_head=dict(
            type='ViLDImageBBoxHead',
            num_classes=1203,
            with_reg=False,
            norm_cfg=dict(type='SyncBN', requires_grad=True),
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
