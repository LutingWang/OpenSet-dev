_base_ = [
    'mask_rcnn_r50_fpn.py',
]

model = dict(
    roi_head=dict(
        type='ViLDEnsembleRoIHead',
        bbox_head=dict(
            type='ViLDTextBBoxHead',
            num_classes=1203,
            bg_class_embedding=True,
            class_embeddings='data/lvis_v1/prompt/detpro_ViT-B-32.pt'),
        ensemble_head=dict(
            type='ViLDImageBBoxHead',
            distiller=dict(
                losses=dict(
                    bbox_kd=dict(
                        tensor_names=['preds', 'targets'],
                        type='L1Loss',
                        weight=dict(
                            type='WarmupScheduler',
                            value=256,
                            iter_=200,
                        ),
                    ),
                ),
            ),
        ),
        mask_head=dict(
            num_classes=1203,
        ),
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
