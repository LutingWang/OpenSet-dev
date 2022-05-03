_base_ = [
    'faster_rcnn_r50_fpn.py',
]

model = dict(
    roi_head=dict(
        type='ViLDEnsembleRoIHead',
        bbox_head=dict(
            type='ViLDTextBBoxHead',
            bg_class_embedding=True,
        ),
        ensemble_head=dict(
            type='ViLDImageBBoxHead',
            distiller=dict(
                losses=dict(bbox_kd=dict(
                    type='L1Loss',
                    tensor_names=['preds', 'targets'],
                    weight=256,
                )),
                schedulers=[dict(
                    type='WarmupScheduler',
                    tensor_names=['loss_bbox_kd'],
                    iter_=200,
                )],
            ),
        ),
    ),
    test_cfg=dict(
        rcnn=dict(
            score_thr=0.0001,
            max_per_img=300,
        ),
    ),
    init_cfg=dict(type='Pretrained', checkpoint='data/ckpts/soco_star_mask_rcnn_r50_fpn_400e.pth'),
)
