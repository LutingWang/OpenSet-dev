_base_ = [
    'detpro_faster_rcnn_r50_glipplv_1x_lvis_v1.py',
]

model = dict(
    glip_neck=dict(refine=dict(
        _delete_=True,
        type='CascadeFusionDyHead',
        num_layers=3,
        mil_classifiers=[
            dict(
                type='GAPClassifier',
                kappa=64, 
                logits_weight=True,
                tau=0.07,
                loss_mil=dict(
                    type='FocalWithLogitsLoss',
                    weight=8,
                ),
                loss_image_kd=dict(
                    type='L1Loss',
                    weight=64,
                ),
            ),
            dict(
                type='GAPClassifier',
                kappa=32, 
                logits_weight=True,
                tau=0.07,
                loss_mil=dict(
                    type='FocalWithLogitsLoss',
                    weight=4,
                ),
                loss_image_kd=dict(
                    type='L1Loss',
                    weight=64,
                ),
            ),
            dict(
                type='GAPClassifier',
                kappa=16, 
                logits_weight=True,
                tau=0.07,
                loss_mil=dict(
                    type='FocalWithLogitsLoss',
                    weight=2,
                ),
                loss_image_kd=dict(
                    type='L1Loss',
                    weight=64,
                ),
            ),
        ],
    )),
    plv_refine=dict(
        mil_classifier=dict(
            kappa=128,
            loss_mil=dict(
                weight=16,
            ),
            loss_image_kd=dict(
                weight=64,
            ),
        ),
    ),
)
