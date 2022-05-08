_base_ = [
    'detpro_faster_rcnn_r50_glip_1x_lvis_v1.py',
]

model = dict(glip_neck=dict(refine=dict(
    type='CascadeFusionDyHead',
    mil_classifier=[
        dict(
            type='DyHeadClassifier',
            kappa=100, 
            logits_weight=False,
        ),
        dict(
            type='DyHeadClassifier',
            kappa=50, 
            logits_weight=False,
        ),
        dict(
            type='DyHeadClassifier',
            kappa=20, 
            logits_weight=False,
        ),
    ]
)))
