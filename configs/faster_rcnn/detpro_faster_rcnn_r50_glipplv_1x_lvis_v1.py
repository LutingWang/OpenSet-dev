_base_ = [
    'detpro_faster_rcnn_r50_glip_1x_lvis_v1.py',
]

model = dict(
    type='GLIPPLVNeckFasterRCNN',
    glip_neck=dict(
        refine=dict(
            _delete_=True,
            type='BaseFusionDyHead',
            num_layers=3, 
        ),
    ),
    plv_refine=dict(
        hidden_dim=512,
        mil_classifier=dict(
            type='DyHeadClassifier',
            kappa=35, 
            logits_weight=True,
            tau=0.07,
            loss_mil=dict(
                type='FocalWithLogitsLoss',
                weight=32,
            ),
            loss_image_kd=dict(
                type='L1Loss',
                weight=256,
            ),
        ),
    ),
)
