_base_ = [
    'detpro_mask_rcnn_r50_fpn_20e_lvis_v1.py',
]

model = dict(backbone=dict(
    frozen_stages=4,
))
