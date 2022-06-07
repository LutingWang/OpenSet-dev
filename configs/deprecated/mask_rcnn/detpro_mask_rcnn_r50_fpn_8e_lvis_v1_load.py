_base_ = [
    'detpro_mask_rcnn_r50_fpn_20e_lvis_v1.py',
]

# optimizer = dict(lr=0.03)
lr_config = dict(step=[6])
runner = dict(max_epochs=8)
load_from = 'data/ckpts/sota_198.pth'
