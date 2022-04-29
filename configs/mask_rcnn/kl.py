_base_ = [
    'detpro_mask_rcnn_r50_fpn_20e_lvis_v1.py',
]

model = dict(roi_head=dict(bbox_head=dict(loss_cls=dict(
    _delete_=True, type='KLDivLossZSL', T=1, loss_weight=1000,
))))
