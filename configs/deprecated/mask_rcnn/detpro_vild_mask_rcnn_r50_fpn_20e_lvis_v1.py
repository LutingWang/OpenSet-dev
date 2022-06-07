_base_ = [
    'detpro_mask_rcnn_r50_fpn_20e_lvis_v1.py',
]

model = dict(roi_head=dict(bbox_head=dict(
    class_embeddings='data/lvis_v1/prompt/detpro_vild_ViT-B-32.pt')))
