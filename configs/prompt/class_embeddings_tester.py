_base_ = [
    '../_base_/default_runtime.py',
]

pipeline = [
    dict(type='LoadPthEmbeddings', data_root='data/lvis_v1/gt_embeddings5/', task_name='val'),
    # dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['bboxes', 'bbox_embeddings'], meta_keys=tuple()),
]
dataset = dict(
    type='LVISV1GZSLDataset',
    ann_file='data/lvis_v1/annotations/lvis_v1_val.json',
    img_prefix='data/coco/',
    pipeline=pipeline,
)
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(),
    val=dataset,
    test=dataset)
model = dict(
    type='ClassEmbeddingsTester',
    class_embeddings='data/lvis_v1/prompt/detpro_vild_ViT-B-32.pt',
    backbone=dict(),
    test_cfg=dict(
        score_thr=0.0001,
        nms=dict(type='nms', iou_threshold=0.5),
        max_per_img=300,
    ),
)
