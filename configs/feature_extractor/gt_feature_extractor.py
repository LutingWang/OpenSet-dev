_base_ = [
    'feature_extractor.py',
]

pipeline = [
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='LoadImageFromRegions', n_px=224),
    dict(type='Collect', keys=['img', 'bboxes'], meta_keys=('id',)),
]
data = dict(
    train=dict(pipeline=pipeline),
    val=dict(test_mode=False, pipeline=pipeline),
    test=dict(test_mode=False, pipeline=pipeline),
)
model = dict(
    data_root='data/lvis_v1/gt_embeddings6/',
)
