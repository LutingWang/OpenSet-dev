_base_ = [
    'lvis.py',
]

pipeline = [
    dict(type='LoadImageFromCLIP', n_px=224),
    dict(type='Collect', keys=['img'], meta_keys=('id',)),
]
data = dict(
    samples_per_gpu=128,
    workers_per_gpu=64,
    train=dict(pipeline=pipeline),
)
model = dict(
    type='CLIPImageFeatureExtractor',
    data_root='data/lvis_v1/image_embeddings/',
)