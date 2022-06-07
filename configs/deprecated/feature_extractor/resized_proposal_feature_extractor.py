_base_ = [
    'proposal_feature_extractor.py',
]

pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadProposals', num_max_proposals=None),
    dict(type='LoadImageFromRegions', n_px=224, transforms=[
        dict(
            type='Resize',
            img_scale=[(1333, 640), (1333, 800)],
            multiscale_mode='range',
            keep_ratio=True),
        dict(type='RandomFlip', flip_ratio=0.5),
    ]),
    dict(type='Collect', keys=['img', 'bboxes'], meta_keys=('id',)),
]
data = dict(
    train=dict(pipeline=pipeline),
    val=dict(pipeline=pipeline),
    test=dict(pipeline=pipeline),
)
