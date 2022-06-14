_base_ = [
    'lvis.py',
]

data_root = 'data/lvis_v1/'
pipeline = [
    dict(type='LoadProposals', num_max_proposals=None),
    dict(type='LoadImageFromRegions', n_px=224),
    dict(type='Collect', keys=['img', 'bboxes'], meta_keys=('id',)),
]
data = dict(
    train=dict(
        pipeline=pipeline,
        proposal_file=data_root + 'proposals/rpn_r101_fpn_lvis_v1_train.pkl',
    ),
    val=dict(
        pipeline=pipeline,
        proposal_file=data_root + 'proposals/rpn_r101_fpn_lvis_v1_val.pkl',
    ),
    test=dict(
        pipeline=pipeline,
        proposal_file=data_root + 'proposals/rpn_r101_fpn_lvis_v1_val.pkl',
    ),
)
model = dict(
    data_root='data/lvis_v1/proposal_embeddings7/',
)
