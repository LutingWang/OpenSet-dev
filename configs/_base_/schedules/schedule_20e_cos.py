# optimizer
optimizer = dict(type='SGD', lr=0.03, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=750,
    warmup_ratio=0.001,
    min_lr=3e-4)
runner = dict(type='EpochBasedRunner', max_epochs=20)
