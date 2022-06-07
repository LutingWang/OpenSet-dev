_base_ = [
    '../_base_/models/rpn_r50_fpn.py', '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]
model = dict(backbone=dict(depth=101, init_cfg=None))
data = dict(
    test=dict(
        type='CocoZSLDataset',
    )
)
