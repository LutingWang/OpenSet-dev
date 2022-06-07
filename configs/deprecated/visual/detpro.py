_base_ = [
    '../mask_rcnn/detpro_mask_rcnn_r50_fpn_20e_lvis_v1.py',
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='Resize',
        img_scale=[(1333, 640), (1333, 800)],
        multiscale_mode='range',
        keep_ratio=True),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='Pad', size_divisor=32),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=['gt_bboxes', 'gt_labels']),
    dict(type='ToDataContainer', fields=(dict(key='img'), dict(key='gt_bboxes'), dict(key='gt_labels'))),
    dict(
        type='Collect',
        keys=['img', 'gt_bboxes', 'gt_labels'],
        meta_keys=('img_shape', 'img_norm_cfg')),
        # meta_keys=tuple()),
]
data = dict(
    val=dict(
        force_train=True,
        pipeline=test_pipeline,
    ),
    test=dict(
        force_train=True,
        pipeline=test_pipeline,
    ),
)
model = dict(
    type='Visualizer',
    distiller=dict(
        student_hooks=dict(
            x=dict(
                type='StandardHook',
                path='neck',
            ),
        ),
        visuals=dict(
            activations=dict(
                type='ActivationVisual',
                multilevel=True,
                tensor_names=['imgs', 'x'],
            ),
            annotations=dict(
                type='AnnotationVisual',
                tensor_names=['img', 'bboxes', 'labels', 'classes'],
            ),
            act_saver=dict(
                type='CV2Saver',
                multilevel=True,
                tensor_names=['activations'],
                root_dir='work_dirs/visuals1',
                trial_name='act',
                suffix='act',
            ),
            ann_saver=dict(
                type='CV2Saver',
                tensor_names=['annotations'],
                root_dir='work_dirs/visuals1',
                trial_name='ann',
                suffix='ann',
            ),
        ),
    ),
)
