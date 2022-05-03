_base_ = [
    '../_base_/datasets/lvis_v1_detection_zsl.py',
    '../_base_/schedules/schedule_1x.py',
    '../_base_/default_runtime.py',
]

dataset_type='LVISV1PromptDataset'
embeddings_root = 'data/lvis_v1/proposal_embeddings7/'
train_pipeline = [
    dict(type='LoadPthEmbeddings', data_root=embeddings_root),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='DefaultFormatBundle'),
    dict(type='ToTensor', keys=['bboxes', 'bbox_embeddings']),
    dict(
        type='ToDataContainer', 
        fields=[dict(key='bboxes'), dict(key='bbox_embeddings')]),
    dict(type='Collect', keys=['gt_bboxes', 'gt_labels', 'bboxes', 'bbox_embeddings'], meta_keys=tuple()),
]
test_pipeline = [
    dict(type='LoadPthEmbeddings', data_root=embeddings_root),
    dict(type='ToTensor', keys=['bboxes', 'bbox_embeddings']),
    dict(
        type='ToDataContainer', 
        fields=[dict(key='bboxes'), dict(key='bbox_embeddings')]),
    dict(type='Collect', keys=['bboxes', 'bbox_embeddings'], meta_keys=tuple()),
]
data = dict(
    samples_per_gpu=1024,
    workers_per_gpu=48,
    train=dict(dataset=dict(
        type=dataset_type,
        pipeline=train_pipeline)),
    val=dict(
        type=dataset_type,
        test_mode=False,
        image2dc=True,
        pipeline=train_pipeline),
    test=dict(
        type=dataset_type,
        test_mode=False,
        image2dc=True,
        pipeline=train_pipeline))
model = dict(
    type='PromptTrainer',
    backbone=dict(),
    distiller=dict(teacher_cfg=dict(
        pretrained='pretrained/clip/ViT-B-32.pt',
        context_length=23,  # sot(1), prompt(8), text(13), and eot(1)
        prompt_length=8,
        text_only=True,
    )),
    num_classes=1203,
    test_cfg=dict(
        score_thr=0.0001,
        nms=dict(type='nms', iou_threshold=0.5),
        max_per_img=300,
    ),
    init_cfg=[],
)
runner = dict(max_epochs=12)
evaluation = dict(interval=1)
