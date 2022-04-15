_base_ = [
    '../_base_/schedules/schedule_1x.py',
    '../_base_/default_runtime.py',
]

model = dict(
    type='PromptTrainer',
    distiller=dict(
        teacher_cfg=dict(
            pretrained='pretrained/clip/RN50.pt',
            context_length=13,  # including sot(1), prompt(8), text(3), and eot(1)
            prompt_length=8,
            text_only=True,
        ),
    ),
    backbone=dict(),  # compat tools/test.py
)
dataset = dict(
    type='PromptDataset',
    data_root='data/coco/proposal_embeddings7.pth',
)
data = dict(
    samples_per_gpu=1024,
    workers_per_gpu=48,
    train=dataset,
    val=dataset,
    test=dataset,
)
runner = dict(max_epochs=12)
evaluation = dict(gpu_collect=True)
