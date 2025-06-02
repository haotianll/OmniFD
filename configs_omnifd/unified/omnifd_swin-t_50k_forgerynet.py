_base_ = [
    '../_base_/datasets/forgerynet_all.py',
    '../_base_/schedules/default_adamw_50k_lr1e-4.py',
    '../_base_/default_runtime.py',
]

train_tasks = ['video', 'image', 'temporal', 'spatial']
test_tasks = ['video']

model = dict(
    type='UnifiedDetector',
    backbone=dict(
        type='SwinTransformerOmni',
        checkpoint_name='omnivore_swinT',
        patch_size=(2, 4, 4),
        embed_dim=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=(8, 7, 7),
        drop_path_rate=0.2,
        with_cp=False,
    ),
    neck=dict(
        type='Interaction',
        latent_channels=512,
        num_latents=64,
        feat_channels=[192, 384, 768, 768],
    ),
    head=dict(
        type='UnifiedHead',
        decoder_dict={
            'image': dict(
                type='ImageClsHead',
                in_channels=512,
                num_classes=2,
                loss_module=dict(type='CrossEntropyLoss', loss_weight=1.0),
            ),
            'spatial': dict(
                type='SpatialHead',
                in_channels=[192, 384, 768, 768],
                channels=256,
                latent_channels=512,
                num_latents=64,
                num_classes=1,
                loss_module=dict(type='BCESegLoss', loss_weight=1.0),
            ),
            'video': dict(
                type='VideoClsHead',
                in_channels=512,
                num_classes=2,
                loss_module=dict(type='CrossEntropyLoss', loss_weight=0.25),
            ),
            'temporal': dict(
                type='TemporalHead',
                window_size=32,
                proj=dict(type='TemporalProj', in_channels=768, out_channels=512),
                neck=dict(type='TemporalFPN', in_channels=512, out_channels=512),
                rpn_head=dict(
                    type='TemporalRPNHead',
                    in_channels=512,
                    feat_channels=512,
                    latent_channels=512,
                    num_classes=1,
                    loss_weight=0.25,
                    loss=dict(
                        cls_loss=dict(type='TadFocalLoss'),
                        reg_loss=dict(type='DIOULoss'),
                    ),
                ),
            ),
        },
        train_tasks=train_tasks,
        test_tasks=test_tasks,
    ),
)
val_evaluator = dict(metrics=dict(test_tasks=test_tasks))
test_evaluator = dict(metrics=dict(test_tasks=test_tasks))
