_base_ = [
    '../_base_/datasets/forgerynet_image.py',
    '../_base_/schedules/default_adamw_50k_lr1e-4.py',
    '../_base_/default_runtime.py',
]

model = dict(
    type='UnifiedDetector',
    backbone=dict(
        type='SwinTransformerOmni',
        checkpoint_name='omnivore_swinS',
        patch_size=(2, 4, 4),
        embed_dim=96,
        depths=[2, 2, 18, 2],
        num_heads=[3, 6, 12, 24],
        window_size=(8, 7, 7),
        drop_path_rate=0.3,
        with_cp=True,
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
        },
        train_tasks=['image'],
        test_tasks=['image'],
    ),
)
