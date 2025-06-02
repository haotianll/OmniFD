_base_ = [
    '../_base_/datasets/forgerynet_temporal.py',
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
            'temporal': dict(
                type='TemporalHead',
                window_size=32,
                window_size_test=64,
                proj=dict(type='TemporalProj', in_channels=768, out_channels=512),
                neck=dict(type='TemporalFPN', in_channels=512, out_channels=512),
                rpn_head=dict(
                    type='TemporalRPNHead',
                    in_channels=512,
                    feat_channels=512,
                    latent_channels=512,
                    num_classes=1,
                    loss_weight=1.0,
                    loss=dict(
                        cls_loss=dict(type='TadFocalLoss'),
                        reg_loss=dict(type='DIOULoss'),
                    ),
                ),
            ),
        },
        train_tasks=['temporal'],
        test_tasks=['temporal'],
    ),
)
