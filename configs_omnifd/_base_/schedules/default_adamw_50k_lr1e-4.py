# optimizer
optim_wrapper = dict(
    type='AmpOptimWrapper',
    optimizer=dict(
        type='AdamW',
        lr=1e-4,
        weight_decay=1e-4,
        eps=1e-8,
        betas=(0.9, 0.999)
    ),
    paramwise_cfg=dict(
        norm_decay_mult=0.0,
        bias_decay_mult=0.0,
        flat_decay_mult=0.0,
        custom_keys={
            '.pos_embed': dict(decay_mult=0.0),
            '.pos_embed_2d': dict(decay_mult=0.0),
            '.absolute_pos_embed': dict(decay_mult=0.0),
            '.relative_position_bias_table': dict(decay_mult=0.0)
        },
        bypass_duplicate=True,
    ),
    clip_grad=dict(max_norm=5.0),
)

# learning policy
param_scheduler = [
    # warm up learning rate scheduler
    dict(type='LinearLR', start_factor=1e-3, by_epoch=False, end=2000),
    # main learning rate scheduler
    dict(type='CosineAnnealingLR', T_max=45000, eta_min=1e-6, by_epoch=False, begin=2000, end=50000),
]

# train, val, test setting
train_cfg = dict(by_epoch=False, max_iters=50000, val_interval=50000)
val_cfg = dict()
test_cfg = dict()

# NOTE: `auto_scale_lr` is for automatically scaling LR, based on the actual training batch size.
# auto_scale_lr = dict(base_batch_size=16)
