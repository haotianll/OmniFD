data_root = '../data/forgery/'

ann_files = [
    'ForgeryNet/annotations/unified/image_train.json',
    'ForgeryNet/annotations/unified/image_val.json',
]

data_prefixes = [
    dict(img_path='ForgeryNet/Training/image', seg_map_path='ForgeryNet/Training/spatial_localize'),
    dict(img_path='ForgeryNet/Validation/image', seg_map_path='ForgeryNet/Validation/spatial_localize'),
]

batch_size = 128

data_preprocessor = dict(
    type='UnifiedDataPreprocessor',
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True, format_shape='NCTHW',
)

albu_transform_list = [
    dict(type='GaussNoise', p=1.0, var_limit=(10.0, 50.0), per_channel=True, mean=0),
    dict(type='Sharpen', p=1.0, alpha=(0.2, 0.5), lightness=(0.5, 1.0)),
    dict(type='RandomBrightnessContrast',
         brightness_limit=(-0.5, 0.5), contrast_limit=(-0.5, 0.5), brightness_by_max=True, p=1.0),
    dict(type='ImageCompression', quality_lower=1, quality_upper=99, p=1.0),
    dict(type='GaussianBlur', blur_limit=(3, 11), p=1.0),
    dict(type='CLAHE', clip_limit=(1, 8), tile_grid_size=(8, 8), p=1.0),
    dict(type='RandomGamma', gamma_limit=(10, 150), eps=None, p=1.0),
    dict(type='ToGray', p=1.0),
    dict(type='ChannelShuffle', p=1.0),
]

albu_transforms = [
    dict(
        type='OneOf',
        transforms=[
            dict(
                type='OneOf',
                transforms=[
                    dict(type='SomeOf', transforms=albu_transform_list, n=2, p=1.0),
                    dict(type='SomeOf', transforms=albu_transform_list, n=3, p=1.0),
                    dict(type='SomeOf', transforms=albu_transform_list, n=4, p=1.0),
                ], p=0.98
            ),
            dict(type='OneOf', transforms=albu_transform_list, p=0.01)
        ], p=0.99
    )
]

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='FilterKeys', tasks=['image']),
    dict(type='FaceCrop'),
    dict(type='RandomResize', scale=(224, 224), ratio_range=(1., 8. / 7.)),
    dict(type='RandomCrop', crop_size=224),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='Albu', transforms=albu_transforms),
    dict(type='PackImageInputs'),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='FilterKeys', tasks=['image']),
    dict(type='FaceCrop'),
    dict(type='Resize', scale=224),
    dict(type='PackImageInputs'),
]

train_dataset = dict(
    type='UnifiedImageDataset',
    data_root=data_root,
    ann_file=data_root + ann_files[0],
    data_prefix=data_prefixes[0],
    pipeline=train_pipeline,
)

test_dataset = dict(
    type='UnifiedImageDataset',
    data_root=data_root,
    ann_file=data_root + ann_files[1],
    data_prefix=data_prefixes[1],
    pipeline=test_pipeline,
)

train_dataloader = dict(
    batch_size=batch_size,
    num_workers=8,
    dataset=train_dataset,
    sampler=dict(type='DefaultSampler', shuffle=True),
    collate_fn=dict(type='pseudo_collate'),
)

test_dataloader = dict(
    batch_size=batch_size,
    num_workers=8,
    dataset=test_dataset,
    sampler=dict(type='DefaultSampler', shuffle=False),
    collate_fn=dict(type='pseudo_collate'),
)

test_evaluator = dict(
    type='UnifiedEvaluator',
    metrics=dict(
        type='UnifiedMetric',
        task_metrics={
            'image': [
                dict(type='ForgeryNetAccuracy'),
                dict(type='AUC'),
            ],
        }
    )
)

val_dataloader = test_dataloader
val_evaluator = test_evaluator
