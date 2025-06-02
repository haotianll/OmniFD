data_root = './data/'

data_preprocessor = dict(
    type='UnifiedDataPreprocessor',
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True, format_shape='NCTHW'
)

batch_size = 4

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

albu_transforms_video = [
    dict(
        type='Compose',
        transforms=albu_transforms,
        additional_targets={f'image{i}': 'image' for i in range(0, 512)}
    )
]

train_pipeline = [
    dict(type='VideoDecordInitDir'),
    dict(type='VideoSampleFrames', clip_len=32, frame_interval=1, num_clips=1),
    dict(type='LoadFramesFromDir'),
    dict(type='VideoRandomResize', scale=(224, 224), ratio_range=(1., 8. / 7.)),
    dict(type='VideoRandomCrop', crop_size=224),
    dict(type='VideoRandomFlip', prob=0.5, direction='horizontal'),
    dict(type='VideoAlbu', transforms=albu_transforms_video),
    dict(type='VideoFormatShape', input_format='NCTHW', collapse=True),
    dict(type='PackVideoInputs'),
]

test_pipeline = [
    dict(type='VideoDecordInitDir'),
    dict(type='VideoSampleFrames', clip_len=32, frame_interval=1, num_clips=1, test_mode=True),
    dict(type='LoadFramesFromDir'),
    dict(type='VideoResize', scale=224),
    dict(type='VideoFormatShape', input_format='NCTHW', collapse=True),
    dict(type='PackVideoInputs'),
]

train_dataset = dict(
    type='UnifiedVideoDataset',
    data_root=data_root,
    ann_file='FaceForensics++/annotations/train_video.json',
    data_prefix=dict(video_path=''),
    pipeline=train_pipeline,
)

test_dataset = dict(
    type='UnifiedVideoDataset',
    data_root=data_root,
    ann_file='FaceForensics++/annotations/test_video.json',
    data_prefix=dict(video_path=''),
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
            'video': [
                dict(type='Accuracy', topk=(1,)),
                dict(type='AUC'),
            ],
        }
    )
)

val_dataloader = test_dataloader
val_evaluator = test_evaluator
