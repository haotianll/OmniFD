import math
from numbers import Number
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from mmengine.model import BaseDataPreprocessor, stack_batch

from mmpretrain.registry import MODELS
from .unified_data_sample import UnifiedDataSample


@MODELS.register_module()
class UnifiedImageDataPreprocessor(BaseDataPreprocessor):
    """Image pre-processor for classification tasks.

    Comparing with the :class:`mmengine.model.ImgDataPreprocessor`,

    1. It won't do normalization if ``mean`` is not specified.
    2. It does normalization and color space conversion after stacking batch.
    3. It supports batch augmentations like mixup and cutmix.

    It provides the data pre-processing as follows

    - Collate and move data to the target device.
    - Pad inputs to the maximum size of current batch with defined
      ``pad_value``. The padding size can be divisible by a defined
      ``pad_size_divisor``
    - Stack inputs to batch_inputs.
    - Convert inputs from bgr to rgb if the shape of input is (3, H, W).
    - Normalize image with defined std and mean.
    - Do batch augmentations like Mixup and Cutmix during training.

    Args:
        mean (Sequence[Number], optional): The pixel mean of R, G, B channels.
            Defaults to None.
        std (Sequence[Number], optional): The pixel standard deviation of
            R, G, B channels. Defaults to None.
        pad_size_divisor (int): The size of padded image should be
            divisible by ``pad_size_divisor``. Defaults to 1.
        pad_value (Number): The padded pixel value. Defaults to 0.
        to_rgb (bool): whether to convert image from BGR to RGB.
            Defaults to False.
        to_onehot (bool): Whether to generate one-hot format gt-labels and set
            to data samples. Defaults to False.
        num_classes (int, optional): The number of classes. Defaults to None.
        batch_augments (dict, optional): The batch augmentations settings,
            including "augments" and "probs". For more details, see
            :class:`mmpretrain.models.RandomBatchAugment`.
    """

    def __init__(self,
                 mean: Sequence[Number] = None,
                 std: Sequence[Number] = None,
                 pad_size_divisor: int = 1,
                 pad_value: Number = 0,
                 to_rgb: bool = False,
                 to_onehot: bool = False,
                 num_classes: Optional[int] = None,
                 batch_augments: Optional[dict] = None,
                 **kwargs):
        super().__init__()
        self.pad_size_divisor = pad_size_divisor
        self.pad_value = pad_value
        self.to_rgb = to_rgb
        self.to_onehot = to_onehot
        self.num_classes = num_classes

        if mean is not None:
            assert std is not None, 'To enable the normalization in ' \
                                    'preprocessing, please specify both `mean` and `std`.'
            # Enable the normalization in preprocessing.
            self._enable_normalize = True
            self.register_buffer('mean', torch.tensor(mean).view(-1, 1, 1), False)
            self.register_buffer('std', torch.tensor(std).view(-1, 1, 1), False)
        else:
            self._enable_normalize = False

        if self.pad_size_divisor > 1:
            raise NotImplementedError()

    def forward(self, data: dict, training: bool = False) -> dict:
        """Perform normalization, padding, bgr2rgb conversion and batch
        augmentation based on ``BaseDataPreprocessor``.

        Args:
            data (dict): data sampled from dataloader.
            training (bool): Whether to enable training time augmentation.

        Returns:
            dict: Data in the same format as the model input.
        """
        inputs = self.cast_data(data['inputs'])

        if isinstance(inputs, torch.Tensor):
            # The branch if use `default_collate` as the collate_fn in the
            # dataloader.

            # ------ To RGB ------
            if self.to_rgb and inputs.size(1) == 3:
                inputs = inputs.flip(1)

            # -- Normalization ---
            inputs = inputs.float()
            if self._enable_normalize:
                inputs = (inputs - self.mean) / self.std

            # ------ Padding -----
            if self.pad_size_divisor > 1:
                h, w = inputs.shape[-2:]

                target_h = math.ceil(h / self.pad_size_divisor) * self.pad_size_divisor
                target_w = math.ceil(w / self.pad_size_divisor) * self.pad_size_divisor
                pad_h = target_h - h
                pad_w = target_w - w
                inputs = F.pad(inputs, (0, pad_w, 0, pad_h), 'constant', self.pad_value)
        else:
            # The branch if use `pseudo_collate` as the collate_fn in the dataloader.

            processed_inputs = []
            for input_ in inputs:
                # ------ To RGB ------
                if self.to_rgb and input_.size(0) == 3:
                    input_ = input_.flip(0)

                # -- Normalization ---
                input_ = input_.float()
                if self._enable_normalize:
                    input_ = (input_ - self.mean) / self.std

                processed_inputs.append(input_)
            # Combine padding and stack
            inputs = stack_batch(processed_inputs, self.pad_size_divisor, self.pad_value)

        data_samples = data.get('data_samples', None)
        sample_item = data_samples[0] if data_samples is not None else None

        if isinstance(sample_item, UnifiedDataSample):
            data_samples = self.cast_data(data_samples)
        else:
            raise NotImplementedError()

        return {'inputs': inputs, 'data_samples': data_samples}


@MODELS.register_module()
class UnifiedVideoDataPreprocessor(BaseDataPreprocessor):
    """Video pre-processor for classification tasks.

    Args:
        mean (Sequence[float or int], optional): The pixel mean of channels
            of images or stacked optical flow. Defaults to None.
        std (Sequence[float or int], optional): The pixel standard deviation
            of channels of images or stacked optical flow. Defaults to None.
        to_rgb (bool): Whether to convert image from BGR to RGB.
            Defaults to False.
        to_float32 (bool): Whether to convert data to float32.
            Defaults to True.
        blending (dict, optional): Config for batch blending.
            Defaults to None.
        format_shape (str): Format shape of input data.
            Defaults to ``'NCHW'``.
    """

    def __init__(self,
                 mean: Optional[Sequence[Union[float, int]]] = None,
                 std: Optional[Sequence[Union[float, int]]] = None,
                 to_rgb: bool = False,
                 to_float32: bool = True,
                 # blending: Optional[dict] = None,
                 format_shape: str = 'NCHW',
                 **kwargs
                 ) -> None:
        super().__init__()
        self.to_rgb = to_rgb
        self.to_float32 = to_float32
        self.format_shape = format_shape

        if mean is not None:
            assert std is not None, 'To enable the normalization in ' \
                                    'preprocessing, please specify both ' \
                                    '`mean` and `std`.'
            # Enable the normalization in preprocessing.
            self._enable_normalize = True
            if self.format_shape == 'NCHW':
                normalizer_shape = (-1, 1, 1)
            elif self.format_shape in ['NCTHW', 'MIX2d3d']:
                normalizer_shape = (-1, 1, 1, 1)
            else:
                raise ValueError(f'Invalid format shape: {format_shape}')

            self.register_buffer('mean', torch.tensor(mean, dtype=torch.float32).view(normalizer_shape), False)
            self.register_buffer('std', torch.tensor(std, dtype=torch.float32).view(normalizer_shape), False)
        else:
            self._enable_normalize = False

    def forward(self,
                data: Union[dict, Tuple[dict]],
                training: bool = False) -> Union[dict, Tuple[dict]]:
        """Perform normalization, padding, bgr2rgb conversion and batch
        augmentation based on ``BaseDataPreprocessor``.

        Args:
            data (dict or Tuple[dict]): data sampled from dataloader.
            training (bool): Whether to enable training time augmentation.

        Returns:
            dict or Tuple[dict]: Data in the same format as the model input.
        """
        data = self.cast_data(data)
        if isinstance(data, dict):
            return self.forward_onesample(data, training=training)
        elif isinstance(data, tuple):
            outputs = []
            for data_sample in data:
                output = self.forward_onesample(data_sample, training=training)
                outputs.append(output)
            return tuple(outputs)
        else:
            raise TypeError(f'Unsupported data type: {type(data)}!')

    def forward_onesample(self, data, training: bool = False) -> dict:
        """Perform normalization, padding, bgr2rgb conversion and batch
        augmentation on one data sample.

        Args:
            data (dict): data sampled from dataloader.
            training (bool): Whether to enable training time augmentation.

        Returns:
            dict: Data in the same format as the model input.
        """
        inputs, data_samples = data['inputs'], data['data_samples']
        inputs = self.preprocess_inputs(inputs)

        data['inputs'] = inputs
        data['data_samples'] = data_samples
        return data

    def preprocess_inputs(self, inputs: List[torch.Tensor]) -> Tuple:
        # --- Pad and stack --
        batch_inputs = stack_batch(inputs)

        if self.format_shape == 'MIX2d3d':
            if batch_inputs.ndim == 4:
                format_shape, view_shape = 'NCHW', (-1, 1, 1)
            else:
                format_shape, view_shape = 'NCTHW', None
        else:
            format_shape, view_shape = self.format_shape, None

        # ------ To RGB ------
        if self.to_rgb:
            if format_shape == 'NCHW':
                batch_inputs = batch_inputs[..., [2, 1, 0], :, :]
            elif format_shape == 'NCTHW':
                batch_inputs = batch_inputs[..., [2, 1, 0], :, :, :]
            else:
                raise ValueError(f'Invalid format shape: {format_shape}')

        # -- Normalization ---
        if self._enable_normalize:
            if view_shape is None:
                batch_inputs = (batch_inputs - self.mean) / self.std
            else:
                mean = self.mean.view(view_shape)
                std = self.std.view(view_shape)
                batch_inputs = (batch_inputs - mean) / std
        elif self.to_float32:
            batch_inputs = batch_inputs.to(torch.float32)

        return batch_inputs


@MODELS.register_module()
class UnifiedDataPreprocessor(BaseDataPreprocessor):
    def __init__(self,
                 mean: Sequence[Number] = None,
                 std: Sequence[Number] = None,
                 to_rgb: bool = False,
                 format_shape='NCHW',
                 batch_sizes=None,
                 **kwargs):
        super().__init__()

        self.image_preprocessor = UnifiedImageDataPreprocessor(mean=mean, std=std, to_rgb=to_rgb, **kwargs)
        self.video_preprocessor = UnifiedVideoDataPreprocessor(mean=mean, std=std, to_rgb=to_rgb,
                                                               format_shape=format_shape, **kwargs)
        self.batch_sizes = batch_sizes

    def forward(self, data: dict, training: bool = False) -> dict:
        if self.training and self.batch_sizes is not None:
            inputs = [[] for i in range(len(self.batch_sizes))]
            data_samples = [[] for i in range(len(self.batch_sizes))]
            data_types = []
        else:
            inputs = [[]]
            data_samples = [[]]
            data_types = []

        def find_batch_index_np(batch_sizes, i):
            if batch_sizes is None:
                return 0
            boundaries = np.cumsum(batch_sizes)
            return np.searchsorted(boundaries, i, side='right')

        for i, (x, data_sample) in enumerate(zip(data['inputs'], data['data_samples'])):
            if self.training:
                index = find_batch_index_np(self.batch_sizes, i)
            else:
                index = 0
            inputs[index].append(x)
            data_samples[index].append(data_sample)

        for x in inputs:
            if isinstance(x, List):
                data_shape = x[0].shape
            elif isinstance(x, torch.Tensor):
                data_shape = x.shape
            else:
                raise NotImplementedError(type(x))
            data_types.append('image' if len(data_shape) == 3 else 'video')

        new_data = {'inputs': [], 'data_samples': []}

        for inputs_i, data_samples_i, data_type in zip(inputs, data_samples, data_types):
            data_i = {'inputs': inputs_i, 'data_samples': data_samples_i}

            if data_type == 'image':
                data_i = self.image_preprocessor(data_i, training=training)
            elif data_type == 'video':
                data_i = self.video_preprocessor(data_i, training=training)

            new_data['inputs'].append(data_i['inputs'])
            new_data['data_samples'].append(data_i['data_samples'])

        new_data['inputs'] = {
            'data': new_data['inputs'],
            'data_types': data_types
        }
        return new_data
