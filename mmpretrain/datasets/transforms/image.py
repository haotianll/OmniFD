import math
import os
import warnings
from typing import Optional, Sequence

import mmcv
import mmengine
import mmengine.fileio as fileio
import numpy as np
import torch
import torchvision.transforms.functional as F
from PIL import Image
from mmcv.transforms import BaseTransform
from mmcv.transforms import LoadAnnotations as MMCV_LoadAnnotations
from mmengine.structures import PixelData

from mmpretrain.models.omnifd.base import UnifiedDataSample
from mmpretrain.registry import TRANSFORMS
from .formatting import to_tensor
from .processing import RandomCrop as _RandomCrop


def to_tensor(data):
    """Convert objects of various python types to :obj:`torch.Tensor`.

    Supported types are: :class:`numpy.ndarray`, :class:`torch.Tensor`,
    :class:`Sequence`, :class:`int` and :class:`float`.
    """
    if isinstance(data, torch.Tensor):
        return data
    elif isinstance(data, np.ndarray):
        return torch.from_numpy(data)
    elif isinstance(data, Sequence) and not mmengine.is_str(data):
        return torch.tensor(data)
    elif isinstance(data, int):
        return torch.LongTensor([data])
    elif isinstance(data, float):
        return torch.FloatTensor([data])
    else:
        raise TypeError(
            f'Type {type(data)} cannot be converted to tensor.'
            'Supported types are: `numpy.ndarray`, `torch.Tensor`, '
            '`Sequence`, `int` and `float`')


@TRANSFORMS.register_module()
class FaceCrop(BaseTransform):
    def __init__(self,
                 scale=1., ):
        self.scale = scale

    @staticmethod
    def rescale_bbox(bbox, height, width, scale):
        x1, y1, x2, y2 = bbox

        w = int((x2 - x1) * scale)
        h = int((y2 - y1) * scale)
        center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2

        x1 = max(int(center_x - w // 2), 0)
        y1 = max(int(center_y - h // 2), 0)
        w = min(width - x1, w)
        h = min(height - y1, h)

        x2 = x1 + w
        y2 = y1 + h
        return x1, y1, x2, y2

    def transform(self, results: dict) -> Optional[dict]:

        face_data = results.get('face_data', None)
        if len(face_data) > 0:
            bbox = face_data
        else:
            bbox = None

        results['full_shape'] = results['ori_shape']

        if bbox is not None:
            h, w = results['ori_shape']

            bbox = self.rescale_bbox(bbox, h, w, scale=self.scale)
            x1, y1, x2, y2 = bbox

            img = results['img']
            img = img[y1:y2, x1:x2]

            results['img'] = img
            results['img_shape'] = img.shape[:2]
            results['ori_shape'] = img.shape[:2]

            results['face_pad_info'] = x1, w - x2, y1, h - y2  # top, bottom, left, right

            if results.get('gt_seg_map') is not None:
                gt_seg_map = results['gt_seg_map']
                gt_seg_map = gt_seg_map[y1:y2, x1:x2]
                results['gt_seg_map'] = gt_seg_map

        return results


@TRANSFORMS.register_module()
class LoadMaskAnnotations(MMCV_LoadAnnotations):
    """Load and process the ``instances`` and ``seg_map`` annotation provided
    by dataset.

    The annotation format is as the following:

    .. code-block:: python

        {
            # Filename of semantic or panoptic segmentation ground truth file.
            'seg_map_path': 'a/b/c'
        }

    After this module, the annotation has been changed to the format below:

    .. code-block:: python

        {
             # In uint8 type.
            'gt_seg_map': np.ndarray (H, W)
        }

    Required Keys:

    - seg_map_path (str): Path of semantic segmentation ground truth file.

    Added Keys:

    - gt_seg_map (np.uint8)

    Args:
        with_bbox (bool): Whether to parse and load the bbox annotation.
            Defaults to True.
        with_label (bool): Whether to parse and load the label annotation.
            Defaults to True.
        with_seg (bool): Whether to parse and load the semantic segmentation
            annotation. Defaults to False.
        with_keypoints (bool): Whether to parse and load the keypoints
            annotation. Defaults to False.
        imdecode_backend (str): The image decoding backend type. The backend
            argument for :func:`mmcv.imfrombytes`.
            See :func:`mmcv.imfrombytes` for details.
            Defaults to 'cv2'.
        file_client_args (dict, optional): Arguments to instantiate a
            FileClient. See :class:`mmengine.fileio.FileClient` for details.
            Defaults to None. It will be deprecated in future. Please use
            ``backend_args`` instead.
            Deprecated in version 2.0.0rc4.
        backend_args (dict, optional): Instantiates the corresponding file
            backend. It may contain `backend` key to specify the file
            backend. If it contains, the file backend corresponding to this
            value will be used and initialized with the remaining values,
            otherwise the corresponding file backend will be selected
            based on the prefix of the file path. Defaults to None.
            New in version 2.0.0rc4.
    """

    def __init__(
            self,
            imdecode_backend='cv2',
            mode='train',
            **kwargs
    ) -> None:
        super().__init__(
            with_bbox=False,
            with_label=False,
            with_seg=True,
            with_keypoints=False,
            imdecode_backend=imdecode_backend, **kwargs)

        self.mode = mode

    def _load_seg_map(self, results: dict) -> None:
        """Private function to load semantic segmentation annotations.

        Args:
            results (dict): Result dict from
                :class:`mmengine.dataset.BaseDataset`.

        Returns:
            dict: The dict contains loaded semantic segmentation annotations.
        """
        if os.path.exists(results['seg_map_path']):
            if self.file_client_args is not None:
                file_client = fileio.FileClient.infer_client(self.file_client_args, results['seg_map_path'])
                img_bytes = file_client.get(results['seg_map_path'])
            else:
                img_bytes = fileio.get(results['seg_map_path'], backend_args=self.backend_args)

            results['gt_seg_map'] = mmcv.imfrombytes(
                img_bytes, flag='grayscale', backend=self.imdecode_backend).squeeze()
        else:
            if self.mode == 'test':
                gt_seg_map = 255 * np.ones((1, 1), dtype=np.uint8)
            else:
                gt_seg_map = np.zeros(results['ori_shape'], dtype=np.uint8)

            results['gt_seg_map'] = gt_seg_map


@TRANSFORMS.register_module()
class RandomCrop(_RandomCrop):
    def transform(self, results: dict) -> dict:
        """Transform function to randomly crop images.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Randomly cropped results, 'img_shape'
                key in result dict is updated according to crop size.
        """
        img = results['img']

        if self.padding is not None:
            img = mmcv.impad(img, padding=self.padding, pad_val=self.pad_val)

        # pad img if needed
        if self.pad_if_needed:
            h_pad = math.ceil(max(0, self.crop_size[0] - img.shape[0]) / 2)
            w_pad = math.ceil(max(0, self.crop_size[1] - img.shape[1]) / 2)

            img = mmcv.impad(img, padding=(w_pad, h_pad, w_pad, h_pad),
                             pad_val=self.pad_val, padding_mode=self.padding_mode)

        offset_h, offset_w, target_h, target_w = self.rand_crop_params(img)
        img = mmcv.imcrop(
            img,
            np.array([
                offset_w,
                offset_h,
                offset_w + target_w - 1,
                offset_h + target_h - 1,
            ]))
        results['img'] = img
        results['img_shape'] = img.shape

        gt_seg_map = results.get('gt_seg_map', None)
        if gt_seg_map is not None:
            if self.padding is not None:
                gt_seg_map = mmcv.impad(gt_seg_map, padding=self.padding, pad_val=0)

            if self.pad_if_needed:
                gt_seg_map = mmcv.impad(gt_seg_map, padding=(w_pad, h_pad, w_pad, h_pad),
                                        pad_val=0, padding_mode=self.padding_mode)

            gt_seg_map = mmcv.imcrop(gt_seg_map,
                                     np.array([offset_w, offset_h, offset_w + target_w - 1, offset_h + target_h - 1, ]))

            results['gt_seg_map'] = gt_seg_map

        return results


@TRANSFORMS.register_module()
class FilterKeys(BaseTransform):
    def __init__(self, tasks=None):
        self.tasks = tasks

    def transform(self, results: dict) -> Optional[dict]:
        if self.tasks is None:  # keep all task annotations
            return results

        if 'image' not in self.tasks and 'video' not in self.tasks:
            results.pop('gt_label', None)

        if 'temporal' not in self.tasks:
            results.pop('gt_segments', None)
            results.pop('gt_labels', None)

        if 'spatial' not in self.tasks:
            results.pop('seg_map_path', None)
            results.pop('gt_seg_map', None)

        return results


@TRANSFORMS.register_module()
class PackImageInputs(BaseTransform):
    DEFAULT_META_KEYS = (
        'sample_idx', 'img_path', 'ori_shape', 'img_shape', 'scale_factor', 'flip', 'flip_direction',
        'gt_seg_map', 'seg_map_path', 'face_pad_info', 'full_shape',
        'face_data'
    )

    def __init__(self,
                 input_key='img',
                 algorithm_keys=(),
                 meta_keys=DEFAULT_META_KEYS,
                 ):
        self.input_key = input_key
        self.algorithm_keys = algorithm_keys
        self.meta_keys = meta_keys

    @staticmethod
    def format_input(input_):
        if isinstance(input_, list):
            return [PackImageInputs.format_input(item) for item in input_]
        elif isinstance(input_, np.ndarray):
            if input_.ndim == 2:  # For grayscale image.
                input_ = np.expand_dims(input_, -1)
            if input_.ndim == 3 and not input_.flags.c_contiguous:
                input_ = np.ascontiguousarray(input_.transpose(2, 0, 1))
                input_ = to_tensor(input_)
            elif input_.ndim == 3:
                # convert to tensor first to accelerate, see
                # https://github.com/open-mmlab/mmdetection/pull/9533
                input_ = to_tensor(input_).permute(2, 0, 1).contiguous()
            else:
                # convert input with other shape to tensor without permute,
                # like video input (num_crops, C, T, H, W).
                input_ = to_tensor(input_)
        elif isinstance(input_, Image.Image):
            input_ = F.pil_to_tensor(input_)
        elif not isinstance(input_, torch.Tensor):
            raise TypeError(f'Unsupported input type {type(input_)}.')

        return input_

    def transform(self, results: dict) -> dict:
        """Method to pack the input data."""

        packed_results = dict()
        if self.input_key in results:
            input_ = results[self.input_key]
            packed_results['inputs'] = self.format_input(input_)

        data_sample = UnifiedDataSample(data_type='image')

        # Set default keys
        # Classification
        if 'gt_label' in results:
            data_sample.tasks('image').set_gt_label(results['gt_label'])

        # Segmentation
        if 'gt_seg_map' in results:
            if len(results['gt_seg_map'].shape) == 2:
                data = to_tensor(results['gt_seg_map'][None, ...].astype(np.int64))
            else:
                warnings.warn(f'Please pay attention your ground truth segmentation map, '
                              f'usually the segmentation map is 2D, but got {results["gt_seg_map"].shape}')
                data = to_tensor(results['gt_seg_map'].astype(np.int64))

            gt_sem_seg_data = dict(data=data)
            data_sample.tasks('spatial').gt_sem_seg = PixelData(**gt_sem_seg_data)

        # Set custom algorithm keys
        for key in self.algorithm_keys:
            if key in results:
                data_sample.set_field(results[key], key)

        # Set meta keys
        for key in self.meta_keys:
            if key in results:
                data_sample.set_field(results[key], key, field_type='metainfo')

        packed_results['data_samples'] = data_sample
        return packed_results
