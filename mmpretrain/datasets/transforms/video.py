import inspect
import io
import math
import os
import os.path as osp
import random
from typing import Callable, Union, Dict, List, Optional, Tuple, Sequence

import albumentations
import cv2
import mmcv
import mmengine
import mmengine.fileio as fileio
import numpy as np
import torch
from mmcv.image.geometric import _scale_size
from mmcv.transforms import BaseTransform
from mmcv.transforms.processing import Resize, RandomResize, RandomFlip
from mmengine.fileio import FileClient
from mmengine.structures import InstanceData

from mmpretrain.datasets.transforms.image import PackImageInputs, to_tensor
from mmpretrain.datasets.transforms.processing import RandomCrop
from mmpretrain.models.omnifd.base import UnifiedDataSample
from mmpretrain.registry import TRANSFORMS

Transform = Union[dict, Callable[[dict], dict]]


def _init_lazy_if_proper(results, lazy):
    """Initialize lazy operation properly.

    Make sure that a lazy operation is properly initialized,
    and avoid a non-lazy operation accidentally getting mixed in.

    Required keys in results are "imgs" if "img_shape" not in results,
    otherwise, Required keys in results are "img_shape", add or modified keys
    are "img_shape", "lazy".
    Add or modified keys in "lazy" are "original_shape", "crop_bbox", "flip",
    "flip_direction", "interpolation".

    Args:
        results (dict): A dict stores data pipeline result.
        lazy (bool): Determine whether to apply lazy operation. Default: False.
    """

    if 'img_shape' not in results:
        results['img_shape'] = results['imgs'][0].shape[:2]
    if lazy:
        if 'lazy' not in results:
            img_h, img_w = results['img_shape']
            lazyop = dict()
            lazyop['original_shape'] = results['img_shape']
            lazyop['crop_bbox'] = np.array([0, 0, img_w, img_h],
                                           dtype=np.float32)
            lazyop['flip'] = False
            lazyop['flip_direction'] = None
            lazyop['interpolation'] = None
            results['lazy'] = lazyop
    else:
        assert 'lazy' not in results, 'Use Fuse after lazy operations'


@TRANSFORMS.register_module()
class VideoDecordInit(BaseTransform):
    """Using decord to initialize the video_reader.

    Decord: https://github.com/dmlc/decord

    Required Keys:

        - filename

    Added Keys:

        - video_reader
        - total_frames
        - fps

    Args:
        io_backend (str): io backend where frames are store.
            Defaults to ``'disk'``.
        num_threads (int): Number of thread to decode the video. Defaults to 1.
        kwargs (dict): Args for file client.
    """

    def __init__(self,
                 io_backend: str = 'disk',
                 num_threads: int = 1,
                 **kwargs) -> None:
        self.io_backend = io_backend
        self.num_threads = num_threads
        self.kwargs = kwargs
        self.file_client = None

    def _get_video_reader(self, filename: str) -> object:
        if osp.splitext(filename)[0] == filename:
            filename = filename + '.mp4'
        try:
            import decord
        except ImportError:
            raise ImportError(
                'Please run "pip install decord" to install Decord first.')

        if self.file_client is None:
            self.file_client = FileClient(self.io_backend, **self.kwargs)
        file_obj = io.BytesIO(self.file_client.get(filename))
        container = decord.VideoReader(file_obj, num_threads=self.num_threads)
        return container

    def transform(self, results: Dict) -> Dict:
        """Perform the Decord initialization.

        Args:
            results (dict): The result dict.

        Returns:
            dict: The result dict.
        """
        container = self._get_video_reader(results['video_path'])
        results['total_frames'] = len(container)

        results['video_reader'] = container
        results['avg_fps'] = container.get_avg_fps()
        return results

    def __repr__(self) -> str:
        repr_str = (f'{self.__class__.__name__}('
                    f'io_backend={self.io_backend}, '
                    f'num_threads={self.num_threads})')
        return repr_str


@TRANSFORMS.register_module()
class VideoDecordInitDir(BaseTransform):
    """Using decord to initialize the video_reader.

    Decord: https://github.com/dmlc/decord

    Required Keys:

        - filename

    Added Keys:

        - video_reader
        - total_frames
        - fps

    Args:
        io_backend (str): io backend where frames are store.
            Defaults to ``'disk'``.
        num_threads (int): Number of thread to decode the video. Defaults to 1.
        kwargs (dict): Args for file client.
    """

    def __init__(self,
                 extensions=('jpg', 'jpeg', 'png', 'bmp', 'tif'),
                 **kwargs) -> None:
        self.kwargs = kwargs
        self.extensions = extensions

    def transform(self, results: Dict) -> Dict:
        """Perform the Decord initialization.

        Args:
            results (dict): The result dict.

        Returns:
            dict: The result dict.
        """
        results['total_frames'] = len(results['frames_list'])
        results['avg_fps'] = None
        return results

    def __repr__(self) -> str:
        repr_str = (f'{self.__class__.__name__})')
        return repr_str


@TRANSFORMS.register_module()
class LoadTadAnnotations:
    def __init__(self, revise_label=True):
        self.revise_label = revise_label

    def __call__(self, results):
        gt_segments = []
        gt_labels = []

        for ann in results['gt_segments']:
            if self.revise_label:  # only predict fake class
                if ann['label'] == 0:  # filter real segment
                    continue
                elif ann['label'] == 1:
                    ann['label'] = 0
                else:
                    raise NotImplementedError()

            gt_segments.append(ann['frame'])
            gt_labels.append(ann['label'])

        results['gt_segments'] = np.array(gt_segments)
        results['gt_labels'] = np.array(gt_labels)
        results['with_tad'] = True
        return results


@TRANSFORMS.register_module()
class VideoSlidingWindow:
    def __init__(
            self,
            feature_stride=4,  # the frames between two adjacent features, such as 4 frames
            sample_stride=1,  # if you want to extract the feature[::sample_stride]
            offset_frames=0,  # the start offset frame of the input feature
            window_size=-1,  # the number of features in a window
            window_overlap_ratio=0.25,  # the overlap ratio of two adjacent windows
            ioa_thresh=0.,  # the threshold of the completeness of the gt inside the window
            is_eval=False,
    ):
        # feature settings
        self.feature_stride = int(feature_stride)
        self.sample_stride = int(sample_stride)
        self.offset_frames = int(offset_frames)
        self.snippet_stride = int(feature_stride * sample_stride)

        # window settings
        self.window_size = int(window_size)
        self.window_stride = int(window_size * (1 - window_overlap_ratio))
        self.ioa_thresh = ioa_thresh

        self.is_eval = is_eval

    @staticmethod
    def compute_gt_completeness(gt_boxes, anchors):
        """Compute the completeness of the gt_bboxes.
           GT will be first truncated by the anchor start/end, then the completeness is defined as the ratio of the truncated_gt_len / original_gt_len.
           If this ratio is too small, it means this gt is not complete enough to be used for training.
        Args:
            gt_boxes: np.array shape [N, 2]
            anchors:  np.array shape [2]
        """

        scores = np.zeros(gt_boxes.shape[0])
        valid_idx = np.logical_and(gt_boxes[:, 0] < anchors[1], gt_boxes[:, 1] > anchors[0])  # valid gt
        valid_gt_boxes = gt_boxes[valid_idx]

        truncated_valid_gt_len = np.minimum(valid_gt_boxes[:, 1], anchors[1]) - np.maximum(valid_gt_boxes[:, 0],
                                                                                           anchors[0])
        original_valid_gt_len = np.maximum(valid_gt_boxes[:, 1] - valid_gt_boxes[:, 0], 1e-6)
        scores[valid_idx] = truncated_valid_gt_len / original_valid_gt_len

        truncated_gt_boxes = np.stack(
            [np.maximum(gt_boxes[:, 0], anchors[0]), np.minimum(gt_boxes[:, 1], anchors[1])], axis=1
        )
        return scores, truncated_gt_boxes

    def split_video_to_windows(self, results):
        video_snippet_centers = np.arange(0, results['total_frames'], self.snippet_stride)
        snippet_num = len(video_snippet_centers)

        data_list = []
        last_window = False

        for idx in range(max(1, snippet_num // self.window_stride)):
            window_start = idx * self.window_stride
            window_end = window_start + self.window_size

            if window_end > snippet_num:
                window_end = snippet_num
                window_start = max(0, window_end - self.window_size)
                last_window = True

            window_snippet_centers = video_snippet_centers[window_start:window_end]
            window_start_frame = window_snippet_centers[0]
            window_end_frame = window_snippet_centers[-1]

            if not self.is_eval and results.get('gt_segments') is not None and self.ioa_thresh > 0:
                gt_segments = results['gt_segments']
                gt_labels = results['gt_labels']
                anchor = np.array([window_start_frame, window_end_frame])

                gt_completeness, truncated_gt = self.compute_gt_completeness(gt_segments, anchor)
                valid_idx = gt_completeness > self.ioa_thresh

                if np.sum(valid_idx) > 0:
                    window_ann = dict(
                        gt_segments=truncated_gt[valid_idx],
                        gt_labels=gt_labels[valid_idx],
                    )
                    data_list.append([window_ann, window_snippet_centers])
            else:
                data_list.append([None, window_snippet_centers])

            if last_window:
                break

        if not self.is_eval:
            data = random.choice(data_list)
        else:
            data = data_list[0]
        return data

    def __call__(self, results):

        data = self.split_video_to_windows(results)
        ann, window_snippet_centers = data

        if ann is not None:
            results.update(ann)

        if 'gt_segments' in results.keys() and len(results['gt_segments']) > 0:
            if self.is_eval:
                results['gt_segments'] = results['gt_segments'] / results['avg_fps']
            else:
                results['gt_segments'] = results['gt_segments'] - window_snippet_centers[0] - self.offset_frames
                results['gt_segments'] = results['gt_segments'] / self.snippet_stride
                results['video_snippet_centers'] = window_snippet_centers

        results['feature_start_idx'] = int(window_snippet_centers[0] / self.snippet_stride)
        results['feature_end_idx'] = int(window_snippet_centers[-1] / self.snippet_stride)
        results['snippet_stride'] = self.snippet_stride
        results['window_start_frame'] = window_snippet_centers[0]

        results['window_size'] = self.window_size
        results['offset_frames'] = self.offset_frames
        return results


@TRANSFORMS.register_module()
class LoadFrames:
    def __init__(
            self,
            num_clips=1,
            scale_factor=1,
            method='resize',
            trunc_len=None,
            trunc_thresh=0.,
            crop_ratio=None,
    ):
        self.num_clips = num_clips
        self.scale_factor = scale_factor
        self.method = method

        self.trunc_len = trunc_len
        self.trunc_thresh = trunc_thresh
        self.crop_ratio = crop_ratio

    def random_trunc(self, feats, trunc_len, gt_segments, gt_labels, offset=0, max_num_trials=200):
        feat_len = feats.shape[0]
        num_segs = gt_segments.shape[0]

        trunc_len = trunc_len
        if feat_len <= trunc_len:
            if self.crop_ratio is None:
                return feats, gt_segments, gt_labels
            else:
                trunc_len = random.randint(
                    max(round(self.crop_ratio[0] * feat_len), 1),
                    min(round(self.crop_ratio[1] * feat_len), feat_len),
                )

                if feat_len == trunc_len:
                    return feats, gt_segments, gt_labels
                
        if num_segs == 0:
            st = random.randint(0, feat_len - trunc_len)
            ed = st + trunc_len
            feats = feats[st:ed]
            return feats, gt_segments, gt_labels

        for _ in range(max_num_trials):
            st = random.randint(0, feat_len - trunc_len)
            ed = st + trunc_len
            window = np.array([st, ed], dtype=np.float32)

            window = np.repeat(window[None, :], num_segs, axis=0)
            left = np.maximum(window[:, 0] - offset, gt_segments[:, 0])
            right = np.minimum(window[:, 1] + offset, gt_segments[:, 1])
            inter = np.clip(right - left, a_min=0, a_max=None)
            area_segs = np.abs(gt_segments[:, 1] - gt_segments[:, 0])
            inter_ratio = inter / area_segs

            seg_idx = inter_ratio >= self.trunc_thresh

            if seg_idx.sum().item() > 0:
                break

        feats = feats[st:ed]
        gt_segments = np.stack((left[seg_idx], right[seg_idx]), axis=1)
        gt_segments = gt_segments - st
        gt_labels = gt_labels[seg_idx]
        return feats, gt_segments, gt_labels

    def __call__(self, results):
        assert 'total_frames' in results.keys(), 'should have total_frames as a key'
        total_frames = results['total_frames']
        fps = results['avg_fps']

        if self.method == 'resize':
            assert 'resize_length' in results.keys(), 'should have resize_length as a key'
            frame_num = results['resize_length'] * self.scale_factor
            frame_stride = total_frames / frame_num
            frame_idxs = np.arange(
                frame_stride / 2 - 0.5,
                total_frames + frame_stride / 2 - 0.5,
                frame_stride,
            )
            masks = torch.ones(results['resize_length']).bool()

            if 'gt_segments' in results.keys():
                results['gt_segments'] = np.clip(results['gt_segments'] / results['duration'], 0.0, 1.0)
                results['gt_segments'] *= results['resize_length']

        elif self.method == 'random_trunc':
            assert results['snippet_stride'] >= self.scale_factor, 'snippet_stride should be larger than scale_factor'
            assert (
                    results['snippet_stride'] % self.scale_factor == 0
            ), 'snippet_stride should be divisible by scale_factor'

            frame_num = self.trunc_len * self.scale_factor
            frame_stride = results['snippet_stride'] // self.scale_factor
            frame_idxs = np.arange(0, total_frames, frame_stride)

            frame_idxs, gt_segments, gt_labels = self.random_trunc(
                frame_idxs,
                trunc_len=frame_num,
                gt_segments=results['gt_segments'] * self.scale_factor,
                gt_labels=results['gt_labels'],
            )
            results['gt_segments'] = gt_segments / self.scale_factor
            results['gt_labels'] = gt_labels

            if len(frame_idxs) < frame_num:
                valid_len = len(frame_idxs) // self.scale_factor
                frame_idxs = np.pad(frame_idxs, (0, frame_num - len(frame_idxs)), mode='edge')
                masks = torch.cat([torch.ones(valid_len), torch.zeros(self.trunc_len - valid_len)]).bool()
            else:
                masks = torch.ones(self.trunc_len).bool()

        elif self.method == 'sliding_window':
            assert results['snippet_stride'] >= self.scale_factor, 'snippet_stride should be larger than scale_factor'
            assert (
                    results['snippet_stride'] % self.scale_factor == 0
            ), 'snippet_stride should be divisible by scale_factor'

            window_size = results['window_size']
            frame_num = window_size * self.scale_factor
            frame_stride = results['snippet_stride'] // self.scale_factor
            frame_idxs = np.arange(0, total_frames, frame_stride)

            start_idx = min(results['feature_start_idx'] * self.scale_factor, len(frame_idxs))
            end_idx = min((results['feature_end_idx'] + 1) * self.scale_factor, len(frame_idxs))

            frame_idxs = frame_idxs[start_idx:end_idx]

            if len(frame_idxs) < frame_num:
                valid_len = len(frame_idxs) // self.scale_factor
                frame_idxs = np.pad(frame_idxs, (0, frame_num - len(frame_idxs)), mode='edge')
                masks = torch.cat([torch.ones(valid_len), torch.zeros(window_size - valid_len)]).bool()
            else:
                masks = torch.ones(window_size).bool()

        elif self.method == 'padding':
            raise NotImplementedError

        frame_idxs = np.clip(frame_idxs, 0, total_frames - 1).round()

        assert frame_idxs.shape[0] == frame_num, 'snippet center number should be equal to snippet number'

        results['frame_inds'] = frame_idxs.astype(int)
        results['num_clips'] = self.num_clips
        results['clip_len'] = frame_num // self.num_clips
        results['masks'] = masks

        return results


@TRANSFORMS.register_module()
class LoadFramesFromDir(BaseTransform):

    def __init__(self,
                 color_type: str = 'color',
                 imdecode_backend: str = 'cv2',
                 backend_args: Optional[dict] = None
                 ):
        self.color_type = color_type
        self.imdecode_backend = imdecode_backend
        self.backend_args = backend_args

    def _load_frames(self, root, frames) -> List[np.ndarray]:
        imgs = []
        for frame in frames:
            filename = os.path.join(root, frame)
            img_bytes = fileio.get(filename, backend_args=self.backend_args)
            img = mmcv.imfrombytes(img_bytes, flag=self.color_type, backend=self.imdecode_backend)
            imgs.append(img)
        return imgs

    def transform(self, results: Dict) -> Dict:
        if results['frame_inds'].ndim != 1:
            results['frame_inds'] = np.squeeze(results['frame_inds'])

        frame_inds = results['frame_inds']
        frames = [results['frames_list'][i] for i in frame_inds]

        video_path = results['video_path']

        imgs = self._load_frames(video_path, frames)

        results['imgs'] = imgs
        results['original_shape'] = imgs[0].shape[:2]
        results['img_shape'] = imgs[0].shape[:2]

        return results

    def __repr__(self) -> str:
        repr_str = f'{self.__class__.__name__}(mode={self.mode})'
        return repr_str


@TRANSFORMS.register_module()
class VideoSampleFrames(BaseTransform):
    """Sample frames from the video.

    Required Keys:

        - total_frames
        - start_index

    Added Keys:

        - frame_inds
        - frame_interval
        - num_clips

    Args:
        clip_len (int): Frames of each sampled output clip.
        frame_interval (int): Temporal interval of adjacent sampled frames.
            Defaults to 1.
        num_clips (int): Number of clips to be sampled. Default: 1.
        temporal_jitter (bool): Whether to apply temporal jittering.
            Defaults to False.
        twice_sample (bool): Whether to use twice sample when testing.
            If set to True, it will sample frames with and without fixed shift,
            which is commonly used for testing in TSM model. Defaults to False.
        out_of_bound_opt (str): The way to deal with out of bounds frame
            indexes. Available options are 'loop', 'repeat_last'.
            Defaults to 'loop'.
        test_mode (bool): Store True when building test or validation dataset.
            Defaults to False.
        keep_tail_frames (bool): Whether to keep tail frames when sampling.
            Defaults to False.
        target_fps (optional, int): Convert input videos with arbitrary frame
            rates to the unified target FPS before sampling frames. If
            ``None``, the frame rate will not be adjusted. Defaults to
            ``None``.
    """

    def __init__(self,
                 clip_len: int,
                 frame_interval: int = 1,
                 num_clips: int = 1,
                 temporal_jitter: bool = False,
                 twice_sample: bool = False,
                 out_of_bound_opt: str = 'loop',
                 test_mode: bool = False,
                 keep_tail_frames: bool = False,
                 target_fps: Optional[int] = None,
                 **kwargs) -> None:

        self.clip_len = clip_len
        self.frame_interval = frame_interval
        self.num_clips = num_clips
        self.temporal_jitter = temporal_jitter
        self.twice_sample = twice_sample
        self.out_of_bound_opt = out_of_bound_opt
        self.test_mode = test_mode
        self.keep_tail_frames = keep_tail_frames
        self.target_fps = target_fps
        assert self.out_of_bound_opt in ['loop', 'repeat_last']

    def _get_train_clips(self, num_frames: int,
                         ori_clip_len: float) -> np.array:
        """Get clip offsets in train mode.

        It will calculate the average interval for selected frames,
        and randomly shift them within offsets between [0, avg_interval].
        If the total number of frames is smaller than clips num or origin
        frames length, it will return all zero indices.

        Args:
            num_frames (int): Total number of frame in the video.
            ori_clip_len (float): length of original sample clip.

        Returns:
            np.ndarray: Sampled frame indices in train mode.
        """

        if self.keep_tail_frames:
            avg_interval = (num_frames - ori_clip_len + 1) / float(
                self.num_clips)
            if num_frames > ori_clip_len - 1:
                base_offsets = np.arange(self.num_clips) * avg_interval
                clip_offsets = (base_offsets + np.random.uniform(
                    0, avg_interval, self.num_clips)).astype(np.int32)
            else:
                clip_offsets = np.zeros((self.num_clips,), dtype=np.int32)
        else:
            avg_interval = (num_frames - ori_clip_len + 1) // self.num_clips

            if avg_interval > 0:
                base_offsets = np.arange(self.num_clips) * avg_interval
                clip_offsets = base_offsets + np.random.randint(
                    avg_interval, size=self.num_clips)
            elif num_frames > max(self.num_clips, ori_clip_len):
                clip_offsets = np.sort(
                    np.random.randint(
                        num_frames - ori_clip_len + 1, size=self.num_clips))
            elif avg_interval == 0:
                ratio = (num_frames - ori_clip_len + 1.0) / self.num_clips
                clip_offsets = np.around(np.arange(self.num_clips) * ratio)
            else:
                clip_offsets = np.zeros((self.num_clips,), dtype=np.int32)

        return clip_offsets

    def _get_test_clips(self, num_frames: int,
                        ori_clip_len: float) -> np.array:
        """Get clip offsets in test mode.

        If the total number of frames is
        not enough, it will return all zero indices.

        Args:
            num_frames (int): Total number of frame in the video.
            ori_clip_len (float): length of original sample clip.

        Returns:
            np.ndarray: Sampled frame indices in test mode.
        """
        if self.clip_len == 1:  # 2D recognizer
            # assert self.frame_interval == 1
            avg_interval = num_frames / float(self.num_clips)
            base_offsets = np.arange(self.num_clips) * avg_interval
            clip_offsets = base_offsets + avg_interval / 2.0
            if self.twice_sample:
                clip_offsets = np.concatenate([clip_offsets, base_offsets])
        else:  # 3D recognizer
            max_offset = max(num_frames - ori_clip_len, 0)
            if self.twice_sample:
                num_clips = self.num_clips * 2
            else:
                num_clips = self.num_clips
            if num_clips > 1:
                num_segments = self.num_clips - 1
                # align test sample strategy with `PySlowFast` repo
                if self.target_fps is not None:
                    offset_between = np.floor(max_offset / float(num_segments))
                    clip_offsets = np.arange(num_clips) * offset_between
                else:
                    offset_between = max_offset / float(num_segments)
                    clip_offsets = np.arange(num_clips) * offset_between
                    clip_offsets = np.round(clip_offsets)
            else:
                clip_offsets = np.array([max_offset // 2])
        return clip_offsets

    def _sample_clips(self, num_frames: int, ori_clip_len: float) -> np.array:
        """Choose clip offsets for the video in a given mode.

        Args:
            num_frames (int): Total number of frame in the video.

        Returns:
            np.ndarray: Sampled frame indices.
        """
        if self.test_mode:
            clip_offsets = self._get_test_clips(num_frames, ori_clip_len)
        else:
            clip_offsets = self._get_train_clips(num_frames, ori_clip_len)

        return clip_offsets

    def _get_ori_clip_len(self, fps_scale_ratio: float) -> float:
        """calculate length of clip segment for different strategy.

        Args:
            fps_scale_ratio (float): Scale ratio to adjust fps.
        """
        if self.target_fps is not None:
            # align test sample strategy with `PySlowFast` repo
            ori_clip_len = self.clip_len * self.frame_interval
            ori_clip_len = np.maximum(1, ori_clip_len * fps_scale_ratio)
        elif self.test_mode:
            ori_clip_len = (self.clip_len - 1) * self.frame_interval + 1
        else:
            ori_clip_len = self.clip_len * self.frame_interval

        return ori_clip_len

    def transform(self, results: dict) -> dict:
        """Perform the SampleFrames loading.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        total_frames = results['total_frames']
        # if can't get fps, same value of `fps` and `target_fps`
        # will perform nothing
        fps = results.get('avg_fps')
        if self.target_fps is None or not fps:
            fps_scale_ratio = 1.0
        else:
            fps_scale_ratio = fps / self.target_fps
        ori_clip_len = self._get_ori_clip_len(fps_scale_ratio)
        clip_offsets = self._sample_clips(total_frames, ori_clip_len)

        if self.target_fps:
            frame_inds = clip_offsets[:, None] + np.linspace(
                0, ori_clip_len - 1, self.clip_len).astype(np.int32)
        else:
            frame_inds = clip_offsets[:, None] + np.arange(
                self.clip_len)[None, :] * self.frame_interval
            frame_inds = np.concatenate(frame_inds)

        if self.temporal_jitter:
            perframe_offsets = np.random.randint(
                self.frame_interval, size=len(frame_inds))
            frame_inds += perframe_offsets

        frame_inds = frame_inds.reshape((-1, self.clip_len))
        if self.out_of_bound_opt == 'loop':
            frame_inds = np.mod(frame_inds, total_frames)
        elif self.out_of_bound_opt == 'repeat_last':
            safe_inds = frame_inds < total_frames
            unsafe_inds = 1 - safe_inds
            last_ind = np.max(safe_inds * frame_inds, axis=1)
            new_inds = (safe_inds * frame_inds + (unsafe_inds.T * last_ind).T)
            frame_inds = new_inds
        else:
            raise ValueError('Illegal out_of_bound option.')

        start_index = results['start_index']
        frame_inds = np.concatenate(frame_inds) + start_index
        results['frame_inds'] = frame_inds.astype(np.int32)
        results['clip_len'] = self.clip_len
        results['frame_interval'] = self.frame_interval
        results['num_clips'] = self.num_clips
        return results

    def __repr__(self) -> str:
        repr_str = (f'{self.__class__.__name__}('
                    f'clip_len={self.clip_len}, '
                    f'frame_interval={self.frame_interval}, '
                    f'num_clips={self.num_clips}, '
                    f'temporal_jitter={self.temporal_jitter}, '
                    f'twice_sample={self.twice_sample}, '
                    f'out_of_bound_opt={self.out_of_bound_opt}, '
                    f'test_mode={self.test_mode})')
        return repr_str


@TRANSFORMS.register_module()
class VideoDecordDecode(BaseTransform):
    """Using decord to decode the video.

    Decord: https://github.com/dmlc/decord

    Required Keys:

        - video_reader
        - frame_inds

    Added Keys:

        - imgs
        - original_shape
        - img_shape

    Args:
        mode (str): Decoding mode. Options are 'accurate' and 'efficient'.
            If set to 'accurate', it will decode videos into accurate frames.
            If set to 'efficient', it will adopt fast seeking but only return
            key frames, which may be duplicated and inaccurate, and more
            suitable for large scene-based video datasets.
            Defaults to ``'accurate'``.
    """

    def __init__(self, mode: str = 'accurate') -> None:
        self.mode = mode
        assert mode in ['accurate', 'efficient']

    def _decord_load_frames(self, container: object,
                            frame_inds: np.ndarray) -> List[np.ndarray]:
        if self.mode == 'accurate':
            imgs = container.get_batch(frame_inds).asnumpy()
            imgs = list(imgs)
        elif self.mode == 'efficient':
            # This mode is faster, however it always returns I-FRAME
            container.seek(0)
            imgs = list()
            for idx in frame_inds:
                container.seek(idx)
                frame = container.next()
                imgs.append(frame.asnumpy())
        return imgs

    def transform(self, results: Dict) -> Dict:
        """Perform the Decord decoding.

        Args:
            results (dict): The result dict.

        Returns:
            dict: The result dict.
        """
        container = results['video_reader']

        if results['frame_inds'].ndim != 1:
            results['frame_inds'] = np.squeeze(results['frame_inds'])

        frame_inds = results['frame_inds']
        imgs = self._decord_load_frames(container, frame_inds)

        results['video_reader'] = None
        del container

        results['imgs'] = imgs
        results['original_shape'] = imgs[0].shape[:2]
        results['img_shape'] = imgs[0].shape[:2]

        # we resize the gt_bboxes and proposals to their real scale
        if 'gt_bboxes' in results:
            h, w = results['img_shape']
            scale_factor = np.array([w, h, w, h])
            gt_bboxes = results['gt_bboxes']
            gt_bboxes = (gt_bboxes * scale_factor).astype(np.float32)
            results['gt_bboxes'] = gt_bboxes
            if 'proposals' in results and results['proposals'] is not None:
                proposals = results['proposals']
                proposals = (proposals * scale_factor).astype(np.float32)
                results['proposals'] = proposals

        return results

    def __repr__(self) -> str:
        repr_str = f'{self.__class__.__name__}(mode={self.mode})'
        return repr_str


@TRANSFORMS.register_module()
class VideoFaceCrop(BaseTransform):
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

    def transform(self, results):
        if len(results['face_data']) > 0:
            results['face_data'] = [results['face_data'][i] for i in results['frame_inds']]

            for i, bbox in enumerate(results['face_data']):
                h, w = results['imgs'][i].shape[:2]
                bbox = self.rescale_bbox(bbox, h, w, self.scale)
                x1, y1, x2, y2 = bbox
                results['imgs'][i] = results['imgs'][i][y1:y2, x1:x2]
        return results


@TRANSFORMS.register_module()
class VideoResize(Resize):
    def mmcv_resize_img(self, img, scale) -> None:
        if self.keep_ratio:
            img, scale_factor = mmcv.imrescale(
                img,
                scale,
                interpolation=self.interpolation,
                return_scale=True,
                backend=self.backend)
            # the w_scale and h_scale has minor difference
            # a real fix should be done in the mmcv.imrescale in the future
            new_h, new_w = img.shape[:2]
            h, w = img.shape[:2]
            w_scale = new_w / w
            h_scale = new_h / h
        else:
            img, w_scale, h_scale = mmcv.imresize(
                img,
                scale,
                interpolation=self.interpolation,
                return_scale=True,
                backend=self.backend)
        return img

    def _resize_imgs(self, results: dict) -> None:
        results['imgs'] = [
            self.mmcv_resize_img(img, results['scale']) for img in results['imgs']
        ]
        results['img_shape'] = results['imgs'][0].shape[:2]
        results['keep_ratio'] = self.keep_ratio

    def transform(self, results: dict) -> dict:
        """Transform function to resize images, bounding boxes, semantic
        segmentation map and keypoints.

        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Resized results, 'img', 'gt_bboxes', 'gt_seg_map',
            'gt_keypoints', 'scale', 'scale_factor', 'img_shape',
            and 'keep_ratio' keys are updated in result dict.
        """

        if self.scale:
            results['scale'] = self.scale
        else:
            img_shape = results['imgs'][0].shape[:2]
            results['scale'] = _scale_size(img_shape[::-1], self.scale_factor)  # type: ignore
            raise NotImplementedError()

        self._resize_imgs(results)
        return results


@TRANSFORMS.register_module()
class VideoRandomResize(RandomResize):
    """Random resize images & bbox & keypoints.

    How to choose the target scale to resize the image will follow the rules
    below:

    - if ``scale`` is a sequence of tuple

    .. math::
        target\\_scale[0] \\sim Uniform([scale[0][0], scale[1][0]])
    .. math::
        target\\_scale[1] \\sim Uniform([scale[0][1], scale[1][1]])

    Following the resize order of weight and height in cv2, ``scale[i][0]``
    is for width, and ``scale[i][1]`` is for height.

    - if ``scale`` is a tuple

    .. math::
        target\\_scale[0] \\sim Uniform([ratio\\_range[0], ratio\\_range[1]])
            * scale[0]
    .. math::
        target\\_scale[0] \\sim Uniform([ratio\\_range[0], ratio\\_range[1]])
            * scale[1]

    Following the resize order of weight and height in cv2, ``ratio_range[0]``
    is for width, and ``ratio_range[1]`` is for height.

    - if ``keep_ratio`` is True, the minimum value of ``target_scale`` will be
      used to set the shorter side and the maximum value will be used to
      set the longer side.

    - if ``keep_ratio`` is False, the value of ``target_scale`` will be used to
      reisze the width and height accordingly.

    Required Keys:

    - img
    - gt_bboxes
    - gt_seg_map
    - gt_keypoints

    Modified Keys:

    - img
    - gt_bboxes
    - gt_seg_map
    - gt_keypoints
    - img_shape

    Added Keys:

    - scale
    - scale_factor
    - keep_ratio

    Args:
        scale (tuple or Sequence[tuple]): Images scales for resizing.
            Defaults to None.
        ratio_range (tuple[float], optional): (min_ratio, max_ratio).
            Defaults to None.
        resize_type (str): The type of resize class to use. Defaults to
            "Resize".
        **resize_kwargs: Other keyword arguments for the ``resize_type``.

    Note:
        By defaults, the ``resize_type`` is "Resize", if it's not overwritten
        by your registry, it indicates the :class:`mmcv.Resize`. And therefore,
        ``resize_kwargs`` accepts any keyword arguments of it, like
        ``keep_ratio``, ``interpolation`` and so on.

        If you want to use your custom resize class, the class should accept
        ``scale`` argument and have ``scale`` attribution which determines the
        resize shape.
    """

    def __init__(
            self,
            scale: Union[Tuple[int, int], Sequence[Tuple[int, int]]],
            ratio_range: Tuple[float, float] = None,
            resize_type: str = 'VideoResize',
            **resize_kwargs,
    ) -> None:
        self.scale = scale
        self.ratio_range = ratio_range

        self.resize_cfg = dict(type=resize_type, **resize_kwargs)
        # create a empty Reisize object
        self.resize = TRANSFORMS.build({'scale': 0, **self.resize_cfg})


@TRANSFORMS.register_module()
class VideoRandomCrop(RandomCrop):
    def transform(self, results: dict) -> dict:
        """Transform function to randomly crop images.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Randomly cropped results, 'img_shape'
                key in result dict is updated according to crop size.
        """
        if self.padding is not None:
            results['imgs'] = [
                mmcv.impad(img, padding=self.padding, pad_val=self.pad_val) for img in results['imgs']
            ]

        # pad img if needed
        if self.pad_if_needed:
            h_pad = math.ceil(max(0, self.crop_size[0] - results['imgs'][0].shape[0]) / 2)
            w_pad = math.ceil(max(0, self.crop_size[1] - results['imgs'][0].shape[1]) / 2)

            results['imgs'] = [
                mmcv.impad(
                    img,
                    padding=(w_pad, h_pad, w_pad, h_pad),
                    pad_val=self.pad_val,
                    padding_mode=self.padding_mode
                )
                for img in results['imgs']
            ]

        offset_h, offset_w, target_h, target_w = self.rand_crop_params(results['imgs'][0])
        results['imgs'] = [
            mmcv.imcrop(
                img,
                np.array([offset_w, offset_h, offset_w + target_w - 1, offset_h + target_h - 1, ])
            )
            for img in results['imgs']
        ]
        results['img_shape'] = results['imgs'][0].shape
        return results


@TRANSFORMS.register_module()
class VideoRandomFlip(RandomFlip):
    def _flip(self, results: dict) -> None:
        results['imgs'] = [
            mmcv.imflip(img, direction=results['flip_direction']) for img in results['imgs']
        ]


class VideoGaussNoise(albumentations.GaussNoise):
    def apply(self, img, gauss=None, **params):
        from albumentations import functional as F
        if img.shape != gauss.shape:
            h, w = img.shape[:2]
            gauss = cv2.resize(gauss, (w, h), interpolation=cv2.INTER_LINEAR)
        # return F.add_noise(img, gauss)
        return F.gauss_noise(img, gauss=gauss)


@TRANSFORMS.register_module(['VideoAlbumentations', 'VideoAlbu'])
class VideoAlbumentations(BaseTransform):

    def __init__(self, transforms: List[Dict], keymap: Optional[Dict] = None):
        if albumentations is None:
            raise RuntimeError('albumentations is not installed')
        else:
            from albumentations import Compose as albu_Compose

        assert isinstance(transforms, list), 'transforms must be a list.'
        if keymap is not None:
            assert isinstance(keymap, dict), 'keymap must be None or a dict. '

        self.transforms = transforms

        self.aug = albu_Compose(
            [self.albu_builder(t) for t in self.transforms])

        if not keymap:
            self.keymap_to_albu = dict(img='image')
        else:
            self.keymap_to_albu = keymap
        self.keymap_back = {v: k for k, v in self.keymap_to_albu.items()}

    def albu_builder(self, cfg: Dict):
        """Import a module from albumentations.

        It inherits some of :func:`build_from_cfg` logic.
        Args:
            cfg (dict): Config dict. It should at least contain the key "type".
        Returns:
            obj: The constructed object.
        """

        assert isinstance(cfg, dict) and 'type' in cfg, 'each item in ' \
                                                        "transforms must be a dict with keyword 'type'."
        args = cfg.copy()

        obj_type = args.pop('type')
        if obj_type == 'GaussNoise':
            obj_cls = VideoGaussNoise
        elif mmengine.is_str(obj_type):
            obj_cls = getattr(albumentations, obj_type)
        elif inspect.isclass(obj_type):
            obj_cls = obj_type
        else:
            raise TypeError(
                f'type must be a str or valid type, but got {type(obj_type)}')

        if 'transforms' in args:
            args['transforms'] = [
                self.albu_builder(transform)
                for transform in args['transforms']
            ]

        return obj_cls(**args)

    @staticmethod
    def mapper(d, keymap):
        """Dictionary mapper.

        Renames keys according to keymap provided.
        Args:
            d (dict): old dict
            keymap (dict): {'old_key':'new_key'}
        Returns:
            dict: new dict.
        """

        updated_dict = {}
        for k, v in zip(d.keys(), d.values()):
            new_k = keymap.get(k, k)
            updated_dict[new_k] = d[k]
        return updated_dict

    def transform(self, results: Dict) -> Dict:
        """Transform function to perform albumentations transforms.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Transformed results, 'img' and 'img_shape' keys are
                updated in result dict.
        """
        assert 'imgs' in results, 'No `img` field in the input.'

        inputs = {'image': results['imgs'][0]}
        for i, img in enumerate(results['imgs'][1:]):
            inputs[f'image{i + 1}'] = img

        results['imgs'] = list(self.aug(**inputs).values())

        results['img_shape'] = results['imgs'][0].shape[:2]

        return results

    def __repr__(self):
        """Print the basic information of the transform.

        Returns:
            str: Formatted string.
        """
        repr_str = self.__class__.__name__
        repr_str += f'(transforms={repr(self.transforms)})'
        return repr_str


@TRANSFORMS.register_module()
class VideoFormatShape(BaseTransform):
    """Format final imgs shape to the given input_format.

    Required keys:
        - imgs (optional)
        - heatmap_imgs (optional)
        - num_clips
        - clip_len

    Modified Keys:
        - imgs (optional)
        - input_shape (optional)

    Added Keys:
        - heatmap_input_shape (optional)

    Args:
        input_format (str): Define the final data format.
        collapse (bool): To collapse input_format N... to ... (NCTHW to CTHW,
            etc.) if N is 1. Should be set as True when training and testing
            detectors. Defaults to False.
    """

    def __init__(self, input_format: str, collapse: bool = False) -> None:
        self.input_format = input_format
        self.collapse = collapse
        if self.input_format not in [
            'NCTHW', 'NCHW', 'NCHW_Flow', 'NCTHW_Heatmap', 'NPTCHW'
        ]:
            raise ValueError(
                f'The input format {self.input_format} is invalid.')

    def transform(self, results: Dict) -> Dict:
        """Performs the FormatShape formatting.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        if not isinstance(results['imgs'], np.ndarray):
            results['imgs'] = np.array(results['imgs'])

        # [M x H x W x C]
        # M = 1 * N_crops * N_clips * T
        if self.collapse:
            assert results['num_clips'] == 1

        if self.input_format == 'NCTHW':
            if 'imgs' in results:
                imgs = results['imgs']
                num_clips = results['num_clips']
                clip_len = results['clip_len']
                if isinstance(clip_len, dict):
                    clip_len = clip_len['RGB']

                imgs = imgs.reshape((-1, num_clips, clip_len) + imgs.shape[1:])
                # N_crops x N_clips x T x H x W x C
                imgs = np.transpose(imgs, (0, 1, 5, 2, 3, 4))
                # N_crops x N_clips x C x T x H x W
                imgs = imgs.reshape((-1,) + imgs.shape[2:])
                # M' x C x T x H x W
                # M' = N_crops x N_clips
                results['imgs'] = imgs
                results['input_shape'] = imgs.shape

            if 'heatmap_imgs' in results:
                imgs = results['heatmap_imgs']
                num_clips = results['num_clips']
                clip_len = results['clip_len']
                # clip_len must be a dict
                clip_len = clip_len['Pose']

                imgs = imgs.reshape((-1, num_clips, clip_len) + imgs.shape[1:])
                # N_crops x N_clips x T x C x H x W
                imgs = np.transpose(imgs, (0, 1, 3, 2, 4, 5))
                # N_crops x N_clips x C x T x H x W
                imgs = imgs.reshape((-1,) + imgs.shape[2:])
                # M' x C x T x H x W
                # M' = N_crops x N_clips
                results['heatmap_imgs'] = imgs
                results['heatmap_input_shape'] = imgs.shape

        elif self.input_format == 'NCTHW_Heatmap':
            num_clips = results['num_clips']
            clip_len = results['clip_len']
            imgs = results['imgs']

            imgs = imgs.reshape((-1, num_clips, clip_len) + imgs.shape[1:])
            # N_crops x N_clips x T x C x H x W
            imgs = np.transpose(imgs, (0, 1, 3, 2, 4, 5))
            # N_crops x N_clips x C x T x H x W
            imgs = imgs.reshape((-1,) + imgs.shape[2:])
            # M' x C x T x H x W
            # M' = N_crops x N_clips
            results['imgs'] = imgs
            results['input_shape'] = imgs.shape

        elif self.input_format == 'NCHW':
            imgs = results['imgs']
            imgs = np.transpose(imgs, (0, 3, 1, 2))
            # M x C x H x W
            results['imgs'] = imgs
            results['input_shape'] = imgs.shape

        elif self.input_format == 'NCHW_Flow':
            num_imgs = len(results['imgs'])
            assert num_imgs % 2 == 0
            n = num_imgs // 2
            h, w = results['imgs'][0].shape
            x_flow = np.empty((n, h, w), dtype=np.float32)
            y_flow = np.empty((n, h, w), dtype=np.float32)
            for i in range(n):
                x_flow[i] = results['imgs'][2 * i]
                y_flow[i] = results['imgs'][2 * i + 1]
            imgs = np.stack([x_flow, y_flow], axis=-1)

            num_clips = results['num_clips']
            clip_len = results['clip_len']
            imgs = imgs.reshape((-1, num_clips, clip_len) + imgs.shape[1:])
            # N_crops x N_clips x T x H x W x C
            imgs = np.transpose(imgs, (0, 1, 2, 5, 3, 4))
            # N_crops x N_clips x T x C x H x W
            imgs = imgs.reshape((-1, imgs.shape[2] * imgs.shape[3]) +
                                imgs.shape[4:])
            # M' x C' x H x W
            # M' = N_crops x N_clips
            # C' = T x C
            results['imgs'] = imgs
            results['input_shape'] = imgs.shape

        elif self.input_format == 'NPTCHW':
            num_proposals = results['num_proposals']
            num_clips = results['num_clips']
            clip_len = results['clip_len']
            imgs = results['imgs']
            imgs = imgs.reshape((num_proposals, num_clips * clip_len) +
                                imgs.shape[1:])
            # P x M x H x W x C
            # M = N_clips x T
            imgs = np.transpose(imgs, (0, 1, 4, 2, 3))
            # P x M x C x H x W
            results['imgs'] = imgs
            results['input_shape'] = imgs.shape

        if self.collapse:
            assert results['imgs'].shape[0] == 1
            results['imgs'] = results['imgs'].squeeze(0)
            results['input_shape'] = results['imgs'].shape

        return results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f"(input_format='{self.input_format}')"
        return repr_str


@TRANSFORMS.register_module()
class PackVideoInputs(BaseTransform):
    DEFAULT_META_KEYS = (
        'sample_idx', 'video_path',
        'img_shape', 'img_key', 'video_id', 'timestamp',
        'face_data',
        'total_frames', 'avg_fps', 'duration',
        'snippet_stride', 'window_start_frame', 'resize_length', 'window_size', 'offset_frames',  # for sliding window
    )

    def __init__(self,
                 input_key='imgs',
                 collect_keys=[],
                 algorithm_keys=(),
                 meta_keys=DEFAULT_META_KEYS,
                 ):
        self.input_key = input_key
        self.collect_keys = collect_keys
        self.algorithm_keys = algorithm_keys
        self.meta_keys = meta_keys

    @staticmethod
    def format_input(input_):
        if isinstance(input_, list):
            return [PackImageInputs.format_input(item) for item in input_]
        elif isinstance(input_, np.ndarray):
            input_ = to_tensor(input_)
        elif not isinstance(input_, torch.Tensor):
            raise TypeError(f'Unsupported input type {type(input_)}.')
        return input_

    def transform(self, results: dict) -> dict:
        packed_results = dict()

        if self.input_key in results:
            input_ = results[self.input_key]
            packed_results['inputs'] = self.format_input(input_)

        if self.collect_keys is not None:
            for key in self.collect_keys:
                packed_results['inputs'][key] = to_tensor(results[key])

        data_sample = UnifiedDataSample(
            data_type='video',
        )

        # Set default keys
        # cls task
        if 'gt_label' in results:
            data_sample.tasks('video').set_gt_label(results['gt_label'])
            if results.get('masks') is not None:
                data_sample.tasks('video').masks = to_tensor(results['masks'])
            else:
                data_sample.tasks('video').masks = None

        # tad task
        if 'gt_segments' in results and results.get('with_tad'):
            gt_instances = InstanceData()
            gt_instances.segments = to_tensor(results['gt_segments'])
            gt_instances.labels = to_tensor(results['gt_labels'])
            data_sample.tasks('temporal').gt_instances = gt_instances

            if results.get('masks') is not None:
                data_sample.tasks('temporal').masks = to_tensor(results['masks'])
            else:
                data_sample.tasks('temporal').masks = None

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
