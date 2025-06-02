from typing import Sequence, Union

import numpy as np
import torch
from mmengine.utils import is_str

from mmpretrain.structures import DataSample

LABEL_TYPE = Union[torch.Tensor, np.ndarray, Sequence, int]
SCORE_TYPE = Union[torch.Tensor, np.ndarray, Sequence]

from mmengine.structures import BaseDataElement, InstanceData, PixelData


def format_label(value: LABEL_TYPE) -> torch.Tensor:
    """Convert various python types to label-format tensor.

    Supported types are: :class:`numpy.ndarray`, :class:`torch.Tensor`,
    :class:`Sequence`, :class:`int`.

    Args:
        value (torch.Tensor | numpy.ndarray | Sequence | int): Label value.

    Returns:
        :obj:`torch.Tensor`: The foramtted label tensor.
    """

    # Handle single number
    if isinstance(value, (torch.Tensor, np.ndarray)) and value.ndim == 0:
        value = int(value.item())

    if isinstance(value, np.ndarray):
        value = torch.from_numpy(value).to(torch.long)
    elif isinstance(value, Sequence) and not is_str(value):
        value = torch.tensor(value).to(torch.long)
    elif isinstance(value, int):
        value = torch.LongTensor([value])
    elif not isinstance(value, torch.Tensor):
        raise TypeError(f'Type {type(value)} is not an available label type.')
    assert value.ndim == 1, \
        f'The dims of value should be 1, but got {value.ndim}.'

    return value


def format_score(value: SCORE_TYPE) -> torch.Tensor:
    """Convert various python types to score-format tensor.

    Supported types are: :class:`numpy.ndarray`, :class:`torch.Tensor`,
    :class:`Sequence`.

    Args:
        value (torch.Tensor | numpy.ndarray | Sequence): Score values.

    Returns:
        :obj:`torch.Tensor`: The foramtted score tensor.
    """

    if isinstance(value, np.ndarray):
        value = torch.from_numpy(value).float()
    elif isinstance(value, Sequence) and not is_str(value):
        value = torch.tensor(value).float()
    elif not isinstance(value, torch.Tensor):
        raise TypeError(f'Type {type(value)} is not an available label type.')
    assert value.ndim == 1, \
        f'The dims of value should be 1, but got {value.ndim}.'

    return value


class ClsDataSample(BaseDataElement):
    """A general data structure interface.

    It's used as the interface between different components.

    The following fields are convention names in MMPretrain, and we will set or
    get these fields in data transforms, models, and metrics if needed. You can
    also set any new fields for your need.

    Meta fields:
        img_shape (Tuple): The shape of the corresponding input image.
        ori_shape (Tuple): The original shape of the corresponding image.
        sample_idx (int): The index of the sample in the dataset.
        num_classes (int): The number of all categories.

    Data fields:
        gt_label (tensor): The ground truth label.
        gt_score (tensor): The ground truth score.
        pred_label (tensor): The predicted label.
        pred_score (tensor): The predicted score.

    """

    def set_gt_label(self, value: LABEL_TYPE) -> 'DataSample':
        """Set ``gt_label``."""
        self.set_field(format_label(value), 'gt_label', dtype=torch.Tensor)
        return self

    def set_gt_score(self, value: SCORE_TYPE) -> 'DataSample':
        """Set ``gt_score``."""
        score = format_score(value)
        self.set_field(score, 'gt_score', dtype=torch.Tensor)
        if hasattr(self, 'num_classes'):
            assert len(score) == self.num_classes, \
                f'The length of score {len(score)} should be ' \
                f'equal to the num_classes {self.num_classes}.'
        else:
            self.set_field(
                name='num_classes', value=len(score), field_type='metainfo')
        return self

    def set_pred_label(self, value: LABEL_TYPE) -> 'DataSample':
        """Set ``pred_label``."""
        self.set_field(format_label(value), 'pred_label', dtype=torch.Tensor)
        return self

    def set_pred_score(self, value: SCORE_TYPE):
        """Set ``pred_label``."""
        score = format_score(value)
        self.set_field(score, 'pred_score', dtype=torch.Tensor)
        if hasattr(self, 'num_classes'):
            assert len(score) == self.num_classes, \
                f'The length of score {len(score)} should be ' \
                f'equal to the num_classes {self.num_classes}.'
        else:
            self.set_field(
                name='num_classes', value=len(score), field_type='metainfo')
        return self

    def __repr__(self) -> str:
        """Represent the object."""

        def dump_items(items, prefix=''):
            return '\n'.join(f'{prefix}{k}: {v}' for k, v in items)

        repr_ = ''
        if len(self._metainfo_fields) > 0:
            repr_ += '\n\nMETA INFORMATION\n'
            repr_ += dump_items(self.metainfo_items(), prefix=' ' * 4)
        if len(self._data_fields) > 0:
            repr_ += '\n\nDATA FIELDS\n'
            repr_ += dump_items(self.items(), prefix=' ' * 4)

        repr_ = f'<{self.__class__.__name__}({repr_}\n\n) at {hex(id(self))}>'
        return repr_

    @property
    def masks(self):
        return getattr(self, '_masks', None)

    @masks.setter
    def masks(self, value):
        self.set_field(value, '_masks')

    @masks.deleter
    def masks(self):
        del self._masks


class SegDataSample(BaseDataElement):
    """A data structure interface of MMSegmentation. They are used as
    interfaces between different components.

    The attributes in ``SegDataSample`` are divided into several parts:

        - ``gt_sem_seg``(PixelData): Ground truth of semantic segmentation.
        - ``pred_sem_seg``(PixelData): Prediction of semantic segmentation.
        - ``seg_logits``(PixelData): Predicted logits of semantic segmentation.

    Examples:
         >>> import torch
         >>> import numpy as np
         >>> from mmengine.structures import PixelData

         >>> data_sample = SegDataSample()
         >>> img_meta = dict(img_shape=(4, 4, 3),
         ...                 pad_shape=(4, 4, 3))
         >>> gt_segmentations = PixelData(metainfo=img_meta)
         >>> gt_segmentations.data = torch.randint(0, 2, (1, 4, 4))
         >>> data_sample.gt_sem_seg = gt_segmentations
         >>> assert 'img_shape' in data_sample.gt_sem_seg.metainfo_keys()
         >>> data_sample.gt_sem_seg.shape
         (4, 4)
         >>> print(data_sample)
        <SegDataSample(

            META INFORMATION

            DATA FIELDS
            gt_sem_seg: <PixelData(

                    META INFORMATION
                    img_shape: (4, 4, 3)
                    pad_shape: (4, 4, 3)

                    DATA FIELDS
                    data: tensor([[[1, 1, 1, 0],
                                 [1, 0, 1, 1],
                                 [1, 1, 1, 1],
                                 [0, 1, 0, 1]]])
                ) at 0x1c2b4156460>
        ) at 0x1c2aae44d60>

        >>> data_sample = SegDataSample()
        >>> gt_sem_seg_data = dict(sem_seg=torch.rand(1, 4, 4))
        >>> gt_sem_seg = PixelData(**gt_sem_seg_data)
        >>> data_sample.gt_sem_seg = gt_sem_seg
        >>> assert 'gt_sem_seg' in data_sample
        >>> assert 'sem_seg' in data_sample.gt_sem_seg
    """

    @property
    def gt_sem_seg(self) -> PixelData:
        return self._gt_sem_seg

    @gt_sem_seg.setter
    def gt_sem_seg(self, value: PixelData) -> None:
        self.set_field(value, '_gt_sem_seg', dtype=PixelData)

    @gt_sem_seg.deleter
    def gt_sem_seg(self) -> None:
        del self._gt_sem_seg

    @property
    def pred_sem_seg(self) -> PixelData:
        return self._pred_sem_seg

    @pred_sem_seg.setter
    def pred_sem_seg(self, value: PixelData) -> None:
        self.set_field(value, '_pred_sem_seg', dtype=PixelData)

    @pred_sem_seg.deleter
    def pred_sem_seg(self) -> None:
        del self._pred_sem_seg

    @property
    def seg_logits(self) -> PixelData:
        return self._seg_logits

    @seg_logits.setter
    def seg_logits(self, value: PixelData) -> None:
        self.set_field(value, '_seg_logits', dtype=PixelData)

    @seg_logits.deleter
    def seg_logits(self) -> None:
        del self._seg_logits


class TadDataSample(BaseDataElement):
    """A data structure interface of MMDetection. They are used as interfaces
    between different components.

    The attributes in ``DetDataSample`` are divided into several parts:

        - ``proposals``(InstanceData): Region proposals used in two-stage
            detectors.
        - ``gt_instances``(InstanceData): Ground truth of instance annotations.
        - ``pred_instances``(InstanceData): Instances of detection predictions.
        - ``pred_track_instances``(InstanceData): Instances of tracking
            predictions.
        - ``ignored_instances``(InstanceData): Instances to be ignored during
            training/testing.
        - ``gt_panoptic_seg``(PixelData): Ground truth of panoptic
            segmentation.
        - ``pred_panoptic_seg``(PixelData): Prediction of panoptic
           segmentation.
        - ``gt_sem_seg``(PixelData): Ground truth of semantic segmentation.
        - ``pred_sem_seg``(PixelData): Prediction of semantic segmentation.

    Examples:
         >>> import torch
         >>> import numpy as np
         >>> from mmengine.structures import InstanceData

         >>> data_sample = TadDataSample()
         >>> img_meta = dict(img_shape=(224, 224))
         >>> gt_instances = InstanceData(metainfo=img_meta)
         >>> gt_instances.segments = torch.rand((5, 2))
         >>> gt_instances.labels = torch.rand((5,))
         >>> data_sample.gt_instances = gt_instances

         >>> pred_instances = InstanceData(metainfo=img_meta)
         >>> pred_instances.segments = torch.rand((5, 2))
         >>> pred_instances.scores = torch.rand((5,))
         >>> data_sample = TadDataSample(pred_instances=pred_instances)
    """

    @property
    def proposals(self) -> InstanceData:
        return self._proposals

    @proposals.setter
    def proposals(self, value: InstanceData):
        self.set_field(value, '_proposals', dtype=InstanceData)

    @proposals.deleter
    def proposals(self):
        del self._proposals

    @property
    def gt_instances(self) -> InstanceData:
        return self._gt_instances

    @gt_instances.setter
    def gt_instances(self, value: InstanceData):
        self.set_field(value, '_gt_instances', dtype=InstanceData)

    @gt_instances.deleter
    def gt_instances(self):
        del self._gt_instances

    @property
    def pred_instances(self) -> InstanceData:
        return self._pred_instances

    @pred_instances.setter
    def pred_instances(self, value: InstanceData):
        self.set_field(value, '_pred_instances', dtype=InstanceData)

    @pred_instances.deleter
    def pred_instances(self):
        del self._pred_instances

    @property
    def masks(self):
        return getattr(self, '_masks', None)

    @masks.setter
    def masks(self, value):
        self.set_field(value, '_masks')

    @masks.deleter
    def masks(self):
        del self._masks


class UnifiedDataSample(BaseDataElement):
    def __init__(self,
                 data_type='image',
                 metainfo=None,
                 **kwargs):
        super().__init__(metainfo=metainfo, **kwargs)

        self.data_type = data_type

        if data_type == 'image':
            self.set_field(ClsDataSample(), 'image')
            self.set_field(SegDataSample(), 'spatial')
        elif data_type == 'video':
            self.set_field(ClsDataSample(), 'video')
            self.set_field(TadDataSample(), 'temporal')
        else:
            raise NotImplementedError(data_type)

    def tasks(self, task_name):
        return getattr(self, task_name)

    def training_tasks(self):
        tasks = []

        if self.data_type == 'image':
            if 'gt_label' in self.tasks('image'):
                tasks.append('image')

            if 'gt_sem_seg' in self.tasks('spatial'):
                tasks.append('spatial')

        if self.data_type == 'video':
            if 'gt_label' in self.tasks('video'):
                tasks.append('video')

            if 'gt_instances' in self.tasks('temporal'):
                tasks.append('temporal')

        return tasks
