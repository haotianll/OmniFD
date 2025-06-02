# Copyright (c) OpenMMLab. All rights reserved.
import copy

import numpy as np
from mmengine.dataset import BaseDataset, force_full_init

from mmpretrain.registry import DATASETS

from collections import defaultdict
from typing import List, Union
from mmengine.dataset import ClassBalancedDataset as _ClassBalancedDataset

@DATASETS.register_module()
class KFoldDataset:
    """A wrapper of dataset for K-Fold cross-validation.

    K-Fold cross-validation divides all the samples in groups of samples,
    called folds, of almost equal sizes. And we use k-1 of folds to do training
    and use the fold left to do validation.

    Args:
        dataset (:obj:`mmengine.dataset.BaseDataset` | dict): The dataset to be
            divided
        fold (int): The fold used to do validation. Defaults to 0.
        num_splits (int): The number of all folds. Defaults to 5.
        test_mode (bool): Use the training dataset or validation dataset.
            Defaults to False.
        seed (int, optional): The seed to shuffle the dataset before splitting.
            If None, not shuffle the dataset. Defaults to None.
    """

    def __init__(self,
                 dataset,
                 fold=0,
                 num_splits=5,
                 test_mode=False,
                 seed=None):
        if isinstance(dataset, dict):
            self.dataset = DATASETS.build(dataset)
            # Init the dataset wrapper lazily according to the dataset setting.
            lazy_init = dataset.get('lazy_init', False)
        elif isinstance(dataset, BaseDataset):
            self.dataset = dataset
        else:
            raise TypeError(f'Unsupported dataset type {type(dataset)}.')

        self._metainfo = getattr(self.dataset, 'metainfo', {})
        self.fold = fold
        self.num_splits = num_splits
        self.test_mode = test_mode
        self.seed = seed

        self._fully_initialized = False
        if not lazy_init:
            self.full_init()

    @property
    def metainfo(self) -> dict:
        """Get the meta information of ``self.dataset``.

        Returns:
            dict: Meta information of the dataset.
        """
        # Prevent `self._metainfo` from being modified by outside.
        return copy.deepcopy(self._metainfo)

    def full_init(self):
        """fully initialize the dataset."""
        if self._fully_initialized:
            return

        self.dataset.full_init()
        ori_len = len(self.dataset)
        indices = list(range(ori_len))
        if self.seed is not None:
            rng = np.random.default_rng(self.seed)
            rng.shuffle(indices)

        test_start = ori_len * self.fold // self.num_splits
        test_end = ori_len * (self.fold + 1) // self.num_splits
        if self.test_mode:
            indices = indices[test_start:test_end]
        else:
            indices = indices[:test_start] + indices[test_end:]

        self._ori_indices = indices
        self.dataset = self.dataset.get_subset(indices)

        self._fully_initialized = True

    @force_full_init
    def _get_ori_dataset_idx(self, idx: int) -> int:
        """Convert global idx to local index.

        Args:
            idx (int): Global index of ``KFoldDataset``.

        Returns:
            int: The original index in the whole dataset.
        """
        return self._ori_indices[idx]

    @force_full_init
    def get_data_info(self, idx: int) -> dict:
        """Get annotation by index.

        Args:
            idx (int): Global index of ``KFoldDataset``.

        Returns:
            dict: The idx-th annotation of the datasets.
        """
        return self.dataset.get_data_info(idx)

    @force_full_init
    def __len__(self):
        return len(self.dataset)

    @force_full_init
    def __getitem__(self, idx):
        return self.dataset[idx]

    @force_full_init
    def get_cat_ids(self, idx):
        return self.dataset.get_cat_ids(idx)

    @force_full_init
    def get_gt_labels(self):
        return self.dataset.get_gt_labels()

    @property
    def CLASSES(self):
        """Return all categories names."""
        return self._metainfo.get('classes', None)

    @property
    def class_to_idx(self):
        """Map mapping class name to class index.

        Returns:
            dict: mapping from class name to class index.
        """

        return {cat: i for i, cat in enumerate(self.CLASSES)}

    def __repr__(self):
        """Print the basic information of the dataset.

        Returns:
            str: Formatted string.
        """
        head = 'Dataset ' + self.__class__.__name__
        body = []
        type_ = 'test' if self.test_mode else 'training'
        body.append(f'Type: \t{type_}')
        body.append(f'Seed: \t{self.seed}')

        def ordinal(n):
            # Copy from https://codegolf.stackexchange.com/a/74047
            suffix = 'tsnrhtdd'[(n // 10 % 10 != 1) * (n % 10 < 4) * n % 10::4]
            return f'{n}{suffix}'

        body.append(
            f'Fold: \t{ordinal(self.fold + 1)} of {self.num_splits}-fold')
        if self._fully_initialized:
            body.append(f'Number of samples: \t{self.__len__()}')
        else:
            body.append("Haven't been initialized")

        if self.CLASSES is not None:
            body.append(f'Number of categories: \t{len(self.CLASSES)}')
        else:
            body.append('The `CLASSES` meta info is not set.')

        body.append(
            f'Original dataset type:\t{self.dataset.__class__.__name__}')

        lines = [head] + [' ' * 4 + line for line in body]
        return '\n'.join(lines)


@DATASETS.register_module()
class FixedClassBalancedDataset(_ClassBalancedDataset):
    def __init__(self,
                 dataset: Union[BaseDataset, dict],
                 oversample_thr: float,
                 lazy_init: bool = False):
        if isinstance(dataset, dict):
            self.dataset = DATASETS.build(dataset)
        elif isinstance(dataset, BaseDataset):
            self.dataset = dataset
        else:
            raise TypeError(
                'elements in datasets sequence should be config or '
                f'`BaseDataset` instance, but got {type(dataset)}')
        self.oversample_thr = oversample_thr
        self._metainfo = self.dataset.metainfo

        self._fully_initialized = False
        if not lazy_init:
            self.full_init()

    def _get_repeat_factors(self, dataset: BaseDataset,
                            repeat_thr: float) -> List[float]:
        """Get repeat factor for each images in the dataset.

        Args:
            dataset (BaseDataset): The dataset.
            repeat_thr (float): The threshold of frequency. If an image
                contains the categories whose frequency below the threshold,
                it would be repeated.

        Returns:
            List[float]: The repeat factors for each images in the dataset.
        """
        # 1. For each category c, compute the fraction # of images
        #   that contain it: f(c)

        repeat_thr = 0.
        category_freq: defaultdict = defaultdict(float)
        num_images = len(dataset)
        for idx in range(num_images):
            cat_ids = set(self.dataset.get_cat_ids(idx))
            for cat_id in cat_ids:
                category_freq[cat_id] += 1
        for k, v in category_freq.items():
            assert v > 0, f'caterogy {k} does not contain any images'
            category_freq[k] = v / num_images
            repeat_thr = max(repeat_thr, category_freq[k])

        # 2. For each category c, compute the category-level repeat factor:
        #    r(c) = max(1, sqrt(t/f(c)))
        category_repeat = {
            # cat_id: max(1.0, math.sqrt(repeat_thr / cat_freq))
            cat_id: max(1.0, repeat_thr / cat_freq)
            for cat_id, cat_freq in category_freq.items()
        }

        # 3. For each image I and its labels L(I), compute the image-level
        # repeat factor:
        #    r(I) = max_{c in L(I)} r(c)
        repeat_factors = []
        for idx in range(num_images):
            cat_ids = set(self.dataset.get_cat_ids(idx))
            if len(cat_ids) != 0:
                repeat_factor = max(
                    {category_repeat[cat_id]
                     for cat_id in cat_ids})
                repeat_factors.append(repeat_factor)
        return repeat_factors