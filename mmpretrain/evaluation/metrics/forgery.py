import os.path as osp
from collections import OrderedDict
from typing import Dict, List, Optional, Sequence, Any
from typing import Union

import mmengine
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from mmengine.dist import is_main_process
from mmengine.evaluator import BaseMetric
from mmengine.logging import MMLogger
from mmengine.utils import mkdir_or_exist
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from torch import Tensor

from mmpretrain.registry import METRICS


def to_tensor(value):
    """Convert value to torch.Tensor."""
    if isinstance(value, np.ndarray):
        value = torch.from_numpy(value)
    elif isinstance(value, Sequence) and not mmengine.is_str(value):
        value = torch.tensor(value)
    elif not isinstance(value, torch.Tensor):
        raise TypeError(f'{type(value)} is not an available argument.')
    return value


@METRICS.register_module()
class UnifiedMetric(BaseMetric):
    """Metrics for MultiTask
    Args:
        task_metrics(dict): a dictionary in the keys are the names of the tasks
            and the values is a list of the metric corresponds to this task
    """

    default_prefix: Optional[str] = ''

    def __init__(self,
                 task_metrics: Dict,
                 collect_device: str = 'cpu',
                 print_copy_paste=False,

                 test_tasks=[],
                 ) -> None:

        super().__init__(collect_device=collect_device)

        self.task_metrics = task_metrics

        self._metrics = {}

        if test_tasks is None or len(test_tasks) == 0:
            test_tasks = self.task_metrics.keys()

        for task_name in test_tasks:
            self._metrics[task_name] = []
            if task_name not in self.task_metrics.keys():
                continue
            for metric in self.task_metrics[task_name]:
                self._metrics[task_name].append(METRICS.build(metric))

        self.print_copy_paste = print_copy_paste

    def process(self, data_batch, data_samples: Sequence):
        """Process one batch of data samples.

        The processed results should be stored in ``self.results``, which will
        be used to compute the metrics when all batches have been processed.
        Args:
            data_batch: A batch of data from the dataloader.
            data_samples (Sequence[dict]): A batch of outputs from the model.
        """
        for task_name in self._metrics.keys():
            sub_data_samples = []
            for data_sample in data_samples:
                sub_data_sample = data_sample[task_name]
                sub_data_samples.append(sub_data_sample)
            for metric in self._metrics[task_name]:
                metric.process(data_batch, sub_data_samples)

    def compute_metrics(self, results: list) -> dict:
        raise NotImplementedError('compute metrics should not be used here directly')

    def evaluate(self, size):
        """Evaluate the model performance of the whole dataset after processing
        all batches.

        Args:
            size (int): Length of the entire validation dataset. When batch
                size > 1, the dataloader may pad some data samples to make
                sure all ranks have the same length of dataset slice. The
                ``collect_results`` function will drop the padded data based on
                this size.
        Returns:
            dict: Evaluation metrics dict on the val dataset. The keys are
            "{task_name}_{metric_name}" , and the values
            are corresponding results.
        """
        metrics = {}
        for task_name in self._metrics:
            for metric in self._metrics[task_name]:
                name = metric.__class__.__name__
                if name == 'UnifiedMetric' or metric.results:
                    results = metric.evaluate(size)
                else:
                    results = {metric.__class__.__name__: 0}
                for key in results:
                    name = f'{task_name}/{key}'
                    if name in results:
                        raise ValueError(f'There are multiple metric results with the same metric name {name}.')
                    metrics[name] = results[key]

        if self.print_copy_paste:
            copy_paste = []
            for key, value in metrics.items():
                copy_paste.append('{:.4f}'.format(value))
            metrics['copy_paste'] = ' '.join(copy_paste)

        return metrics


## 1. Image/Video Classification

@METRICS.register_module()
class ForgeryNetAccuracy(BaseMetric):
    r"""Accuracy Balanced over classes, introduced in ForgeryNet.

    Args:
        collect_device (str): Device name used for collecting results from
            different ranks during distributed training. Must be 'cpu' or
            'gpu'. Defaults to 'cpu'.
        prefix (str, optional): The prefix that will be added in the metric
            names to disambiguate homonymous metrics of different evaluators.
            If prefix is not provided in the argument, self.default_prefix
            will be used instead. Defaults to None.
    """
    default_prefix: Optional[str] = ''

    def __init__(self,
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None) -> None:
        super().__init__(collect_device=collect_device)

    def process(self, data_batch, data_samples: Sequence[dict]):
        """Process one batch of data samples.

        The processed results should be stored in ``self.results``, which will
        be used to compute the metrics when all batches have been processed.

        Args:
            data_batch: A batch of data from the dataloader.
            data_samples (Sequence[dict]): A batch of outputs from the model.
        """

        for data_sample in data_samples:
            result = dict()

            result['pred_label'] = data_sample['pred_label'].cpu()
            result['gt_label'] = data_sample['gt_label'].cpu()

            self.results.append(result)

    def compute_metrics(self, results: List):
        """Compute the metrics from processed results.

        Args:
            results (dict): The processed results of each batch.

        Returns:
            Dict: The computed metrics. The keys are the names of the metrics,
            and the values are corresponding results.
        """
        # NOTICE: don't access `self.results` from the method.

        # concat
        target = torch.cat([res['gt_label'] for res in results])
        pred = torch.cat([res['pred_label'] for res in results])

        metrics = self.calculate(pred, target)

        return metrics

    @staticmethod
    def calculate(
            pred: Union[torch.Tensor, np.ndarray, Sequence],
            target: Union[torch.Tensor, np.ndarray, Sequence],
    ) -> Union[torch.Tensor, List[List[torch.Tensor]]]:
        """Calculate the accuracy.

        Args:
            pred (torch.Tensor | np.ndarray | Sequence): The prediction
                results. It can be labels (N, ), or scores of every
                class (N, C).
            target (torch.Tensor | np.ndarray | Sequence): The target of
                each prediction with shape (N, ).
            thrs (Sequence[float | None]): Predictions with scores under
                the thresholds are considered negative. It's only used
                when ``pred`` is scores. None means no thresholds.
                Defaults to (0., ).
            thrs (Sequence[float]): Predictions with scores under
                the thresholds are considered negative. It's only used
                when ``pred`` is scores. Defaults to (0., ).

        Returns:
            torch.Tensor | List[List[torch.Tensor]]: Accuracy.

            - torch.Tensor: If the ``pred`` is a sequence of label instead of
              score (number of dimensions is 1). Only return a top-1 accuracy
              tensor, and ignore the argument ``topk` and ``thrs``.
            - List[List[torch.Tensor]]: If the ``pred`` is a sequence of score
              (number of dimensions is 2). Return the accuracy on each ``topk``
              and ``thrs``. And the first dim is ``topk``, the second dim is
              ``thrs``.
        """

        pred = to_tensor(pred)
        target = to_tensor(target).to(torch.int64)

        assert pred.size(0) == target.size(0), \
            f"The size of pred ({pred.size(0)}) doesn't match " \
            f'the target ({target.size(0)}).'

        if pred.ndim == 1:
            pred = pred.tolist()
            target = target.tolist()
        else:
            pred = pred.tolist()
            target = target.flatten().tolist()

        mAcc = balanced_accuracy_score(target, pred)
        metrics = {
            'mAcc': 100. * mAcc,
        }
        return metrics


@METRICS.register_module()
class AUC(BaseMetric):
    r"""AUC
    Args:
        collect_device (str): Device name used for collecting results from
            different ranks during distributed training. Must be 'cpu' or
            'gpu'. Defaults to 'cpu'.
        prefix (str, optional): The prefix that will be added in the metric
            names to disambiguate homonymous metrics of different evaluators.
            If prefix is not provided in the argument, self.default_prefix
            will be used instead. Defaults to None.
    """
    default_prefix: Optional[str] = ''

    def __init__(self,
                 neg_index=0,
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None) -> None:
        super().__init__(collect_device=collect_device, prefix=prefix)

        self.neg_index = neg_index

    def process(self, data_batch, data_samples: Sequence[dict]):
        """Process one batch of data samples.

        The processed results should be stored in ``self.results``, which will
        be used to computed the metrics when all batches have been processed.

        Args:
            data_batch: A batch of data from the dataloader.
            data_samples (Sequence[dict]): A batch of outputs from the model.
        """

        for data_sample in data_samples:
            result = dict()

            result['pred_score'] = data_sample['pred_score'].cpu()
            result['gt_label'] = data_sample['gt_label'].cpu()

            result['pred_score'] = 1 - result['pred_score'][self.neg_index]
            result['gt_label'] = torch.tensor([0]) if result['gt_label'] == self.neg_index else torch.tensor([1])

            self.results.append(result)

    def compute_metrics(self, results: List):
        """Compute the metrics from processed results.
        Args:
            results (list): The processed results of each batch.
        Returns:
            Dict: The computed metrics. The keys are the names of the metrics,
            and the values are corresponding results.
        """
        # NOTICE: don't access `self.results` from the method. `self.results`
        # are a list of results from multiple batch, while the input `results`
        # are the collected results.

        target = torch.cat([res['gt_label'] for res in results])
        pred = torch.stack([res['pred_score'] for res in results])

        metrics = self.calculate(pred, target)
        return metrics

    def calculate(
            self,
            pred: Union[torch.Tensor, np.ndarray, Sequence],
            target: Union[torch.Tensor, np.ndarray, Sequence],
    ) -> Dict:
        """Calculate the precision, recall, f1-score and support.
        Args:
            pred (torch.Tensor | np.ndarray | Sequence): The prediction
                results. It can be labels (N, ), or scores of every
                class (N, C).
            target (torch.Tensor | np.ndarray | Sequence): The target of
                each prediction with shape (N, ).
        Returns:
            - float: 100. * AUC
        """

        pred = to_tensor(pred)
        target = to_tensor(target).to(torch.int64)
        assert pred.size(0) == target.size(0), \
            f"The size of pred ({pred.size(0)}) doesn't match " \
            f'the target ({target.size(0)}).'

        if pred.ndim == 1:
            pred_score = pred.tolist()
            target = target.tolist()
        else:
            pred_score = pred.tolist()
            target = target.flatten().tolist()

        metrics = dict()

        auc = roc_auc_score(target, pred_score)
        metrics['AUC'] = 100. * auc

        return metrics


## 2. Spatial Localization


@METRICS.register_module()
class ForgeryNetSpatialMetric(BaseMetric):
    """ForgeryNet evaluation metric.

    Args:
        ignore_index (int): Index that will be ignored in evaluation.
            Default: 255.
        iou_metrics (list[str] | str): Metrics to be calculated, the options
            includes 'mIoU', 'mDice' and 'mFscore'.
        nan_to_num (int, optional): If specified, NaN values will be replaced
            by the numbers defined by the user. Default: None.
        beta (int): Determines the weight of recall in the combined score.
            Default: 1.
        collect_device (str): Device name used for collecting results from
            different ranks during distributed training. Must be 'cpu' or
            'gpu'. Defaults to 'cpu'.
        output_dir (str): The directory for output prediction. Defaults to
            None.
        format_only (bool): Only format result for results commit without
            perform evaluation. It is useful when you want to save the result
            to a specific format and submit it to the test server.
            Defaults to False.
        prefix (str, optional): The prefix that will be added in the metric
            names to disambiguate homonymous metrics of different evaluators.
            If prefix is not provided in the argument, self.default_prefix
            will be used instead. Defaults to None.
    """
    default_prefix: Optional[str] = ''

    def __init__(self,
                 ignore_index: int = 255,
                 metrics: List[str] = ['mIoU'],
                 nan_to_num: Optional[int] = None,
                 beta: int = 1,
                 collect_device: str = 'cpu',
                 output_dir: Optional[str] = None,
                 format_only: bool = False,
                 prefix: Optional[str] = None,
                 print_copypaste=False,
                 **kwargs) -> None:
        super().__init__(collect_device=collect_device, prefix=prefix)

        self.ignore_index = ignore_index
        self.metrics = metrics
        self.nan_to_num = nan_to_num
        self.beta = beta
        self.output_dir = output_dir
        if self.output_dir and is_main_process():
            mkdir_or_exist(self.output_dir)
        self.format_only = format_only
        self.print_copypaste = print_copypaste

    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        """Process one batch of data and data_samples.

        The processed results should be stored in ``self.results``, which will
        be used to compute the metrics when all batches have been processed.

        Args:
            data_batch (dict): A batch of data from the dataloader.
            data_samples (Sequence[dict]): A batch of outputs from the model.

        # pred_label:
        # label: range [0, 255], one class

        """

        for data_sample in data_samples:
            seg_logits = data_sample['seg_logits']['data'].squeeze()

            # format_only always for test dataset without ground truth
            if not self.format_only:
                label = data_sample['gt_sem_seg']['data'].squeeze().to(seg_logits)

                if label.dim() == 0 and torch.min(label) == 255:
                    continue

                self.results.append(
                    self.process_data_samples(seg_logits, label),
                )

            # format_result
            if self.output_dir is not None:
                basename = data_sample['img_path']
                for s in ['/', '\\', '.', ]:
                    basename = basename.replace(s, '_')

                pred = (torch.sigmoid(seg_logits) * 255).cpu().numpy()
                pred = Image.fromarray(pred.astype(np.uint8))
                pred.save(osp.abspath(osp.join(self.output_dir, f'{basename}.png')))

                label = data_sample['gt_sem_seg']['data'].squeeze().cpu().numpy()
                label = Image.fromarray(label.astype(np.uint8))
                label.save(osp.abspath(osp.join(self.output_dir, f'{basename}_gt.png')))

    @staticmethod
    def process_data_samples(pred, label, tau1=[0.1, 0.2], tau2=[0.01, 0.05, 0.1]):
        h, w = label.shape
        pixel_num = torch.tensor(h * w)

        if pred.shape != label.shape:
            pred = F.interpolate(pred.unsqueeze(0).unsqueeze(0), label.shape, mode='bilinear').squeeze(0).squeeze(0)

        pred = torch.sigmoid(pred)
        label = label.float() / 255

        out = []

        for tau in tau1:
            iou = torch.sum(1 - torch.abs((pred >= tau).float() - (label >= tau).float())).cpu()
            out.append(iou)

        for tau in tau2:
            iou_diff = torch.sum((torch.abs(pred - label) <= tau).float()).cpu()
            out.append(iou_diff)

        l1_loss = torch.sum((torch.abs(pred - label))).cpu()
        out.append(l1_loss)

        out.append(pixel_num)
        return tuple(out)

    def compute_metrics(self, results: list) -> Dict[str, float]:
        """Compute the metrics from processed results.

        Args:
            results (list): The processed results of each batch.

        Returns:
            Dict[str, float]: The computed metrics. The keys are the names of
                the metrics, and the values are corresponding results. The key
                mainly includes aAcc, mIoU, mAcc, mDice, mFscore, mPrecision,
                mRecall.
        """
        logger: MMLogger = MMLogger.get_current_instance()
        if self.format_only:
            logger.info(f'results are saved to {osp.dirname(self.output_dir)}')
            return OrderedDict()
        # convert list of tuples to tuple of lists, e.g.
        # [(A_1, B_1, C_1, D_1), ...,  (A_n, B_n, C_n, D_n)] to
        # ([A_1, ..., A_n], ..., [D_1, ..., D_n])
        results = tuple(zip(*results))
        assert len(results) == 7

        pixel_num = sum(results[6])

        metrics = {
            'iou_0.1': 100 * sum(results[0]) / pixel_num,
            'iou_0.2': 100 * sum(results[1]) / pixel_num,
            'iou_diff_0.01': 100 * sum(results[2]) / pixel_num,
            'iou_diff_0.05': 100 * sum(results[3]) / pixel_num,
            'iou_diff_0.1': 100 * sum(results[4]) / pixel_num,
            'l1_loss': sum(results[5]) / pixel_num,
        }

        if self.print_copypaste:
            metrics['copypaste'] = ' '.join(
                [f'{v:.3f}' if i != 5 else f'{v:.5f}' for i, v in enumerate(metrics.values())]
            )

        return metrics

    @staticmethod
    def intersect_and_union(pred_label: torch.tensor, label: torch.tensor,
                            num_classes: int, ignore_index: int):
        """Calculate Intersection and Union.

        Args:
            pred_label (torch.tensor): Prediction segmentation map
                or predict result filename. The shape is (H, W).
            label (torch.tensor): Ground truth segmentation map
                or label filename. The shape is (H, W).
            num_classes (int): Number of categories.
            ignore_index (int): Index that will be ignored in evaluation.

        Returns:
            torch.Tensor: The intersection of prediction and ground truth
                histogram on all classes.
            torch.Tensor: The union of prediction and ground truth histogram on
                all classes.
            torch.Tensor: The prediction histogram on all classes.
            torch.Tensor: The ground truth histogram on all classes.
        """

        mask = (label != ignore_index)
        pred_label = pred_label[mask]
        label = label[mask]

        intersect = pred_label[pred_label == label]
        area_intersect = torch.histc(intersect.float(), bins=(num_classes), min=0, max=num_classes - 1).cpu()
        area_pred_label = torch.histc(pred_label.float(), bins=(num_classes), min=0, max=num_classes - 1).cpu()
        area_label = torch.histc(label.float(), bins=(num_classes), min=0, max=num_classes - 1).cpu()
        area_union = area_pred_label + area_label - area_intersect
        return area_intersect, area_union, area_pred_label, area_label

    @staticmethod
    def total_area_to_metrics(total_area_intersect: np.ndarray,
                              total_area_union: np.ndarray,
                              total_area_pred_label: np.ndarray,
                              total_area_label: np.ndarray,
                              metrics: List[str] = ['mIoU'],
                              nan_to_num: Optional[int] = None,
                              beta: int = 1):
        """Calculate evaluation metrics
        Args:
            total_area_intersect (np.ndarray): The intersection of prediction
                and ground truth histogram on all classes.
            total_area_union (np.ndarray): The union of prediction and ground
                truth histogram on all classes.
            total_area_pred_label (np.ndarray): The prediction histogram on
                all classes.
            total_area_label (np.ndarray): The ground truth histogram on
                all classes.
            metrics (List[str] | str): Metrics to be evaluated, 'mIoU' and
                'mDice'.
            nan_to_num (int, optional): If specified, NaN values will be
                replaced by the numbers defined by the user. Default: None.
            beta (int): Determines the weight of recall in the combined score.
                Default: 1.
        Returns:
            Dict[str, np.ndarray]: per category evaluation metrics,
                shape (num_classes, ).
        """

        def f_score(precision, recall, beta=1):
            """calculate the f-score value.

            Args:
                precision (float | torch.Tensor): The precision value.
                recall (float | torch.Tensor): The recall value.
                beta (int): Determines the weight of recall in the combined
                    score. Default: 1.

            Returns:
                [torch.tensor]: The f-score value.
            """
            score = (1 + beta ** 2) * (precision * recall) / (
                    (beta ** 2 * precision) + recall)
            return score

        if isinstance(metrics, str):
            metrics = [metrics]
        allowed_metrics = ['mIoU', 'mDice', 'mFscore']
        if not set(metrics).issubset(set(allowed_metrics)):
            raise KeyError(f'metrics {metrics} is not supported')

        all_acc = total_area_intersect.sum() / total_area_label.sum()
        ret_metrics = OrderedDict({'aAcc': all_acc})
        for metric in metrics:
            if metric == 'mIoU':
                iou = total_area_intersect / total_area_union
                acc = total_area_intersect / total_area_label
                ret_metrics['IoU'] = iou
                ret_metrics['Acc'] = acc
            elif metric == 'mDice':
                dice = 2 * total_area_intersect / (
                        total_area_pred_label + total_area_label)
                acc = total_area_intersect / total_area_label
                ret_metrics['Dice'] = dice
                ret_metrics['Acc'] = acc
            elif metric == 'mFscore':
                precision = total_area_intersect / total_area_pred_label
                recall = total_area_intersect / total_area_label
                f_value = torch.tensor([
                    f_score(x[0], x[1], beta) for x in zip(precision, recall)
                ])
                ret_metrics['Fscore'] = f_value
                ret_metrics['Precision'] = precision
                ret_metrics['Recall'] = recall

        ret_metrics = {
            metric: value.numpy()
            for metric, value in ret_metrics.items()
        }
        if nan_to_num is not None:
            ret_metrics = OrderedDict({
                metric: np.nan_to_num(metric_value, nan=nan_to_num)
                for metric, metric_value in ret_metrics.items()
            })
        return ret_metrics


# REF: https://github.com/ChenqiKONG/Detect_and_Locate/blob/master/utils/metrics_cross.py
@METRICS.register_module()
class SpatialMetric(BaseMetric):
    """Spatial Localization evaluation metric for FF++ and FMLD datasets.

    Args:
        ignore_index (int): Index that will be ignored in evaluation.
            Default: 255.
        iou_metrics (list[str] | str): Metrics to be calculated, the options
            includes 'mIoU', 'mDice' and 'mFscore'.
        nan_to_num (int, optional): If specified, NaN values will be replaced
            by the numbers defined by the user. Default: None.
        beta (int): Determines the weight of recall in the combined score.
            Default: 1.
        collect_device (str): Device name used for collecting results from
            different ranks during distributed training. Must be 'cpu' or
            'gpu'. Defaults to 'cpu'.
        output_dir (str): The directory for output prediction. Defaults to
            None.
        format_only (bool): Only format result for results commit without
            perform evaluation. It is useful when you want to save the result
            to a specific format and submit it to the test server.
            Defaults to False.
        prefix (str, optional): The prefix that will be added in the metric
            names to disambiguate homonymous metrics of different evaluators.
            If prefix is not provided in the argument, self.default_prefix
            will be used instead. Defaults to None.
    """
    default_prefix: Optional[str] = ''

    def __init__(self,
                 ignore_index: int = 255,
                 metrics: List[str] = ['mIoU'],
                 nan_to_num: Optional[int] = None,
                 beta: int = 1,
                 collect_device: str = 'cpu',
                 output_dir: Optional[str] = None,
                 format_only: bool = False,
                 prefix: Optional[str] = None,
                 print_copypaste=False,
                 **kwargs) -> None:
        super().__init__(collect_device=collect_device, prefix=prefix)

        self.ignore_index = ignore_index
        self.metrics = metrics
        self.nan_to_num = nan_to_num
        self.beta = beta
        self.output_dir = output_dir
        if self.output_dir and is_main_process():
            mkdir_or_exist(self.output_dir)
        self.format_only = format_only
        self.print_copypaste = print_copypaste

    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        """Process one batch of data and data_samples.

        The processed results should be stored in ``self.results``, which will
        be used to compute the metrics when all batches have been processed.

        Args:
            data_batch (dict): A batch of data from the dataloader.
            data_samples (Sequence[dict]): A batch of outputs from the model.

        # pred_label:
        # label: range [0, 255], one class

        """

        for data_sample in data_samples:
            seg_logits = data_sample['seg_logits']['data'].squeeze()

            # format_only always for test dataset without ground truth
            if not self.format_only:
                label = data_sample['gt_sem_seg']['data'].squeeze().to(seg_logits)

                if label.dim() == 0 and torch.min(label) == 255:
                    continue

                self.results.append(
                    self.process_data_samples(seg_logits.cpu(), label.cpu()),
                    # self.intersect_and_union(pred_label, label, num_classes, self.ignore_index)
                )

            # format_result
            if self.output_dir is not None:
                basename = data_sample['img_path']
                for s in ['/', '\\', '.', ]:
                    basename = basename.replace(s, '_')

                pred = (torch.sigmoid(seg_logits) * 255).cpu().numpy()
                pred = Image.fromarray(pred.astype(np.uint8))
                pred.save(osp.abspath(osp.join(self.output_dir, f'{basename}.png')))

                label = data_sample['gt_sem_seg']['data'].squeeze().cpu().numpy()
                label = Image.fromarray(label.astype(np.uint8))
                label.save(osp.abspath(osp.join(self.output_dir, f'{basename}_gt.png')))

    @staticmethod
    def intersect_and_union(pred_label: torch.tensor, label: torch.tensor,
                            num_classes: int, ignore_index: int):
        """Calculate Intersection and Union.

        Args:
            pred_label (torch.tensor): Prediction segmentation map
                or predict result filename. The shape is (H, W).
            label (torch.tensor): Ground truth segmentation map
                or label filename. The shape is (H, W).
            num_classes (int): Number of categories.
            ignore_index (int): Index that will be ignored in evaluation.

        Returns:
            torch.Tensor: The intersection of prediction and ground truth
                histogram on all classes.
            torch.Tensor: The union of prediction and ground truth histogram on
                all classes.
            torch.Tensor: The prediction histogram on all classes.
            torch.Tensor: The ground truth histogram on all classes.
        """

        intersect = pred_label[pred_label == label]
        area_intersect = torch.histc(
            intersect.float(), bins=(num_classes), min=0,
            max=num_classes - 1).cpu()
        area_pred_label = torch.histc(
            pred_label.float(), bins=(num_classes), min=0,
            max=num_classes - 1).cpu()
        area_label = torch.histc(
            label.float(), bins=(num_classes), min=0,
            max=num_classes - 1).cpu()
        area_union = area_pred_label + area_label - area_intersect
        return area_intersect, area_union, area_pred_label, area_label

    @staticmethod
    def process_data_samples(pred, label):
        def calculate(threshold, dist, actual_issame):
            predict_issame = np.less(1 - dist, 1 - threshold)
            tp = np.sum(np.logical_and(predict_issame, actual_issame))
            fp = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))
            tn = np.sum(np.logical_and(np.logical_not(predict_issame), np.logical_not(actual_issame)))
            fn = np.sum(np.logical_and(np.logical_not(predict_issame), actual_issame))
            return tp, fp, tn, fn

        def I_U(pred, gt):
            intersection = (pred * gt).sum()
            union = pred.sum() + gt.sum() - intersection
            return intersection, union

        def PBCA(msk_pred, msk_gt, thre=0.5):
            gt_bool = (msk_gt > 0.5)
            tp, fp, tn, fn = calculate(thre, msk_pred, gt_bool)
            PBCA = (tp + tn) / ((tp + fp + tn + fn) * 1.0)
            return PBCA

        def IINC(msk_pred, msk_gt, thre=0.5):
            pixel_num = np.prod(msk_pred.shape)
            gt_bool = (msk_gt > thre)
            pred_bool = np.less(1 - msk_pred, 1 - thre)
            Mgt_mean = np.mean(gt_bool)
            Mgt_l1 = np.sum(gt_bool) * 1.0
            Mpred_mean = np.mean(pred_bool)
            Mpred_l1 = np.sum(pred_bool) * 1.0
            if Mgt_mean == 0 and Mpred_mean == 0:
                IINC = 0
            elif Mgt_mean == 0 and Mpred_mean != 0:
                _, U = I_U(pred_bool, gt_bool)
                U_norm = U / pixel_num
                IINC = 1 / (3 - U_norm)
            elif Mgt_mean != 0 and Mpred_mean == 0:
                _, U = I_U(pred_bool, gt_bool)
                U_norm = U / pixel_num
                IINC = 1 / (3 - U_norm)
            else:
                I, U = I_U(pred_bool, gt_bool)
                U_norm = U / pixel_num
                IINC = (2 - I / Mpred_l1 - I / Mgt_l1) * (1 / (3 - U_norm))
            return IINC

        h, w = label.shape

        if pred.shape != label.shape:
            pred = F.interpolate(pred.unsqueeze(0).unsqueeze(0), label.shape, mode='bilinear').squeeze(0).squeeze(0)

        pred = torch.sigmoid(pred).numpy()
        label = (label > 0).int().numpy()

        pbca = PBCA(pred, label)
        iinc = IINC(pred, label)
        iou = I_U(pred, label)
        iou = iou[0] / iou[1]

        out = [pbca, iinc, iou]
        return tuple(out)

    def compute_metrics(self, results: list) -> Dict[str, float]:
        """Compute the metrics from processed results.

        Args:
            results (list): The processed results of each batch.

        Returns:
            Dict[str, float]: The computed metrics. The keys are the names of
                the metrics, and the values are corresponding results. The key
                mainly includes aAcc, mIoU, mAcc, mDice, mFscore, mPrecision,
                mRecall.
        """
        logger: MMLogger = MMLogger.get_current_instance()
        if self.format_only:
            logger.info(f'results are saved to {osp.dirname(self.output_dir)}')
            return OrderedDict()
        # convert list of tuples to tuple of lists, e.g.
        # [(A_1, B_1, C_1, D_1), ...,  (A_n, B_n, C_n, D_n)] to
        # ([A_1, ..., A_n], ..., [D_1, ..., D_n])
        results = tuple(zip(*results))

        num = len(results[-1])

        metrics = {
            'pbca': 100 * sum(results[0]) / num,
            'iinc': 100 * sum(results[1]) / num,
            'iou': 100 * sum(results[2]) / num
        }

        return metrics

    @staticmethod
    def intersect_and_union(pred_label: torch.tensor, label: torch.tensor,
                            num_classes: int, ignore_index: int):
        """Calculate Intersection and Union.

        Args:
            pred_label (torch.tensor): Prediction segmentation map
                or predict result filename. The shape is (H, W).
            label (torch.tensor): Ground truth segmentation map
                or label filename. The shape is (H, W).
            num_classes (int): Number of categories.
            ignore_index (int): Index that will be ignored in evaluation.

        Returns:
            torch.Tensor: The intersection of prediction and ground truth
                histogram on all classes.
            torch.Tensor: The union of prediction and ground truth histogram on
                all classes.
            torch.Tensor: The prediction histogram on all classes.
            torch.Tensor: The ground truth histogram on all classes.
        """

        mask = (label != ignore_index)
        pred_label = pred_label[mask]
        label = label[mask]

        intersect = pred_label[pred_label == label]
        area_intersect = torch.histc(intersect.float(), bins=(num_classes), min=0, max=num_classes - 1).cpu()
        area_pred_label = torch.histc(pred_label.float(), bins=(num_classes), min=0, max=num_classes - 1).cpu()
        area_label = torch.histc(label.float(), bins=(num_classes), min=0, max=num_classes - 1).cpu()
        area_union = area_pred_label + area_label - area_intersect
        return area_intersect, area_union, area_pred_label, area_label

    @staticmethod
    def total_area_to_metrics(total_area_intersect: np.ndarray,
                              total_area_union: np.ndarray,
                              total_area_pred_label: np.ndarray,
                              total_area_label: np.ndarray,
                              metrics: List[str] = ['mIoU'],
                              nan_to_num: Optional[int] = None,
                              beta: int = 1):
        """Calculate evaluation metrics
        Args:
            total_area_intersect (np.ndarray): The intersection of prediction
                and ground truth histogram on all classes.
            total_area_union (np.ndarray): The union of prediction and ground
                truth histogram on all classes.
            total_area_pred_label (np.ndarray): The prediction histogram on
                all classes.
            total_area_label (np.ndarray): The ground truth histogram on
                all classes.
            metrics (List[str] | str): Metrics to be evaluated, 'mIoU' and
                'mDice'.
            nan_to_num (int, optional): If specified, NaN values will be
                replaced by the numbers defined by the user. Default: None.
            beta (int): Determines the weight of recall in the combined score.
                Default: 1.
        Returns:
            Dict[str, np.ndarray]: per category evaluation metrics,
                shape (num_classes, ).
        """

        def f_score(precision, recall, beta=1):
            """calculate the f-score value.

            Args:
                precision (float | torch.Tensor): The precision value.
                recall (float | torch.Tensor): The recall value.
                beta (int): Determines the weight of recall in the combined
                    score. Default: 1.

            Returns:
                [torch.tensor]: The f-score value.
            """
            score = (1 + beta ** 2) * (precision * recall) / (
                    (beta ** 2 * precision) + recall)
            return score

        if isinstance(metrics, str):
            metrics = [metrics]
        allowed_metrics = ['mIoU', 'mDice', 'mFscore']
        if not set(metrics).issubset(set(allowed_metrics)):
            raise KeyError(f'metrics {metrics} is not supported')

        all_acc = total_area_intersect.sum() / total_area_label.sum()
        ret_metrics = OrderedDict({'aAcc': all_acc})
        for metric in metrics:
            if metric == 'mIoU':
                iou = total_area_intersect / total_area_union
                acc = total_area_intersect / total_area_label
                ret_metrics['IoU'] = iou
                ret_metrics['Acc'] = acc
            elif metric == 'mDice':
                dice = 2 * total_area_intersect / (
                        total_area_pred_label + total_area_label)
                acc = total_area_intersect / total_area_label
                ret_metrics['Dice'] = dice
                ret_metrics['Acc'] = acc
            elif metric == 'mFscore':
                precision = total_area_intersect / total_area_pred_label
                recall = total_area_intersect / total_area_label
                f_value = torch.tensor([
                    f_score(x[0], x[1], beta) for x in zip(precision, recall)
                ])
                ret_metrics['Fscore'] = f_value
                ret_metrics['Precision'] = precision
                ret_metrics['Recall'] = recall

        ret_metrics = {
            metric: value.numpy()
            for metric, value in ret_metrics.items()
        }
        if nan_to_num is not None:
            ret_metrics = OrderedDict({
                metric: np.nan_to_num(metric_value, nan=nan_to_num)
                for metric, metric_value in ret_metrics.items()
            })
        return ret_metrics


## 3. Temporal Localization

# REF: https://github.com/ControlNet/LAV-DF/blob/master/metrics.py

class Metadata:
    file: str
    n_fakes: int
    fake_periods: List[List[int]]
    duration: float
    original: Optional[str]
    modify_video: bool
    modify_audio: bool
    split: str
    video_frames: int
    audio_channels: int
    audio_frames: int


def iou_1d(proposal, target) -> Tensor:
    """
    Calculate 1D IOU for N proposals with L labels.

    Args:
        proposal (:class:`~torch.Tensor` | :class:`~numpy.ndarray`): The predicted array with [M, 2]. First column is
            beginning, second column is end.
        target (:class:`~torch.Tensor` | :class:`~numpy.ndarray`): The label array with [N, 2]. First column is
            beginning, second column is end.

    Returns:
        :class:`~torch.Tensor`: The iou result with [M, N].
    """
    if type(proposal) is np.ndarray:
        proposal = torch.from_numpy(proposal)

    if type(target) is np.ndarray:
        target = torch.from_numpy(target)

    proposal_begin = proposal[:, 0].unsqueeze(0).T
    proposal_end = proposal[:, 1].unsqueeze(0).T
    target_begin = target[:, 0]
    target_end = target[:, 1]

    inner_begin = torch.maximum(proposal_begin, target_begin)
    inner_end = torch.minimum(proposal_end, target_end)
    outer_begin = torch.minimum(proposal_begin, target_begin)
    outer_end = torch.maximum(proposal_end, target_end)

    inter = torch.clamp(inner_end - inner_begin, min=0.)
    union = outer_end - outer_begin
    return inter / union


class AP:
    """
    Average Precision

    The mean precision in Precision-Recall curve.
    """

    def __init__(self,
                 iou_thresholds=[0.5, 0.75, 0.95],
                 ap_iou_thresholds=[0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95],
                 tqdm_pos: int = 1):
        super().__init__()
        self.iou_thresholds = iou_thresholds
        self.ap_iou_thresholds = ap_iou_thresholds
        self.tqdm_pos = tqdm_pos
        self.n_labels = 0

        for v in iou_thresholds:
            if v not in ap_iou_thresholds:
                raise ValueError(f'{iou_thresholds} {ap_iou_thresholds}')

    def __call__(self, results: dict) -> dict:

        ap_list = []
        metrics = dict()

        for iou_threshold in self.ap_iou_thresholds:
            values = []
            self.n_labels = 0

            for (labels, proposals) in results:
                proposals = torch.tensor(proposals['segments'])
                labels = torch.tensor(labels['segments'])

                values.append(AP.get_values(iou_threshold, proposals, labels, 1.))
                self.n_labels += len(labels)

            # sort proposals
            values = torch.cat(values)
            ind = values[:, 0].sort(stable=True, descending=True).indices
            values = values[ind]

            # accumulate to calculate precision and recall
            curve = self.calculate_curve(values)
            ap = self.calculate_ap(curve)

            ap = ap.item()
            ap_list.append(ap)
            if iou_threshold in self.iou_thresholds:
                metrics[f'AP_{iou_threshold:.2f}'] = 100. * ap

        metrics['mAP'] = 100. * sum(ap_list) / len(ap_list)
        return metrics

    def calculate_curve(self, values):
        is_TP = values[:, 1]
        acc_TP = torch.cumsum(is_TP, dim=0)
        precision = acc_TP / (torch.arange(len(is_TP)) + 1)
        recall = acc_TP / self.n_labels
        curve = torch.stack([recall, precision]).T
        curve = torch.cat([torch.tensor([[1., 0.]]), torch.flip(curve, dims=(0,))])
        return curve

    @staticmethod
    def calculate_ap(curve):
        x, y = curve.T
        y_max = y.cummax(dim=0).values
        x_diff = x.diff().abs()
        ap = (x_diff * y_max[:-1]).sum()
        return ap

    @staticmethod
    def get_values(
            iou_threshold: float,
            proposals: Tensor,
            labels: Tensor,
            fps: float,
    ) -> Tensor:
        n_labels = len(labels)
        n_proposals = len(proposals)
        if n_labels > 0:
            ious = iou_1d(proposals[:, 1:] / fps, labels)
        else:
            ious = torch.zeros((n_proposals, 0))

        # values: (confidence, is_TP) rows
        n_labels = ious.shape[1]
        detected = torch.full((n_labels,), False)
        confidence = proposals[:, 0]
        potential_TP = ious > iou_threshold

        tp_indexes = []

        for i in range(n_labels):
            potential_TP_index = potential_TP[:, i].nonzero()
            for (j,) in potential_TP_index:
                if j not in tp_indexes:
                    tp_indexes.append(j)
                    break

        is_TP = torch.zeros(n_proposals, dtype=torch.bool)
        if len(tp_indexes) > 0:
            tp_indexes = torch.stack(tp_indexes)
            is_TP[tp_indexes] = True
        values = torch.column_stack([confidence, is_TP])
        return values


class AR:
    """
    Average Recall

    Args:
        n_proposals_list: Number of proposals. 100 for AR@100.
        iou_thresholds: IOU threshold samples for the curve. Default: [0.5:0.05:0.95]

    """

    def __init__(self,
                 n_proposals_list: Union[List[int], int] = 100,
                 iou_thresholds: List[float] = None):
        super().__init__()
        if iou_thresholds is None:
            iou_thresholds = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
        self.n_proposals_list = n_proposals_list if type(n_proposals_list) is list else [n_proposals_list]
        self.n_proposals_list = torch.tensor(self.n_proposals_list)
        self.iou_thresholds = iou_thresholds

    def __call__(self, results) -> dict:
        metrics = dict()

        values = []
        for (labels, proposals) in results:
            proposals = torch.tensor(proposals['segments'])
            labels = torch.tensor(labels['segments'])
            values.append(self.get_values(self.iou_thresholds, proposals, labels, 1.))

        values = torch.stack(values, dim=0)  # shape: (n_metadata, n_iou_thresholds, n_proposal_thresholds, 2)

        values_sum = values.sum(dim=0)

        TP = values_sum[:, :, 0]
        FN = values_sum[:, :, 1]
        recall = TP / (TP + FN)  # (n_iou_thresholds, n_proposal_thresholds)
        for i, n_proposals in enumerate(self.n_proposals_list):
            metrics[f'AR_{int(n_proposals.item())}'] = 100. * recall[:, i].mean().item()

        return metrics

    def get_values(
            self,
            iou_thresholds: List[float],
            proposals: Tensor,
            labels: Tensor,
            fps: float,
    ):
        n_proposals_list = self.n_proposals_list
        max_proposals = max(n_proposals_list)

        proposals = proposals[:max_proposals]

        if proposals.shape[0] < max_proposals:
            proposals = torch.cat([proposals, torch.zeros(max_proposals - proposals.shape[0], 3)])

        n_labels = len(labels)

        if n_labels > 0:
            ious = iou_1d(proposals[:, 1:] / fps, labels)
        else:
            ious = torch.zeros((max_proposals, 0))

        # values: matrix of (TP, FN), shapes (n_iou_thresholds, n_proposal_thresholds, 2)
        iou_max = ious.cummax(0).values[n_proposals_list - 1]  # shape (n_iou_thresholds, n_labels)
        iou_max = iou_max[None]

        iou_thresholds = torch.tensor(iou_thresholds)[:, None, None]
        TP = (iou_max > iou_thresholds).sum(-1)
        FN = n_labels - TP
        values = torch.stack([TP, FN], dim=-1)

        return values


@METRICS.register_module()
class TemporalMetrics(BaseMetric):
    default_prefix: Optional[str] = ''

    def __init__(
            self,
            tiou_thresholds=[0.5],
            n_proposals=[10],
            collect_device: str = 'cpu',
            **kwargs
    ):
        super().__init__(collect_device=collect_device)

        self.tiou_thresholds = tiou_thresholds
        self.n_proposals = n_proposals

        self.ap = AP(tiou_thresholds)
        self.ar = AR(n_proposals)

    # REF: https://github.com/open-mmlab/mmdetection/blob/cfd5d3a985b0249de009b67d04f37263e11cdf3d/mmdet/evaluation/metrics/coco_metric.py#L346
    def process(self, data_batch: Any, data_samples: Sequence[dict]) -> None:
        """Process one batch of data samples and predictions. The processed
        results should be stored in ``self.results``, which will be used to
        compute the metrics when all batches have been processed.

        Args:
            data_batch (dict): A batch of data from the dataloader.
            data_samples (Sequence[dict]): A batch of data samples that
                contain annotations and predictions.
        """

        for data_sample in data_samples:
            pred = data_sample['pred_instances']

            result = torch.cat((pred['scores'].reshape(-1, 1), pred['segments']), dim=1)
            result = dict(segments=result.cpu().numpy())

            gt = data_sample['gt_instances']
            ann = dict(segments=gt['segments'].cpu().numpy())
            self.results.append((ann, result))

    def compute_metrics(self, results: List):
        """Compute the metrics from processed results.

        Args:
            results (dict): The processed results of each batch.

        Returns:
            Dict: The computed metrics. The keys are the names of the metrics,
            and the values are corresponding results.
        """
        # NOTICE: don't access `self.results` from the method.
        metrics = self.calculate(results)
        return metrics

    def calculate(self, results):
        metrics = dict()

        AP = self.ap(results)
        AR = self.ar(results)

        metrics.update(AP)
        metrics.update(AR)
        return metrics
