from abc import abstractmethod
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import build_norm_layer
from mmengine.model import BaseModule

from mmpretrain.models.necks.gap import GlobalAveragePooling
from mmpretrain.registry import MODELS


class BaseClsHead(BaseModule):
    def __init__(self,
                 loss_module: dict = dict(type='CrossEntropyLoss', loss_weight=1.0),
                 init_cfg: Optional[dict] = None):
        super().__init__(init_cfg=init_cfg)

        if not isinstance(loss_module, nn.Module):
            loss_module = MODELS.build(loss_module)
        self.loss_module = loss_module

    @abstractmethod
    def forward(self, inputs, data_samples=None):
        pass

    def loss(self, cls_score, data_samples=None, target=None, task_name=None, **kwargs):
        if target is None:
            if data_samples is None:
                raise ValueError(data_samples)

            if 'gt_score' in data_samples[0].tasks(task_name):
                target = torch.stack([i.tasks(task_name).gt_score for i in data_samples])
            else:
                target = torch.cat([i.tasks(task_name).gt_label for i in data_samples])

        # compute loss
        losses = dict()
        loss = self.loss_module(cls_score, target, avg_factor=cls_score.size(0), **kwargs)
        losses[f'loss_{task_name}'] = loss
        return losses

    def predict(self, pred, data_samples, task_name=None):
        """Post-process the output of head."""

        pred_scores = F.softmax(pred, dim=1)
        pred_labels = pred_scores.argmax(dim=1, keepdim=True).detach()

        for data_sample, score, label in zip(data_samples, pred_scores, pred_labels):
            sub_data_sample = data_sample.tasks(task_name)
            sub_data_sample.set_pred_score(score).set_pred_label(label)

        return data_samples


@MODELS.register_module()
class ImageClsHead(BaseClsHead):
    def __init__(self,
                 num_classes: int,
                 in_channels: int,
                 hidden_dim: Optional[int] = None,
                 avg_pool=True,
                 fc_norm=False,
                 dropout_ratio=0.,
                 norm_cfg=dict(type='LN', eps=1e-6),
                 init_cfg: dict = dict(type='TruncNormal', layer='Linear', std=0.02),
                 **kwargs):
        super().__init__(init_cfg=init_cfg, **kwargs)
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim

        if self.num_classes <= 0:
            raise ValueError(
                f'num_classes={num_classes} must be a positive integer')

        if avg_pool:
            self.gap = GlobalAveragePooling()
            self.gap_3d = GlobalAveragePooling(dim=3)
        else:
            self.gap = nn.Identity()

        if fc_norm:
            self.fc_norm = build_norm_layer(norm_cfg, in_channels)[1]
        else:
            self.fc_norm = nn.Identity()

        if dropout_ratio != 0.:
            self.dropout = nn.Dropout(p=dropout_ratio)
        else:
            self.dropout = nn.Identity()

        self.fc = nn.Linear(self.in_channels, self.num_classes)

    def pre_logits(self, feats) -> torch.Tensor:
        if isinstance(feats, (tuple, list)):
            feats = feats[-1]

        if feats.ndim == 4:
            feats = self.gap(feats)  # B, C, H, W -> B, C
        elif feats.ndim == 5:
            feats = self.gap_3d(feats)

        feats = self.fc_norm(feats)
        return feats

    def forward(self, feats, data_samples=None) -> torch.Tensor:
        feats, latents = feats

        latents = latents.mean(dim=2)
        feats = latents

        pre_logits = self.fc_norm(feats)
        pre_logits = self.dropout(pre_logits)

        cls_score = self.fc(pre_logits)
        return cls_score


@MODELS.register_module()
class VideoClsHead(BaseClsHead):
    def __init__(self,
                 num_classes: int,
                 in_channels: int,
                 avg_pool=True,
                 fc_norm=False,
                 dropout_ratio=0.,
                 norm_cfg=dict(type='LN', eps=1e-6),
                 init_cfg: dict = dict(type='TruncNormal', layer='Linear', std=0.02),
                 **kwargs):
        super().__init__(init_cfg=init_cfg, **kwargs)
        self.in_channels = in_channels
        self.num_classes = num_classes

        if self.num_classes <= 0:
            raise ValueError(
                f'num_classes={num_classes} must be a positive integer')

        if avg_pool:
            self.gap = GlobalAveragePooling()
            self.gap_3d = GlobalAveragePooling(dim=3)
        else:
            self.gap = nn.Identity()

        if fc_norm:
            self.fc_norm = build_norm_layer(norm_cfg, in_channels)[1]
        else:
            self.fc_norm = nn.Identity()

        if dropout_ratio != 0.:
            self.dropout = nn.Dropout(p=dropout_ratio)
        else:
            self.dropout = nn.Identity()

        self.fc = nn.Linear(self.in_channels, self.num_classes)

    def pre_logits(self, feats) -> torch.Tensor:
        if isinstance(feats, (tuple, list)):
            feats = feats[-1]

        if feats.ndim == 4:
            feats = self.gap(feats)  # B, C, H, W -> B, C
        elif feats.ndim == 5:
            feats = self.gap_3d(feats)

        feats = self.fc_norm(feats)
        return feats

    def forward(self, feats, data_samples=None) -> torch.Tensor:
        feats, latents = feats

        latents = latents.mean(dim=2)
        feats = latents

        pre_logits = self.fc_norm(feats)
        pre_logits = self.dropout(pre_logits)
        cls_score = self.fc(pre_logits)
        return cls_score
