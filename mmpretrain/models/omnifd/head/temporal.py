import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import ABCMeta
from mmengine.config import ConfigDict
from mmengine.model import BaseModule
from mmengine.structures import InstanceData
from typing import Optional, List

from mmpretrain.registry import MODELS
from .temporal_helper import AnchorFreeHead
from ..base.post_processing import batched_nms, convert_to_seconds
from ..module import Attention, FeedForward


class FuseBlock(nn.Module):
    def __init__(
            self,
            *,
            dim,
            depth=1,
            dim_head=64,
            heads=8,
            num_latents=64,
            ff_mult=4,
    ):
        super().__init__()

        self.dim = dim
        self.num_latents = num_latents

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        Attention(dim=dim, dim_head=dim_head, heads=heads),
                        FeedForward(dim=dim, mult=ff_mult),
                    ]
                )
            )

        self.norm = nn.LayerNorm(dim)

    def forward(self, x, latents):
        for attn, ff in self.layers:
            latents = attn(x, latents) + latents
            latents = ff(latents) + latents
        return self.norm(latents)


@MODELS.register_module()
class TemporalRPNHead(AnchorFreeHead):
    def __init__(self,
                 *args,
                 latent_channels=512,
                 **kwargs):
        self.latent_channels = latent_channels

        super().__init__(*args, **kwargs)

        self.sample_proj = nn.Sequential(
            nn.Linear(self.latent_channels, self.feat_channels),
            nn.LayerNorm(self.feat_channels),
        )
        self.fuse_block = FuseBlock(dim=self.feat_channels)

    def fuse(self, feat, latents, fuse_block=None):
        if fuse_block is None:
            return feat

        feat = feat.unsqueeze(dim=1).transpose(2, 3)

        latents = latents.unsqueeze(dim=1)

        feat = fuse_block(latents, feat)
        feat = feat.transpose(2, 3).squeeze(1)
        return feat

    def forward_train(self, feat_list, mask_list, gt_segments, gt_labels, **kwargs):
        feat_list, latents = feat_list

        cls_pred = []
        reg_pred = []

        latents = self.sample_proj(latents.transpose(1, 2))

        for l, (feat, mask) in enumerate(zip(feat_list, mask_list)):
            cls_feat = feat
            reg_feat = feat

            for i in range(self.num_convs):
                cls_feat, mask = self.cls_convs[i](cls_feat, mask)
                reg_feat, mask = self.reg_convs[i](reg_feat, mask)

            cls_feat = self.fuse(cls_feat, latents, self.fuse_block)
            reg_feat = self.fuse(reg_feat, latents)

            cls_pred.append(self.cls_head(cls_feat))
            reg_pred.append(F.relu(self.scale[l](self.reg_head(reg_feat))))

        points = self.prior_generator(feat_list)

        losses = self.losses(cls_pred, reg_pred, mask_list, points, gt_segments, gt_labels)
        return losses

    def forward_test(self, feat_list, mask_list, **kwargs):
        feat_list, latents = feat_list

        cls_pred = []
        reg_pred = []

        latents = self.sample_proj(latents.transpose(1, 2))

        for l, (feat, mask) in enumerate(zip(feat_list, mask_list)):
            cls_feat = feat
            reg_feat = feat

            for i in range(self.num_convs):
                cls_feat, mask = self.cls_convs[i](cls_feat, mask)
                reg_feat, mask = self.reg_convs[i](reg_feat, mask)

            cls_feat = self.fuse(cls_feat, latents, self.fuse_block)
            reg_feat = self.fuse(reg_feat, latents)

            cls_pred.append(self.cls_head(cls_feat))
            reg_pred.append(F.relu(self.scale[l](self.reg_head(reg_feat))))

        points = self.prior_generator(feat_list)

        proposals, scores = self.get_valid_proposals_scores(points, reg_pred, cls_pred, mask_list)  # list [T,2]
        return proposals, scores


@MODELS.register_module()
class TemporalHead(BaseModule, metaclass=ABCMeta):
    def __init__(
            self,
            rpn_head=None,
            proj=None,
            neck=None,
            post_cfg=dict(
                nms=dict(sigma=0.7),
            ),
            window_size=None,
            window_size_test=None,
            init_cfg: Optional[dict] = None):
        super().__init__(init_cfg=init_cfg)

        if proj is not None and not isinstance(proj, nn.Module):
            proj = MODELS.build(proj)
        if neck is not None and not isinstance(neck, nn.Module):
            neck = MODELS.build(neck)
        if rpn_head is not None and not isinstance(rpn_head, nn.Module):
            rpn_head = MODELS.build(rpn_head)

        self.projection = proj
        self.neck = neck
        self.rpn_head = rpn_head

        if post_cfg is not None:
            if isinstance(post_cfg, dict):
                post_cfg = ConfigDict(post_cfg)
            self.post_cfg = post_cfg

        self.window_size = window_size
        self.window_size_test = window_size if window_size_test is None else window_size_test

        n_mha_win_size = self.projection.n_mha_win_size
        if isinstance(n_mha_win_size, int):
            self.mha_win_size = [n_mha_win_size] * (1 + proj.arch[-1])
        else:
            assert len(n_mha_win_size) == (1 + proj.arch[-1])
            self.mha_win_size = n_mha_win_size

        self.max_seq_len = self.window_size

        max_div_factor = 1
        for s, w in zip(rpn_head.prior_generator.strides, self.mha_win_size):
            stride = s * (w // 2) * 2 if w > 1 else s
            assert (self.max_seq_len % stride == 0), \
                f"max_seq_len {self.max_seq_len} must be divisible by fpn stride and window size {stride}"
            if max_div_factor < stride:
                max_div_factor = stride
        self.max_div_factor = max_div_factor

    @staticmethod
    def process_feature(x, masks, window_size):
        if not isinstance(x, (tuple, list)):
            x = [x]

        x = torch.mean(x[-1], dim=(3, 4))

        if x.shape[2] != window_size:
            x = F.interpolate(x, size=window_size, mode='linear', align_corners=False)

        if masks is not None and x.dim() == 3:
            x = x * masks.unsqueeze(1).detach().float()
        return x

    def get_window_size(self):
        if self.training:
            return self.window_size
        else:
            return self.window_size_test

    @staticmethod
    def pad_feature(inputs, masks, max_seq_len, max_div_factor):
        feat_len = inputs.shape[-1]
        if feat_len == max_seq_len:
            return inputs, masks
        elif feat_len < max_seq_len:
            max_len = max_seq_len
        else:
            max_len = feat_len
            stride = max_div_factor
            max_len = (max_len + (stride - 1)) // stride * stride

        padding_size = [0, max_len - feat_len]
        inputs = F.pad(inputs, padding_size, value=0)
        pad_masks = torch.zeros((inputs.shape[0], max_len), device=masks.device).bool()
        pad_masks[:, :feat_len] = masks
        return inputs, pad_masks

    def forward(self,
                x: torch.Tensor,
                data_samples: Optional[List] = None,
                **kwargs):
        x, latents = x

        if data_samples is None:
            print('masks is not found in data_samples')
            masks = torch.ones(1, self.get_window_size()).int()
        else:
            masks = torch.vstack([i.tasks('temporal').masks for i in data_samples])

        x = self.process_feature(x, masks, self.get_window_size())

        x, masks = self.pad_feature(x, masks, self.max_seq_len, self.max_div_factor)

        if self.projection is not None:
            x, masks = self.projection(x, masks)

        if self.neck is not None:
            x, masks = self.neck(x, masks)

        x = x, latents
        return x, masks

    def loss(self, inputs, data_samples=None, task_name=None, **kwargs):
        x, masks = inputs

        losses = dict()

        gt_segments = [i.tasks('temporal').gt_instances.segments for i in data_samples]
        gt_labels = [i.tasks('temporal').gt_instances.labels for i in data_samples]

        loss = self.rpn_head.forward_train(
            x,
            masks,
            gt_segments=gt_segments,
            gt_labels=gt_labels,
            **kwargs,
        )
        losses.update(loss)

        return losses

    def predict(self, inputs, data_samples, task_name=None, **kwargs):
        x, masks = inputs

        metas = [i.to_dict() for i in data_samples]

        rpn_proposals, rpn_scores = self.rpn_head.forward_test(x, masks, **kwargs)
        pred = rpn_proposals, rpn_scores

        results = self.post_processing(pred, metas, **kwargs)

        for i in range(len(data_samples)):
            name = data_samples[i].get('video_path')

            segments = []
            labels = []
            scores = []

            for x in results[name]:
                segments.append(x['segment'])
                labels.append(x['label'])
                scores.append(x['score'])

            data = InstanceData()
            data.segments = torch.tensor(segments)
            data.labels = torch.tensor(labels)
            data.scores = torch.tensor(scores)

            data_samples[i].tasks('temporal').pred_instances = data

        return data_samples

    def predict_flops(self, inputs, data_samples, task_name=None, **kwargs):
        x, masks = inputs
        return self.rpn_head.forward_test(x, masks, **kwargs)

    @torch.no_grad()
    def post_processing(self, predictions, metas):
        rpn_proposals, rpn_scores = predictions

        num_classes = rpn_scores[0].shape[-1]

        results = {}
        for i in range(len(metas)):
            segments = rpn_proposals[i].detach().cpu()
            scores = rpn_scores[i].detach().cpu()

            if num_classes == 1:
                scores = scores.squeeze(-1)
                labels = torch.zeros(scores.shape[0]).contiguous()
            else:
                pre_nms_thresh = getattr(self.post_cfg, 'pre_nms_thresh', 0.001)
                pre_nms_topk = getattr(self.post_cfg, 'pre_nms_topk', 2000)

                pred_prob = scores.flatten()  # [N*class]

                # Apply filtering to make NMS faster following detectron2
                # 1. Keep seg with confidence score > a threshold
                keep_idxs1 = pred_prob > pre_nms_thresh
                pred_prob = pred_prob[keep_idxs1]
                topk_idxs = keep_idxs1.nonzero(as_tuple=True)[0]

                # 2. Keep top k top scoring boxes only
                num_topk = min(pre_nms_topk, topk_idxs.size(0))
                pred_prob, idxs = pred_prob.sort(descending=True)
                pred_prob = pred_prob[:num_topk].clone()
                topk_idxs = topk_idxs[idxs[:num_topk]].clone()

                # 3. gather predicted proposals
                pt_idxs = torch.div(topk_idxs, num_classes, rounding_mode='floor')
                cls_idxs = torch.fmod(topk_idxs, num_classes)

                segments = segments[pt_idxs]
                scores = pred_prob
                labels = cls_idxs

            segments, scores, labels = batched_nms(segments, scores, labels, **self.post_cfg.nms)

            video_id = metas[i]['video_path']

            # convert segments to seconds
            segments = convert_to_seconds(segments, metas[i])

            results_per_video = []
            for segment, label, score in zip(segments, labels, scores):
                # convert to python scalars
                results_per_video.append(
                    dict(
                        segment=[round(seg.item(), 2) for seg in segment],
                        label=label.item(),
                        score=round(score.item(), 4),
                    )
                )

            if video_id in results.keys():
                results[video_id].extend(results_per_video)
            else:
                results[video_id] = results_per_video

        return results
