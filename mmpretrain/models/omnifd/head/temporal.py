import copy
import math
from abc import ABCMeta
from typing import Optional, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.config import ConfigDict
from mmengine.model import BaseModule
from mmengine.structures import InstanceData

from mmpretrain.registry import MODELS
from ..base.post_processing import batched_nms, convert_to_seconds
from ..resampler import Attention, FeedForward


class ConvModule(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            conv_cfg=None,
            norm_cfg=None,
            act_cfg=None,  # default to none to remind, act_cfg=dict(type="relu"),
    ):
        super().__init__()
        # norm config
        assert norm_cfg is None or isinstance(norm_cfg, dict)
        self.with_norm = norm_cfg is not None

        # conv config
        conv_cfg_base = dict(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )

        if self.with_norm:
            conv_cfg_base["bias"] = False  # bias is not necessary with a normalization layer

        assert conv_cfg is None or isinstance(conv_cfg, dict)
        if conv_cfg is not None:  # update conv_cfg_base
            conv_cfg_base.update(conv_cfg)

        # build conv layer
        self.conv = nn.Conv1d(**conv_cfg_base)

        # build norm layer
        if self.with_norm:
            norm_cfg = copy.copy(norm_cfg)  # make a copy
            norm_type = norm_cfg["type"]
            norm_cfg.pop("type")
            self.norm_type = norm_type

            if norm_type == "BN":
                self.norm = nn.BatchNorm1d(num_features=out_channels, **norm_cfg)
            elif norm_type == "GN":
                self.norm = nn.GroupNorm(num_channels=out_channels, **norm_cfg)
            elif norm_type == "LN":
                self.norm = nn.LayerNorm(out_channels, eps=1e-6)

        # build activation layer
        assert act_cfg is None or isinstance(act_cfg, dict)
        self.with_act = act_cfg is not None

        if self.with_act:
            act_cfg = copy.copy(act_cfg)  # make a copy
            act_type = act_cfg["type"]
            act_cfg.pop("type")

            if act_type == "relu":
                self.act = nn.ReLU(inplace=True, **act_cfg)
            else:  # other type
                self.act = eval(act_type)(**act_cfg)

        # init weights
        self.apply(self.__init_weights__)

    def __init_weights__(self, module):
        if isinstance(module, nn.Conv1d):
            # use pytorch's default init
            nn.init.kaiming_uniform_(module.weight, a=math.sqrt(5))
            # set nn.Conv1d bias term to 0
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.0)
        elif isinstance(module, (nn.BatchNorm1d, nn.GroupNorm, nn.LayerNorm)):
            nn.init.constant_(module.weight, 1.0)
            nn.init.constant_(module.bias, 0.0)

    def forward(self, x, mask=None):
        x = self.conv(x)

        if mask is not None:  # masking before the norm
            if mask.shape[-1] != x.shape[-1]:
                mask = (
                    F.interpolate(mask.unsqueeze(1).to(x.dtype), size=x.size(-1), mode="nearest")
                    .squeeze(1)
                    .to(mask.dtype)
                )
            x = x * mask.unsqueeze(1).float().detach()  # [B,C,T]

        if self.with_norm:
            if self.norm_type == "LN":
                x = self.norm(x.permute(0, 2, 1)).permute(0, 2, 1)
            else:
                x = self.norm(x)

        if self.with_act:
            x = self.act(x)

        if mask is not None:  # masking the output
            x = x * mask.unsqueeze(1).float().detach()  # [B,C,T]
            return x, mask
        else:
            return x


class Scale(nn.Module):
    def __init__(self, init_value=1.0):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(init_value, dtype=torch.float32), requires_grad=True)

    def forward(self, x):
        return x * self.scale


@MODELS.register_module()
class PointGenerator:
    def __init__(
            self,
            strides,
            regression_range,
            use_offset=False,
    ):
        super().__init__()
        self.strides = strides
        self.regression_range = regression_range
        self.use_offset = use_offset

    def __call__(self, feat_list):
        pts_list = []
        for i, feat in enumerate(feat_list):
            T = feat.shape[-1]

            points = torch.linspace(0, T - 1, T, dtype=torch.float) * self.strides[i]
            reg_range = torch.as_tensor(self.regression_range[i], dtype=torch.float)
            stride = torch.as_tensor(self.strides[i], dtype=torch.float)

            if self.use_offset:
                points += 0.5 * stride

            points = points[:, None]
            reg_range = reg_range[None].repeat(T, 1)
            stride = stride[None].repeat(T, 1)
            pts_list.append(torch.cat((points, reg_range, stride), dim=1).to(feat.device))
        return pts_list


class AnchorFreeHead(nn.Module):
    def __init__(
            self,
            num_classes,
            in_channels,
            feat_channels,
            num_convs=2,
            cls_prior_prob=0.01,
            prior_generator=dict(
                type='PointGenerator',
                strides=[1, 2, 4, 8, 16, 32],
                regression_range=[(0, 4), (4, 8), (8, 16), (16, 32), (32, 64), (64, 10000)],
            ),
            loss=None,
            loss_normalizer=100,
            loss_normalizer_momentum=0.9,
            center_sample='radius',
            center_sample_radius=1.5,
            label_smoothing=0.0,
            reg_loss_weight=1.0,
            loss_weight=1.0,
            filter_similar_gt=True,
    ):
        super(AnchorFreeHead, self).__init__()

        self.num_classes = num_classes
        self.in_channels = in_channels
        self.feat_channels = feat_channels
        self.num_convs = num_convs
        self.cls_prior_prob = cls_prior_prob
        self.label_smoothing = label_smoothing
        self.filter_similar_gt = filter_similar_gt

        self.reg_loss_weight = reg_loss_weight
        self.center_sample = center_sample
        self.center_sample_radius = center_sample_radius
        self.loss_normalizer_momentum = loss_normalizer_momentum
        self.register_buffer("loss_normalizer", torch.tensor(loss_normalizer))

        self.prior_generator = MODELS.build(prior_generator)

        self._init_layers()

        self.cls_loss = MODELS.build(loss.cls_loss)
        self.reg_loss = MODELS.build(loss.reg_loss)

        self.loss_weight = loss_weight

    def _init_layers(self):
        self._init_cls_convs()
        self._init_reg_convs()
        self._init_heads()

    def _init_cls_convs(self):
        self.cls_convs = nn.ModuleList([])
        for i in range(self.num_convs):
            self.cls_convs.append(
                ConvModule(
                    self.in_channels if i == 0 else self.feat_channels,
                    self.feat_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    norm_cfg=dict(type="LN"),
                    act_cfg=dict(type="relu"),
                )
            )

    def _init_reg_convs(self):
        self.reg_convs = nn.ModuleList([])
        for i in range(self.num_convs):
            self.reg_convs.append(
                ConvModule(
                    self.in_channels if i == 0 else self.feat_channels,
                    self.feat_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    norm_cfg=dict(type="LN"),
                    act_cfg=dict(type="relu"),
                )
            )

    def _init_heads(self):
        self.cls_head = nn.Conv1d(self.feat_channels, self.num_classes, kernel_size=3, padding=1)
        self.reg_head = nn.Conv1d(self.feat_channels, 2, kernel_size=3, padding=1)
        self.scale = nn.ModuleList([Scale() for _ in range(len(self.prior_generator.strides))])

        if self.cls_prior_prob > 0:
            bias_value = -(math.log((1 - self.cls_prior_prob) / self.cls_prior_prob))
            nn.init.constant_(self.cls_head.bias, bias_value)

    def forward_train(self, feat_list, mask_list, gt_segments, gt_labels, **kwargs):
        cls_pred = []
        reg_pred = []

        for l, (feat, mask) in enumerate(zip(feat_list, mask_list)):
            cls_feat = feat
            reg_feat = feat

            for i in range(self.num_convs):
                cls_feat, mask = self.cls_convs[i](cls_feat, mask)
                reg_feat, mask = self.reg_convs[i](reg_feat, mask)

            cls_pred.append(self.cls_head(cls_feat))
            reg_pred.append(F.relu(self.scale[l](self.reg_head(reg_feat))))

        points = self.prior_generator(feat_list)

        losses = self.losses(cls_pred, reg_pred, mask_list, points, gt_segments, gt_labels)
        return losses

    def forward_test(self, feat_list, mask_list, **kwargs):
        cls_pred = []
        reg_pred = []

        for l, (feat, mask) in enumerate(zip(feat_list, mask_list)):
            cls_feat = feat
            reg_feat = feat

            for i in range(self.num_convs):
                cls_feat, mask = self.cls_convs[i](cls_feat, mask)
                reg_feat, mask = self.reg_convs[i](reg_feat, mask)

            cls_pred.append(self.cls_head(cls_feat))
            reg_pred.append(F.relu(self.scale[l](self.reg_head(reg_feat))))

        points = self.prior_generator(feat_list)

        proposals, scores = self.get_valid_proposals_scores(points, reg_pred, cls_pred, mask_list)  # list [T,2]
        return proposals, scores

    def get_refined_proposals(self, points, reg_pred):
        points = torch.cat(points, dim=0)
        reg_pred = torch.cat(reg_pred, dim=-1).permute(0, 2, 1).contiguous()

        start = points[:, 0][None] - reg_pred[:, :, 0] * points[:, 3][None]
        end = points[:, 0][None] + reg_pred[:, :, 1] * points[:, 3][None]
        proposals = torch.stack((start, end), dim=-1)
        return proposals

    def get_valid_proposals_scores(self, points, reg_pred, cls_pred, mask_list):
        proposals = self.get_refined_proposals(points, reg_pred)
        scores = torch.cat(cls_pred, dim=-1).permute(0, 2, 1).contiguous().sigmoid()

        masks = torch.cat(mask_list, dim=1)
        new_proposals, new_scores = [], []
        for proposal, score, mask in zip(proposals, scores, masks):
            new_proposals.append(proposal[mask])
            new_scores.append(score[mask])
        return new_proposals, new_scores

    def losses(self, cls_pred, reg_pred, mask_list, points, gt_segments, gt_labels):
        gt_cls, gt_reg = self.prepare_targets(points, gt_segments, gt_labels)

        # positive mask
        gt_cls = torch.stack(gt_cls)
        valid_mask = torch.cat(mask_list, dim=1)
        pos_mask = torch.logical_and((gt_cls.sum(-1) > 0), valid_mask)
        num_pos = pos_mask.sum().item()

        if self.training:
            self.loss_normalizer = self.loss_normalizer_momentum * self.loss_normalizer + (
                    1 - self.loss_normalizer_momentum
            ) * max(num_pos, 1)
            loss_normalizer = self.loss_normalizer
        else:
            loss_normalizer = max(num_pos, 1)

        # 1. classification loss
        cls_pred = [x.permute(0, 2, 1) for x in cls_pred]
        cls_pred = torch.cat(cls_pred, dim=1)[valid_mask]
        gt_target = gt_cls[valid_mask]

        # optional label smoothing
        gt_target *= 1 - self.label_smoothing
        gt_target += self.label_smoothing / (self.num_classes + 1)

        cls_loss = self.cls_loss(cls_pred, gt_target, reduction_override='sum')
        cls_loss /= loss_normalizer

        # 2. regression using IoU/GIoU/DIOU loss (defined on positive samples)
        split_size = [reg.shape[-1] for reg in reg_pred]
        gt_reg = torch.stack(gt_reg).permute(0, 2, 1).contiguous().split(split_size, dim=-1)  # [B,2,T]
        pred_segments = self.get_refined_proposals(points, reg_pred)[pos_mask]
        gt_segments = self.get_refined_proposals(points, gt_reg)[pos_mask]

        if num_pos == 0:
            reg_loss = pred_segments.sum() * 0
        else:
            reg_loss = self.reg_loss(pred_segments, gt_segments, reduction="sum")
            reg_loss /= loss_normalizer

        if self.reg_loss_weight > 0:
            reg_loss_weight = self.reg_loss_weight
        else:
            reg_loss_weight = cls_loss.detach() / max(reg_loss.item(), 0.01)

        return {
            "loss_rpn_cls": cls_loss * self.loss_weight,
            "loss_rpn_reg": reg_loss * reg_loss_weight * self.loss_weight
        }

    @torch.no_grad()
    def prepare_targets(self, points, gt_segments, gt_labels):
        concat_points = torch.cat(points, dim=0)
        num_pts = concat_points.shape[0]
        gt_cls, gt_reg = [], []

        for gt_segment, gt_label in zip(gt_segments, gt_labels):
            num_gts = gt_segment.shape[0]

            # corner case where current sample does not have actions
            if num_gts == 0:
                gt_cls.append(gt_segment.new_full((num_pts, self.num_classes), 0))
                gt_reg.append(gt_segment.new_zeros((num_pts, 2)))
                continue

            # compute the lengths of all segments -> F T x N
            lens = gt_segment[:, 1] - gt_segment[:, 0]
            lens = lens[None, :].repeat(num_pts, 1)

            # compute the distance of every point to each segment boundary
            # auto broadcasting for all reg target-> F T x N x2
            gt_segs = gt_segment[None].expand(num_pts, num_gts, 2)
            left = concat_points[:, 0, None] - gt_segs[:, :, 0]
            right = gt_segs[:, :, 1] - concat_points[:, 0, None]
            reg_targets = torch.stack((left, right), dim=-1)

            if self.center_sample == "radius":
                # center of all segments F T x N
                center_pts = 0.5 * (gt_segs[:, :, 0] + gt_segs[:, :, 1])
                # center sampling based on stride radius
                # compute the new boundaries:
                # concat_points[:, 3] stores the stride
                t_mins = center_pts - concat_points[:, 3, None] * self.center_sample_radius
                t_maxs = center_pts + concat_points[:, 3, None] * self.center_sample_radius
                # prevent t_mins / maxs from over-running the action boundary
                # left: torch.maximum(t_mins, gt_segs[:, :, 0])
                # right: torch.minimum(t_maxs, gt_segs[:, :, 1])
                # F T x N (distance to the new boundary)
                cb_dist_left = concat_points[:, 0, None] - torch.maximum(t_mins, gt_segs[:, :, 0])
                cb_dist_right = torch.minimum(t_maxs, gt_segs[:, :, 1]) - concat_points[:, 0, None]
                # F T x N x 2
                center_seg = torch.stack((cb_dist_left, cb_dist_right), -1)
                # F T x N
                inside_gt_seg_mask = center_seg.min(-1)[0] > 0
            else:
                # inside an gt action
                inside_gt_seg_mask = reg_targets.min(-1)[0] > 0

            # limit the regression range for each location
            max_regress_distance = reg_targets.max(-1)[0]
            # F T x N
            inside_regress_range = torch.logical_and(
                (max_regress_distance >= concat_points[:, 1, None]), (max_regress_distance <= concat_points[:, 2, None])
            )

            # if there are still more than one actions for one moment
            # pick the one with the shortest duration (easiest to regress)
            lens.masked_fill_(inside_gt_seg_mask == 0, float("inf"))
            lens.masked_fill_(inside_regress_range == 0, float("inf"))
            # F T x N -> F T
            min_len, min_len_inds = lens.min(dim=1)

            # corner case: multiple actions with very similar durations (e.g., THUMOS14)
            if self.filter_similar_gt:
                min_len_mask = torch.logical_and((lens <= (min_len[:, None] + 1e-3)), (lens < float("inf")))
            else:
                min_len_mask = lens < float("inf")
            min_len_mask = min_len_mask.to(reg_targets.dtype)

            # cls_targets: F T x C; reg_targets F T x 2
            gt_label_one_hot = F.one_hot(gt_label.long(), self.num_classes).to(reg_targets.dtype)
            cls_targets = min_len_mask @ gt_label_one_hot
            # to prevent multiple GT actions with the same label and boundaries
            cls_targets.clamp_(min=0.0, max=1.0)
            # OK to use min_len_inds
            reg_targets = reg_targets[range(num_pts), min_len_inds]
            # normalization based on stride
            reg_targets /= concat_points[:, 3, None]

            gt_cls.append(cls_targets)
            gt_reg.append(reg_targets)
        return gt_cls, gt_reg


class MaskedMHCA(nn.Module):
    """
    Multi Head Conv Attention with mask

    Add a depthwise convolution within a standard MHA
    The extra conv op can be used to
    (1) encode relative position information (replace position encoding);
    (2) downsample the features if needed;
    (3) match the feature channels

    Note: With current implementation, the downsample feature will be aligned
    to every s+1 time step, where s is the downsampling stride. This allows us
    to easily interpolate the corresponding positional embeddings.

    Modified from https://github.com/karpathy/minGPT/blob/master/mingpt/model.py
    """

    def __init__(
            self,
            n_embd,  # dimension of the output features
            n_head,  # number of heads in multi-head self-attention
            n_qx_stride=1,  # downsampling stride for query and input
            n_kv_stride=1,  # downsampling stride for key and value
            attn_pdrop=0.0,  # dropout rate for the attention map
            proj_pdrop=0.0,  # dropout rate for projection op
    ):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_embd = n_embd
        self.n_head = n_head
        self.n_channels = n_embd // n_head
        self.scale = 1.0 / math.sqrt(self.n_channels)

        # conv/pooling operations
        assert (n_qx_stride == 1) or (n_qx_stride % 2 == 0)
        assert (n_kv_stride == 1) or (n_kv_stride % 2 == 0)
        self.n_qx_stride = n_qx_stride
        self.n_kv_stride = n_kv_stride

        # query conv (depthwise)
        kernel_size = self.n_qx_stride + 1 if self.n_qx_stride > 1 else 3
        stride, padding = self.n_kv_stride, kernel_size // 2
        self.query_conv = ConvModule(
            self.n_embd,
            self.n_embd,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            conv_cfg=dict(groups=n_embd, bias=False),
        )
        self.query_norm = nn.LayerNorm(self.n_embd)

        # key, value conv (depthwise)
        kernel_size = self.n_kv_stride + 1 if self.n_kv_stride > 1 else 3
        stride, padding = self.n_kv_stride, kernel_size // 2
        self.key_conv = ConvModule(
            self.n_embd,
            self.n_embd,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            conv_cfg=dict(groups=n_embd, bias=False),
        )
        self.key_norm = nn.LayerNorm(self.n_embd)

        self.value_conv = ConvModule(
            self.n_embd,
            self.n_embd,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            conv_cfg=dict(groups=n_embd, bias=False),
        )
        self.value_norm = nn.LayerNorm(self.n_embd)

        # key, query, value projections for all heads
        # it is OK to ignore masking, as the mask will be attached on the attention
        self.key = nn.Conv1d(self.n_embd, self.n_embd, 1)
        self.query = nn.Conv1d(self.n_embd, self.n_embd, 1)
        self.value = nn.Conv1d(self.n_embd, self.n_embd, 1)

        # regularization
        self.attn_drop = nn.Dropout(attn_pdrop)
        self.proj_drop = nn.Dropout(proj_pdrop)

        # output projection
        self.proj = nn.Conv1d(self.n_embd, self.n_embd, 1)

    def forward(self, x, mask):
        # x: batch size, feature channel, sequence length,
        # mask: batch size, 1, sequence length (bool)
        B, C, T = x.size()

        # query conv -> (B, nh * hs, T')
        q, qx_mask = self.query_conv(x, mask)
        q = self.query_norm(q.permute(0, 2, 1)).permute(0, 2, 1)
        # key, value conv -> (B, nh * hs, T'')
        k, kv_mask = self.key_conv(x, mask)
        k = self.key_norm(k.permute(0, 2, 1)).permute(0, 2, 1)
        v, _ = self.value_conv(x, mask)
        v = self.value_norm(v.permute(0, 2, 1)).permute(0, 2, 1)

        # projections
        q = self.query(q)
        k = self.key(k)
        v = self.value(v)

        # move head forward to be the batch dim
        # (B, nh * hs, T'/T'') -> (B, nh, T'/T'', hs)
        k = k.view(B, self.n_head, self.n_channels, -1).transpose(2, 3)
        q = q.view(B, self.n_head, self.n_channels, -1).transpose(2, 3)
        v = v.view(B, self.n_head, self.n_channels, -1).transpose(2, 3)

        # self-attention: (B, nh, T', hs) x (B, nh, hs, T'') -> (B, nh, T', T'')
        att = (q * self.scale) @ k.transpose(-2, -1)
        # prevent q from attending to invalid tokens
        att = att.masked_fill(torch.logical_not(kv_mask[:, None, None, :]), float("-inf"))
        # softmax attn
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        # (B, nh, T', T'') x (B, nh, T'', hs) -> (B, nh, T', hs)
        out = att @ (v * kv_mask[:, None, :, None].to(v.dtype))
        # re-assemble all head outputs side by side
        out = out.transpose(2, 3).contiguous().view(B, C, -1)

        # output projection + skip connection
        out = self.proj_drop(self.proj(out)) * qx_mask.unsqueeze(1).to(out.dtype)
        return out, qx_mask


class LocalMaskedMHCA(nn.Module):
    """
    Local Multi Head Conv Attention with mask

    Add a depthwise convolution within a standard MHA
    The extra conv op can be used to
    (1) encode relative position information (replace position encoding);
    (2) downsample the features if needed;
    (3) match the feature channels

    Note: With current implementation, the downsample feature will be aligned
    to every s+1 time step, where s is the downsampling stride. This allows us
    to easily interpolate the corresponding positional embeddings.

    The implementation is fairly tricky, code reference from
    https://github.com/huggingface/transformers/blob/master/src/transformers/models/longformer/modeling_longformer.py
    """

    def __init__(
            self,
            n_embd,  # dimension of the output features
            n_head,  # number of heads in multi-head self-attention
            window_size,  # size of the local attention window
            n_qx_stride=1,  # downsampling stride for query and input
            n_kv_stride=1,  # downsampling stride for key and value
            attn_pdrop=0.0,  # dropout rate for the attention map
            proj_pdrop=0.0,  # dropout rate for projection op
            use_rel_pe=False,  # use relative position encoding
    ):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_embd = n_embd
        self.n_head = n_head
        self.n_channels = n_embd // n_head
        self.scale = 1.0 / math.sqrt(self.n_channels)
        self.window_size = window_size
        self.window_overlap = window_size // 2
        # must use an odd window size
        assert self.window_size > 1 and self.n_head >= 1
        self.use_rel_pe = use_rel_pe

        # conv/pooling operations
        assert (n_qx_stride == 1) or (n_qx_stride % 2 == 0)
        assert (n_kv_stride == 1) or (n_kv_stride % 2 == 0)
        self.n_qx_stride = n_qx_stride
        self.n_kv_stride = n_kv_stride

        # query conv (depthwise)
        kernel_size = self.n_qx_stride + 1 if self.n_qx_stride > 1 else 3
        stride, padding = self.n_kv_stride, kernel_size // 2
        self.query_conv = ConvModule(
            self.n_embd,
            self.n_embd,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            conv_cfg=dict(groups=n_embd, bias=False),
        )
        self.query_norm = nn.LayerNorm(self.n_embd)

        # key, value conv (depthwise)
        kernel_size = self.n_kv_stride + 1 if self.n_kv_stride > 1 else 3
        stride, padding = self.n_kv_stride, kernel_size // 2
        self.key_conv = ConvModule(
            self.n_embd,
            self.n_embd,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            conv_cfg=dict(groups=n_embd, bias=False),
        )
        self.key_norm = nn.LayerNorm(self.n_embd)

        self.value_conv = ConvModule(
            self.n_embd,
            self.n_embd,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            conv_cfg=dict(groups=n_embd, bias=False),
        )
        self.value_norm = nn.LayerNorm(self.n_embd)

        # key, query, value projections for all heads
        # it is OK to ignore masking, as the mask will be attached on the attention
        self.key = nn.Conv1d(self.n_embd, self.n_embd, 1)
        self.query = nn.Conv1d(self.n_embd, self.n_embd, 1)
        self.value = nn.Conv1d(self.n_embd, self.n_embd, 1)

        # regularization
        self.attn_drop = nn.Dropout(attn_pdrop)
        self.proj_drop = nn.Dropout(proj_pdrop)

        # output projection
        self.proj = nn.Conv1d(self.n_embd, self.n_embd, 1)

    @staticmethod
    def _chunk(x, window_overlap):
        """convert into overlapping chunks. Chunk size = 2w, overlap size = w"""
        # x: B x nh, T, hs
        # non-overlapping chunks of size = 2w -> B x nh, T//2w, 2w, hs
        x = x.view(
            x.size(0),
            x.size(1) // (window_overlap * 2),
            window_overlap * 2,
            x.size(2),
        )

        # use `as_strided` to make the chunks overlap with an overlap size = window_overlap
        chunk_size = list(x.size())
        chunk_size[1] = chunk_size[1] * 2 - 1
        chunk_stride = list(x.stride())
        chunk_stride[1] = chunk_stride[1] // 2

        # B x nh, #chunks = T//w - 1, 2w, hs
        return x.as_strided(size=chunk_size, stride=chunk_stride)

    @staticmethod
    def _pad_and_transpose_last_two_dims(x, padding):
        """pads rows and then flips rows and columns"""
        # padding value is not important because it will be overwritten
        x = nn.functional.pad(x, padding)
        x = x.view(*x.size()[:-2], x.size(-1), x.size(-2))
        return x

    @staticmethod
    def _mask_invalid_locations(input_tensor, affected_seq_len):
        beginning_mask_2d = input_tensor.new_ones(affected_seq_len, affected_seq_len + 1).tril().flip(dims=[0])
        beginning_mask = beginning_mask_2d[None, :, None, :]
        ending_mask = beginning_mask.flip(dims=(1, 3))
        beginning_input = input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1]
        beginning_mask = beginning_mask.expand(beginning_input.size())
        # `== 1` converts to bool or uint8
        beginning_input.masked_fill_(beginning_mask == 1, -float("inf"))
        ending_input = input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1):]
        ending_mask = ending_mask.expand(ending_input.size())
        # `== 1` converts to bool or uint8
        ending_input.masked_fill_(ending_mask == 1, -float("inf"))

    @staticmethod
    def _pad_and_diagonalize(x):
        """
        shift every row 1 step right, converting columns into diagonals.
        Example::
              chunked_hidden_states: [ 0.4983,  2.6918, -0.0071,  1.0492,
                                       -1.8348,  0.7672,  0.2986,  0.0285,
                                       -0.7584,  0.4206, -0.0405,  0.1599,
                                       2.0514, -1.1600,  0.5372,  0.2629 ]
              window_overlap = num_rows = 4
             (pad & diagonalize) =>
             [ 0.4983,  2.6918, -0.0071,  1.0492, 0.0000,  0.0000,  0.0000
               0.0000,  -1.8348,  0.7672,  0.2986,  0.0285, 0.0000,  0.0000
               0.0000,  0.0000, -0.7584,  0.4206, -0.0405,  0.1599, 0.0000
               0.0000,  0.0000,  0.0000, 2.0514, -1.1600,  0.5372,  0.2629 ]
        """
        total_num_heads, num_chunks, window_overlap, hidden_dim = x.size()
        # total_num_heads x num_chunks x window_overlap x (hidden_dim+window_overlap+1).
        x = nn.functional.pad(x, (0, window_overlap + 1))
        # total_num_heads x num_chunks x window_overlap*window_overlap+window_overlap
        x = x.view(total_num_heads, num_chunks, -1)
        # total_num_heads x num_chunks x window_overlap*window_overlap
        x = x[:, :, :-window_overlap]
        x = x.view(total_num_heads, num_chunks, window_overlap, window_overlap + hidden_dim)
        x = x[:, :, :, :-1]
        return x

    def _sliding_chunks_query_key_matmul(self, query, key, num_heads, window_overlap):
        """
        Matrix multiplication of query and key tensors using with a sliding window attention pattern. This implementation splits the input into overlapping chunks of size 2w with an overlap of size w (window_overlap)
        """
        # query / key: B*nh, T, hs
        bnh, seq_len, head_dim = query.size()
        batch_size = bnh // num_heads
        assert seq_len % (window_overlap * 2) == 0
        assert query.size() == key.size()

        chunks_count = seq_len // window_overlap - 1

        # B * num_heads, head_dim, #chunks=(T//w - 1), 2w
        chunk_query = self._chunk(query, window_overlap)
        chunk_key = self._chunk(key, window_overlap)

        # matrix multiplication
        # bcxd: batch_size * num_heads x chunks x 2window_overlap x head_dim
        # bcyd: batch_size * num_heads x chunks x 2window_overlap x head_dim
        # bcxy: batch_size * num_heads x chunks x 2window_overlap x 2window_overlap
        diagonal_chunked_attention_scores = torch.einsum("bcxd,bcyd->bcxy", (chunk_query, chunk_key))

        # convert diagonals into columns
        # B * num_heads, #chunks, 2w, 2w+1
        diagonal_chunked_attention_scores = self._pad_and_transpose_last_two_dims(
            diagonal_chunked_attention_scores, padding=(0, 0, 0, 1)
        )

        # allocate space for the overall attention matrix where the chunks are combined. The last dimension
        # has (window_overlap * 2 + 1) columns. The first (window_overlap) columns are the window_overlap lower triangles (attention from a word to
        # window_overlap previous words). The following column is attention score from each word to itself, then
        # followed by window_overlap columns for the upper triangle.
        diagonal_attention_scores = diagonal_chunked_attention_scores.new_empty(
            (batch_size * num_heads, chunks_count + 1, window_overlap, window_overlap * 2 + 1)
        )

        # copy parts from diagonal_chunked_attention_scores into the combined matrix of attentions
        # - copying the main diagonal and the upper triangle
        diagonal_attention_scores[:, :-1, :, window_overlap:] = diagonal_chunked_attention_scores[
            :, :, :window_overlap, : window_overlap + 1
        ]
        diagonal_attention_scores[:, -1, :, window_overlap:] = diagonal_chunked_attention_scores[
            :, -1, window_overlap:, : window_overlap + 1
        ]
        # - copying the lower triangle
        diagonal_attention_scores[:, 1:, :, :window_overlap] = diagonal_chunked_attention_scores[
            :, :, -(window_overlap + 1): -1, window_overlap + 1:
        ]

        diagonal_attention_scores[:, 0, 1:window_overlap, 1:window_overlap] = diagonal_chunked_attention_scores[
            :, 0, : window_overlap - 1, 1 - window_overlap:
        ]

        # separate batch_size and num_heads dimensions again
        diagonal_attention_scores = diagonal_attention_scores.view(
            batch_size, num_heads, seq_len, 2 * window_overlap + 1
        ).transpose(2, 1)

        self._mask_invalid_locations(diagonal_attention_scores, window_overlap)
        return diagonal_attention_scores

    def _sliding_chunks_matmul_attn_probs_value(self, attn_probs, value, num_heads, window_overlap):
        """
        Same as _sliding_chunks_query_key_matmul but for attn_probs and value tensors. Returned tensor will be of the
        same shape as `attn_probs`
        """
        bnh, seq_len, head_dim = value.size()
        batch_size = bnh // num_heads
        assert seq_len % (window_overlap * 2) == 0
        assert attn_probs.size(3) == 2 * window_overlap + 1
        chunks_count = seq_len // window_overlap - 1
        # group batch_size and num_heads dimensions into one, then chunk seq_len into chunks of size 2 window overlap

        chunked_attn_probs = attn_probs.transpose(1, 2).reshape(
            batch_size * num_heads, seq_len // window_overlap, window_overlap, 2 * window_overlap + 1
        )

        # pad seq_len with w at the beginning of the sequence and another window overlap at the end
        padded_value = nn.functional.pad(value, (0, 0, window_overlap, window_overlap), value=-1)

        # chunk padded_value into chunks of size 3 window overlap and an overlap of size window overlap
        chunked_value_size = (batch_size * num_heads, chunks_count + 1, 3 * window_overlap, head_dim)
        chunked_value_stride = padded_value.stride()
        chunked_value_stride = (
            chunked_value_stride[0],
            window_overlap * chunked_value_stride[1],
            chunked_value_stride[1],
            chunked_value_stride[2],
        )
        chunked_value = padded_value.as_strided(size=chunked_value_size, stride=chunked_value_stride)

        chunked_attn_probs = self._pad_and_diagonalize(chunked_attn_probs)

        context = torch.einsum("bcwd,bcdh->bcwh", (chunked_attn_probs, chunked_value))
        return context.view(batch_size, num_heads, seq_len, head_dim)

    def forward(self, x, mask):
        # x: batch size, feature channel, sequence length,
        # mask: batch size, 1, sequence length (bool)
        B, C, T = x.size()

        # step 1: depth convolutions
        # query conv -> (B, nh * hs, T')
        q, qx_mask = self.query_conv(x, mask)
        q = self.query_norm(q.permute(0, 2, 1)).permute(0, 2, 1)
        # key, value conv -> (B, nh * hs, T'')
        k, kv_mask = self.key_conv(x, mask)
        k = self.key_norm(k.permute(0, 2, 1)).permute(0, 2, 1)
        v, _ = self.value_conv(x, mask)
        v = self.value_norm(v.permute(0, 2, 1)).permute(0, 2, 1)

        # step 2: query, key, value transforms & reshape
        # projections
        q = self.query(q)
        k = self.key(k)
        v = self.value(v)
        # (B, nh * hs, T) -> (B, nh, T, hs)
        q = q.view(B, self.n_head, self.n_channels, -1).transpose(2, 3)
        k = k.view(B, self.n_head, self.n_channels, -1).transpose(2, 3)
        v = v.view(B, self.n_head, self.n_channels, -1).transpose(2, 3)
        # view as (B * nh, T, hs)
        q = q.view(B * self.n_head, -1, self.n_channels).contiguous()
        k = k.view(B * self.n_head, -1, self.n_channels).contiguous()
        v = v.view(B * self.n_head, -1, self.n_channels).contiguous()

        # step 3: compute local self-attention with rel pe and masking
        q *= self.scale
        # chunked query key attention -> B, T, nh, 2w+1 = window_size
        att = self._sliding_chunks_query_key_matmul(q, k, self.n_head, self.window_overlap)

        # rel pe
        if self.use_rel_pe:
            att += self.rel_pe
        # kv_mask -> B, T'', 1
        inverse_kv_mask = torch.logical_not(kv_mask[:, None, :, None].view(B, -1, 1))
        # 0 for valid slot, -inf for masked ones
        float_inverse_kv_mask = inverse_kv_mask.type_as(q).masked_fill(inverse_kv_mask, -1e4)
        # compute the diagonal mask (for each local window)
        diagonal_mask = self._sliding_chunks_query_key_matmul(
            float_inverse_kv_mask.new_ones(size=float_inverse_kv_mask.size()),
            float_inverse_kv_mask,
            1,
            self.window_overlap,
        )
        att += diagonal_mask

        # ignore input masking for now
        att = nn.functional.softmax(att, dim=-1)
        # softmax sometimes inserts NaN if all positions are masked, replace them with 0
        att = att.masked_fill(torch.logical_not(kv_mask[:, :, None, None]), 0.0)
        att = self.attn_drop(att)

        # step 4: compute attention value product + output projection
        # chunked attn value product -> B, nh, T, hs
        out = self._sliding_chunks_matmul_attn_probs_value(att, v, self.n_head, self.window_overlap)
        # transpose to B, nh, hs, T -> B, nh*hs, T
        out = out.transpose(2, 3).contiguous().view(B, C, -1)
        # output projection + skip connection
        out = self.proj_drop(self.proj(out)) * qx_mask.unsqueeze(1).to(out.dtype)
        return out, qx_mask


def drop_path(x, drop_prob=0.0, training=False):
    """
    Stochastic Depth per sample.
    """
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    mask = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    mask.floor_()  # binarize
    output = x.div(keep_prob) * mask
    return output


class AffineDropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks) with a per channel scaling factor (and zero init)
    See: https://arxiv.org/pdf/2103.17239.pdf
    """

    def __init__(self, num_dim, drop_prob=0.0, init_scale_value=1e-4, transpose=False):
        super().__init__()
        self.scale = nn.Parameter(init_scale_value * torch.ones((1, num_dim, 1)), requires_grad=True)
        self.drop_prob = drop_prob
        self.transpose = transpose  # if False, the input is B,C,T, otherwise, the input is B,T,C

    def forward(self, x):
        if self.transpose:
            x = x.transpose(1, 2)
            x = drop_path(self.scale * x, self.drop_prob, self.training)
            return x.transpose(1, 2)
        else:
            return drop_path(self.scale * x, self.drop_prob, self.training)

    def __repr__(self):
        return f"{self.__class__.__name__}(drop_prob={self.drop_prob})"


class TransformerBlock(nn.Module):
    """
    Adapted from https://github.com/happyharrycn/actionformer_release/blob/main/libs/modeling/blocks.py#L644

    Originally modified from https://github.com/karpathy/minGPT/blob/master/mingpt/model.py
    """

    def __init__(
            self,
            in_channels,  # dimension of the input features
            n_head,  # number of attention heads
            n_ds_strides=(1, 1),  # downsampling strides for q & x, k & v
            n_out=None,  # output dimension, if None, set to input dim
            n_hidden=None,  # dimension of the hidden layer in MLP
            act_layer=nn.GELU,  # nonlinear activation used in MLP, default GELU
            attn_pdrop=0.0,  # dropout rate for the attention map
            proj_pdrop=0.0,  # dropout rate for the projection / MLP
            path_pdrop=0.0,  # drop path rate
            mha_win_size=-1,  # > 0 to use window mha
    ):
        super().__init__()
        assert len(n_ds_strides) == 2

        # layer norm for order (B C T)
        self.ln1 = nn.LayerNorm(in_channels)
        self.ln2 = nn.LayerNorm(in_channels)

        # specify the attention module
        if mha_win_size > 1:
            self.attn = LocalMaskedMHCA(
                in_channels,
                n_head,
                window_size=mha_win_size,
                n_qx_stride=n_ds_strides[0],
                n_kv_stride=n_ds_strides[1],
                attn_pdrop=attn_pdrop,
                proj_pdrop=proj_pdrop,
            )
        else:
            self.attn = MaskedMHCA(
                in_channels,
                n_head,
                n_qx_stride=n_ds_strides[0],
                n_kv_stride=n_ds_strides[1],
                attn_pdrop=attn_pdrop,
                proj_pdrop=proj_pdrop,
            )

        # input
        if n_ds_strides[0] > 1:
            kernel_size, stride, padding = n_ds_strides[0] + 1, n_ds_strides[0], (n_ds_strides[0] + 1) // 2
            self.pool_skip = nn.MaxPool1d(kernel_size, stride=stride, padding=padding)
        else:
            self.pool_skip = nn.Identity()

        # two layer mlp
        if n_hidden is None:
            n_hidden = 4 * in_channels  # default
        if n_out is None:
            n_out = in_channels
        # ok to use conv1d here with stride=1
        self.mlp = nn.Sequential(
            nn.Conv1d(in_channels, n_hidden, 1),
            act_layer(),
            nn.Dropout(proj_pdrop, inplace=True),
            nn.Conv1d(n_hidden, n_out, 1),
            nn.Dropout(proj_pdrop, inplace=True),
        )

        # drop path
        if path_pdrop > 0.0:
            self.drop_path_attn = AffineDropPath(in_channels, drop_prob=path_pdrop)
            self.drop_path_mlp = AffineDropPath(n_out, drop_prob=path_pdrop)
        else:
            self.drop_path_attn = nn.Identity()
            self.drop_path_mlp = nn.Identity()

    def forward(self, x, mask):
        # pre-LN transformer: https://arxiv.org/pdf/2002.04745.pdf
        out, out_mask = self.attn(self.ln1(x.permute(0, 2, 1)).permute(0, 2, 1), mask)
        out_mask_float = out_mask.to(out.dtype)
        out = self.pool_skip(x) * out_mask_float.unsqueeze(1) + self.drop_path_attn(out)
        # FFN
        out = out + self.drop_path_mlp(
            self.mlp(self.ln2(out.permute(0, 2, 1)).permute(0, 2, 1)) * out_mask_float.unsqueeze(1)
        )
        return out, out_mask


def get_sinusoid_encoding(n_position, d_hid):
    """Sinusoid position encoding table"""

    def get_position_angle_vec(position):
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    # return a tensor of size 1 C T
    return torch.FloatTensor(sinusoid_table).unsqueeze(0).transpose(1, 2)


@MODELS.register_module()
class TemporalProj(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            arch=(1, 1, 5),
            conv_cfg=dict(kernel_size=3, proj_pdrop=0.0),
            norm_cfg=dict(type='LN'),
            attn_cfg=dict(n_head=4, n_mha_win_size=-1),
            path_pdrop=0.1,
            use_abs_pe=False,
            max_seq_len=2304,
            input_pdrop=0.0,
    ):
        super().__init__()
        assert len(arch) == 3

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.arch = arch
        self.kernel_size = conv_cfg["kernel_size"]
        self.proj_pdrop = conv_cfg["proj_pdrop"]
        self.scale_factor = 2  # as default
        self.n_mha_win_size = attn_cfg["n_mha_win_size"]
        self.n_head = attn_cfg["n_head"]
        self.attn_pdrop = 0.0  # as default
        self.path_pdrop = path_pdrop
        self.with_norm = norm_cfg is not None
        self.use_abs_pe = use_abs_pe
        self.max_seq_len = max_seq_len

        self.input_pdrop = nn.Dropout1d(p=input_pdrop) if input_pdrop > 0 else None

        if isinstance(self.n_mha_win_size, int):
            self.mha_win_size = [self.n_mha_win_size] * (1 + arch[-1])
        else:
            assert len(self.n_mha_win_size) == (1 + arch[-1])
            self.mha_win_size = self.n_mha_win_size

        if isinstance(self.in_channels, (list, tuple)):
            assert isinstance(self.out_channels, (list, tuple)) and len(self.in_channels) == len(self.out_channels)
            self.proj = nn.ModuleList([])
            for n_in, n_out in zip(self.in_channels, self.out_channels):
                self.proj.append(
                    ConvModule(
                        n_in,
                        n_out,
                        kernel_size=1,
                        stride=1,
                        padding=0,
                    )
                )
            in_channels = out_channels = sum(self.out_channels)
        else:
            self.proj = None

        # position embedding (1, C, T), rescaled by 1/sqrt(n_embed)
        if self.use_abs_pe:
            pos_embed = get_sinusoid_encoding(self.max_seq_len, out_channels) / (out_channels ** 0.5)
            self.register_buffer("pos_embed", pos_embed, persistent=False)

        # embedding network using convs
        self.embed = nn.ModuleList()
        for i in range(arch[0]):
            self.embed.append(
                ConvModule(
                    in_channels if i == 0 else out_channels,
                    out_channels,
                    kernel_size=self.kernel_size,
                    stride=1,
                    padding=self.kernel_size // 2,
                    norm_cfg=norm_cfg,
                    act_cfg=dict(type="relu"),
                )
            )

        # stem network using (vanilla) transformer
        self.stem = nn.ModuleList()
        for idx in range(arch[1]):
            self.stem.append(
                TransformerBlock(
                    out_channels,
                    self.n_head,
                    n_ds_strides=(1, 1),
                    attn_pdrop=self.attn_pdrop,
                    proj_pdrop=self.proj_pdrop,
                    path_pdrop=self.path_pdrop,
                    mha_win_size=self.mha_win_size[0],
                )
            )

        # main branch using transformer with pooling
        self.branch = TransformerBlock(
            out_channels,
            self.n_head,
            n_ds_strides=(self.scale_factor, self.scale_factor),
            attn_pdrop=self.attn_pdrop,
            proj_pdrop=self.proj_pdrop,
            path_pdrop=self.path_pdrop,
            mha_win_size=self.mha_win_size[1],
        )

        # init weights
        self.apply(self.__init_weights__)

    def __init_weights__(self, module):
        # set nn.Linear/nn.Conv1d bias term to 0
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            if module.bias is not None:
                torch.nn.init.constant_(module.bias, 0.0)

    def forward(self, x, mask):
        # feature projection
        if self.proj is not None:
            x = torch.cat([proj(s, mask)[0] for proj, s in zip(self.proj, x.split(self.in_channels, dim=1))], dim=1)

        # drop out input if needed
        if self.input_pdrop is not None:
            x = self.input_pdrop(x)

        # embedding network
        for idx in range(len(self.embed)):
            x, mask = self.embed[idx](x, mask)

        # training: using fixed length position embeddings
        if self.use_abs_pe and self.training:
            assert x.shape[-1] <= self.max_seq_len, "Reached max length."
            pe = self.pos_embed
            # add pe to x
            x = x + pe[:, :, : x.shape[-1]] * mask.unsqueeze(1).to(x.dtype)

        # inference: re-interpolate position embeddings for over-length sequences
        if self.use_abs_pe and (not self.training):
            if x.shape[-1] >= self.max_seq_len:
                pe = F.interpolate(self.pos_embed, x.shape[-1], mode="linear", align_corners=False)
            else:
                pe = self.pos_embed
            # add pe to x
            x = x + pe[:, :, : x.shape[-1]] * mask.unsqueeze(1).to(x.dtype)

        # stem transformer
        for idx in range(len(self.stem)):
            x, mask = self.stem[idx](x, mask)

        # prep for outputs
        out_feats = (x,)
        out_masks = (mask,)

        # main branch with downsampling
        for idx in range(self.arch[2]):
            x, mask = self.branch(x, mask)
            out_feats += (x,)
            out_masks += (mask,)
        return out_feats, out_masks


@MODELS.register_module()
class TemporalFPN(nn.Module):
    def __init__(
            self,
            in_channels,  # input feature channels, len(in_channels) = #levels
            out_channels,  # output feature channel
            num_levels=6,
            scale_factor=2.0,  # downsampling rate between two fpn levels
            start_level=0,  # start fpn level
            end_level=-1,  # end fpn level
            norm_cfg=dict(type="LN"),  # if no norm, set to none
    ):
        super().__init__()

        self.in_channels = [in_channels] * num_levels
        self.out_channel = out_channels
        self.scale_factor = scale_factor

        self.start_level = start_level
        if end_level == -1:
            self.end_level = len(self.in_channels)
        else:
            self.end_level = end_level
        assert self.end_level <= len(self.in_channels)
        assert (self.start_level >= 0) and (self.start_level < self.end_level)

        if norm_cfg is not None:
            norm_cfg = copy.copy(norm_cfg)  # make a copy
            norm_type = norm_cfg["type"]
            norm_cfg.pop("type")
            self.norm_type = norm_type
        else:
            self.norm_type = None

        self.fpn_norms = nn.ModuleList()
        for i in range(self.start_level, self.end_level):
            # check feat dims
            assert self.in_channels[i] == self.out_channel

            if self.norm_type == "BN":
                fpn_norm = nn.BatchNorm1d(num_features=out_channels, **norm_cfg)
            elif self.norm_type == "GN":
                fpn_norm = nn.GroupNorm(num_channels=out_channels, **norm_cfg)
            elif self.norm_type == "LN":
                fpn_norm = nn.LayerNorm(out_channels, eps=1e-6)
            else:
                assert self.norm_type is None
                fpn_norm = nn.Identity()
            self.fpn_norms.append(fpn_norm)

    def forward(self, inputs, fpn_masks):
        # inputs must be a list / tuple
        assert len(inputs) == len(self.in_channels)
        assert len(fpn_masks) == len(self.in_channels)

        # apply norms, fpn_masks will remain the same with 1x1 convs
        fpn_feats = tuple()
        new_fpn_masks = tuple()
        for i in range(len(self.fpn_norms)):
            x = inputs[i + self.start_level]
            if self.norm_type == "LN":
                x = self.fpn_norms[i](x.permute(0, 2, 1)).permute(0, 2, 1)
            else:
                x = self.fpn_norms[i](x)
            fpn_feats += (x,)
            new_fpn_masks += (fpn_masks[i + self.start_level],)

        return fpn_feats, new_fpn_masks


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

        # assert len(x) == 1  # num_clips = 1

        # reduce:  (B, C, T, H, W) -> (B, C, T)
        x = torch.mean(x[-1], dim=(3, 4))

        # interpolate:  (B, C, T) -> (B, C, window_size)
        if x.shape[2] != window_size:
            x = F.interpolate(x, size=window_size, mode='linear', align_corners=False)

        # apply mask
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
        else:  # feat_len > self.max_seq_len
            max_len = feat_len
            # pad the input to the next divisible size
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

        # x: [ (B, C, T, H, W) ]

        x = self.process_feature(x, masks, self.get_window_size())

        # x: (B, C, T)

        x, masks = self.pad_feature(x, masks, self.max_seq_len, self.max_div_factor)

        # x: (B, C, T)

        if self.projection is not None:
            x, masks = self.projection(x, masks)

        # x: [ (B, C, Tx) ],  Tx = 32,16,8,4,2,1

        if self.neck is not None:
            x, masks = self.neck(x, masks)

        # x: [ (B, C, Tx) ],  Tx = 32,16,8,4,2,1

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

        # x: [ (B, C, Tx) ],  Tx = 32,16,8,4,2,1

        rpn_proposals, rpn_scores = self.rpn_head.forward_test(x, masks, **kwargs)
        pred = rpn_proposals, rpn_scores

        # rpn_proposals: (N, 2)
        # rpn_scores: (N, 2)

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
        # rpn_proposals,  # [B, K, 2]
        # rpn_scores,  # [B, K, num_classes] after sigmoid

        num_classes = rpn_scores[0].shape[-1]

        results = {}
        for i in range(len(metas)):  # processing each video
            segments = rpn_proposals[i].detach().cpu()  # [N,2]
            scores = rpn_scores[i].detach().cpu()  # [N,class]

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

            # if not sliding window, do nms
            # if self.post_cfg.sliding_window == False and self.post_cfg.nms is not None:
            #     segments, scores, labels = batched_nms(segments, scores, labels, **self.post_cfg.nms)
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

    def fuse(self, feat, latents, fuse_block):
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
            # reg_feat = self.fuse(reg_feat, latents, self.fuse_block)

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
            # reg_feat = self.fuse(reg_feat, latents, self.fuse_block)

            cls_pred.append(self.cls_head(cls_feat))
            reg_pred.append(F.relu(self.scale[l](self.reg_head(reg_feat))))

        points = self.prior_generator(feat_list)

        proposals, scores = self.get_valid_proposals_scores(points, reg_pred, cls_pred, mask_list)  # list [T,2]
        return proposals, scores


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
