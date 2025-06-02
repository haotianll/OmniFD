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
            act_cfg=None,
    ):
        super().__init__()
        assert norm_cfg is None or isinstance(norm_cfg, dict)
        self.with_norm = norm_cfg is not None

        conv_cfg_base = dict(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )

        if self.with_norm:
            conv_cfg_base["bias"] = False

        assert conv_cfg is None or isinstance(conv_cfg, dict)
        if conv_cfg is not None:
            conv_cfg_base.update(conv_cfg)

        self.conv = nn.Conv1d(**conv_cfg_base)

        if self.with_norm:
            norm_cfg = copy.copy(norm_cfg)
            norm_type = norm_cfg["type"]
            norm_cfg.pop("type")
            self.norm_type = norm_type

            if norm_type == "BN":
                self.norm = nn.BatchNorm1d(num_features=out_channels, **norm_cfg)
            elif norm_type == "GN":
                self.norm = nn.GroupNorm(num_channels=out_channels, **norm_cfg)
            elif norm_type == "LN":
                self.norm = nn.LayerNorm(out_channels, eps=1e-6)

        assert act_cfg is None or isinstance(act_cfg, dict)
        self.with_act = act_cfg is not None

        if self.with_act:
            act_cfg = copy.copy(act_cfg)
            act_type = act_cfg["type"]
            act_cfg.pop("type")

            if act_type == "relu":
                self.act = nn.ReLU(inplace=True, **act_cfg)
            else:
                self.act = eval(act_type)(**act_cfg)

        self.apply(self.__init_weights__)

    def __init_weights__(self, module):
        if isinstance(module, nn.Conv1d):
            nn.init.kaiming_uniform_(module.weight, a=math.sqrt(5))
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.0)
        elif isinstance(module, (nn.BatchNorm1d, nn.GroupNorm, nn.LayerNorm)):
            nn.init.constant_(module.weight, 1.0)
            nn.init.constant_(module.bias, 0.0)

    def forward(self, x, mask=None):
        x = self.conv(x)

        if mask is not None:
            if mask.shape[-1] != x.shape[-1]:
                mask = (
                    F.interpolate(mask.unsqueeze(1).to(x.dtype), size=x.size(-1), mode="nearest")
                    .squeeze(1)
                    .to(mask.dtype)
                )
            x = x * mask.unsqueeze(1).float().detach()

        if self.with_norm:
            if self.norm_type == "LN":
                x = self.norm(x.permute(0, 2, 1)).permute(0, 2, 1)
            else:
                x = self.norm(x)

        if self.with_act:
            x = self.act(x)

        if mask is not None:
            x = x * mask.unsqueeze(1).float().detach()
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
        self.scale_factor = 2
        self.n_mha_win_size = attn_cfg["n_mha_win_size"]
        self.n_head = attn_cfg["n_head"]
        self.attn_pdrop = 0.0
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

        if self.use_abs_pe:
            pos_embed = get_sinusoid_encoding(self.max_seq_len, out_channels) / (out_channels ** 0.5)
            self.register_buffer("pos_embed", pos_embed, persistent=False)

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

        self.branch = TransformerBlock(
            out_channels,
            self.n_head,
            n_ds_strides=(self.scale_factor, self.scale_factor),
            attn_pdrop=self.attn_pdrop,
            proj_pdrop=self.proj_pdrop,
            path_pdrop=self.path_pdrop,
            mha_win_size=self.mha_win_size[1],
        )

        self.apply(self.__init_weights__)

    def __init_weights__(self, module):
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            if module.bias is not None:
                torch.nn.init.constant_(module.bias, 0.0)

    def forward(self, x, mask):
        if self.proj is not None:
            x = torch.cat([proj(s, mask)[0] for proj, s in zip(self.proj, x.split(self.in_channels, dim=1))], dim=1)

        if self.input_pdrop is not None:
            x = self.input_pdrop(x)

        for idx in range(len(self.embed)):
            x, mask = self.embed[idx](x, mask)

        if self.use_abs_pe and self.training:
            assert x.shape[-1] <= self.max_seq_len, "Reached max length."
            pe = self.pos_embed
            x = x + pe[:, :, : x.shape[-1]] * mask.unsqueeze(1).to(x.dtype)

        if self.use_abs_pe and (not self.training):
            if x.shape[-1] >= self.max_seq_len:
                pe = F.interpolate(self.pos_embed, x.shape[-1], mode="linear", align_corners=False)
            else:
                pe = self.pos_embed

            x = x + pe[:, :, : x.shape[-1]] * mask.unsqueeze(1).to(x.dtype)

        for idx in range(len(self.stem)):
            x, mask = self.stem[idx](x, mask)

        out_feats = (x,)
        out_masks = (mask,)

        for idx in range(self.arch[2]):
            x, mask = self.branch(x, mask)
            out_feats += (x,)
            out_masks += (mask,)
        return out_feats, out_masks


@MODELS.register_module()
class TemporalFPN(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            num_levels=6,
            scale_factor=2.0,
            start_level=0,
            end_level=-1,
            norm_cfg=dict(type="LN"),
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
            norm_cfg = copy.copy(norm_cfg)
            norm_type = norm_cfg["type"]
            norm_cfg.pop("type")
            self.norm_type = norm_type
        else:
            self.norm_type = None

        self.fpn_norms = nn.ModuleList()
        for i in range(self.start_level, self.end_level):
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
        assert len(inputs) == len(self.in_channels)
        assert len(fpn_masks) == len(self.in_channels)

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

        cls_pred = [x.permute(0, 2, 1) for x in cls_pred]
        cls_pred = torch.cat(cls_pred, dim=1)[valid_mask]
        gt_target = gt_cls[valid_mask]

        gt_target *= 1 - self.label_smoothing
        gt_target += self.label_smoothing / (self.num_classes + 1)

        cls_loss = self.cls_loss(cls_pred, gt_target, reduction_override='sum')
        cls_loss /= loss_normalizer

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

            if num_gts == 0:
                gt_cls.append(gt_segment.new_full((num_pts, self.num_classes), 0))
                gt_reg.append(gt_segment.new_zeros((num_pts, 2)))
                continue

            lens = gt_segment[:, 1] - gt_segment[:, 0]
            lens = lens[None, :].repeat(num_pts, 1)

            gt_segs = gt_segment[None].expand(num_pts, num_gts, 2)
            left = concat_points[:, 0, None] - gt_segs[:, :, 0]
            right = gt_segs[:, :, 1] - concat_points[:, 0, None]
            reg_targets = torch.stack((left, right), dim=-1)

            if self.center_sample == "radius":
                center_pts = 0.5 * (gt_segs[:, :, 0] + gt_segs[:, :, 1])
                t_mins = center_pts - concat_points[:, 3, None] * self.center_sample_radius
                t_maxs = center_pts + concat_points[:, 3, None] * self.center_sample_radius
                cb_dist_left = concat_points[:, 0, None] - torch.maximum(t_mins, gt_segs[:, :, 0])
                cb_dist_right = torch.minimum(t_maxs, gt_segs[:, :, 1]) - concat_points[:, 0, None]
                center_seg = torch.stack((cb_dist_left, cb_dist_right), -1)
                inside_gt_seg_mask = center_seg.min(-1)[0] > 0
            else:
                inside_gt_seg_mask = reg_targets.min(-1)[0] > 0

            max_regress_distance = reg_targets.max(-1)[0]

            inside_regress_range = torch.logical_and(
                (max_regress_distance >= concat_points[:, 1, None]), (max_regress_distance <= concat_points[:, 2, None])
            )

            lens.masked_fill_(inside_gt_seg_mask == 0, float("inf"))
            lens.masked_fill_(inside_regress_range == 0, float("inf"))

            min_len, min_len_inds = lens.min(dim=1)

            if self.filter_similar_gt:
                min_len_mask = torch.logical_and((lens <= (min_len[:, None] + 1e-3)), (lens < float("inf")))
            else:
                min_len_mask = lens < float("inf")
            min_len_mask = min_len_mask.to(reg_targets.dtype)

            gt_label_one_hot = F.one_hot(gt_label.long(), self.num_classes).to(reg_targets.dtype)
            cls_targets = min_len_mask @ gt_label_one_hot

            cls_targets.clamp_(min=0.0, max=1.0)
            reg_targets = reg_targets[range(num_pts), min_len_inds]
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
            n_embd,
            n_head,
            n_qx_stride=1,
            n_kv_stride=1,
            attn_pdrop=0.0,
            proj_pdrop=0.0,
    ):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_embd = n_embd
        self.n_head = n_head
        self.n_channels = n_embd // n_head
        self.scale = 1.0 / math.sqrt(self.n_channels)

        assert (n_qx_stride == 1) or (n_qx_stride % 2 == 0)
        assert (n_kv_stride == 1) or (n_kv_stride % 2 == 0)
        self.n_qx_stride = n_qx_stride
        self.n_kv_stride = n_kv_stride

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

        self.key = nn.Conv1d(self.n_embd, self.n_embd, 1)
        self.query = nn.Conv1d(self.n_embd, self.n_embd, 1)
        self.value = nn.Conv1d(self.n_embd, self.n_embd, 1)

        self.attn_drop = nn.Dropout(attn_pdrop)
        self.proj_drop = nn.Dropout(proj_pdrop)

        self.proj = nn.Conv1d(self.n_embd, self.n_embd, 1)

    def forward(self, x, mask):
        B, C, T = x.size()

        q, qx_mask = self.query_conv(x, mask)
        q = self.query_norm(q.permute(0, 2, 1)).permute(0, 2, 1)
        k, kv_mask = self.key_conv(x, mask)
        k = self.key_norm(k.permute(0, 2, 1)).permute(0, 2, 1)
        v, _ = self.value_conv(x, mask)
        v = self.value_norm(v.permute(0, 2, 1)).permute(0, 2, 1)

        q = self.query(q)
        k = self.key(k)
        v = self.value(v)

        k = k.view(B, self.n_head, self.n_channels, -1).transpose(2, 3)
        q = q.view(B, self.n_head, self.n_channels, -1).transpose(2, 3)
        v = v.view(B, self.n_head, self.n_channels, -1).transpose(2, 3)

        att = (q * self.scale) @ k.transpose(-2, -1)
        att = att.masked_fill(torch.logical_not(kv_mask[:, None, None, :]), float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)

        out = att @ (v * kv_mask[:, None, :, None].to(v.dtype))
        out = out.transpose(2, 3).contiguous().view(B, C, -1)

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
            n_embd,
            n_head,
            window_size,
            n_qx_stride=1,
            n_kv_stride=1,
            attn_pdrop=0.0,
            proj_pdrop=0.0,
            use_rel_pe=False,
    ):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_embd = n_embd
        self.n_head = n_head
        self.n_channels = n_embd // n_head
        self.scale = 1.0 / math.sqrt(self.n_channels)
        self.window_size = window_size
        self.window_overlap = window_size // 2

        assert self.window_size > 1 and self.n_head >= 1
        self.use_rel_pe = use_rel_pe

        assert (n_qx_stride == 1) or (n_qx_stride % 2 == 0)
        assert (n_kv_stride == 1) or (n_kv_stride % 2 == 0)
        self.n_qx_stride = n_qx_stride
        self.n_kv_stride = n_kv_stride

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

        self.key = nn.Conv1d(self.n_embd, self.n_embd, 1)
        self.query = nn.Conv1d(self.n_embd, self.n_embd, 1)
        self.value = nn.Conv1d(self.n_embd, self.n_embd, 1)

        self.attn_drop = nn.Dropout(attn_pdrop)
        self.proj_drop = nn.Dropout(proj_pdrop)

        self.proj = nn.Conv1d(self.n_embd, self.n_embd, 1)

    @staticmethod
    def _chunk(x, window_overlap):
        x = x.view(
            x.size(0),
            x.size(1) // (window_overlap * 2),
            window_overlap * 2,
            x.size(2),
        )

        chunk_size = list(x.size())
        chunk_size[1] = chunk_size[1] * 2 - 1
        chunk_stride = list(x.stride())
        chunk_stride[1] = chunk_stride[1] // 2

        return x.as_strided(size=chunk_size, stride=chunk_stride)

    @staticmethod
    def _pad_and_transpose_last_two_dims(x, padding):
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

        beginning_input.masked_fill_(beginning_mask == 1, -float("inf"))
        ending_input = input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1):]
        ending_mask = ending_mask.expand(ending_input.size())

        ending_input.masked_fill_(ending_mask == 1, -float("inf"))

    @staticmethod
    def _pad_and_diagonalize(x):
        total_num_heads, num_chunks, window_overlap, hidden_dim = x.size()
        x = nn.functional.pad(x, (0, window_overlap + 1))
        x = x.view(total_num_heads, num_chunks, -1)
        x = x[:, :, :-window_overlap]
        x = x.view(total_num_heads, num_chunks, window_overlap, window_overlap + hidden_dim)
        x = x[:, :, :, :-1]
        return x

    def _sliding_chunks_query_key_matmul(self, query, key, num_heads, window_overlap):
        bnh, seq_len, head_dim = query.size()
        batch_size = bnh // num_heads
        assert seq_len % (window_overlap * 2) == 0
        assert query.size() == key.size()

        chunks_count = seq_len // window_overlap - 1

        chunk_query = self._chunk(query, window_overlap)
        chunk_key = self._chunk(key, window_overlap)

        diagonal_chunked_attention_scores = torch.einsum("bcxd,bcyd->bcxy", (chunk_query, chunk_key))

        diagonal_chunked_attention_scores = self._pad_and_transpose_last_two_dims(
            diagonal_chunked_attention_scores, padding=(0, 0, 0, 1)
        )

        diagonal_attention_scores = diagonal_chunked_attention_scores.new_empty(
            (batch_size * num_heads, chunks_count + 1, window_overlap, window_overlap * 2 + 1)
        )

        diagonal_attention_scores[:, :-1, :, window_overlap:] = diagonal_chunked_attention_scores[
            :, :, :window_overlap, : window_overlap + 1
        ]
        diagonal_attention_scores[:, -1, :, window_overlap:] = diagonal_chunked_attention_scores[
            :, -1, window_overlap:, : window_overlap + 1
        ]

        diagonal_attention_scores[:, 1:, :, :window_overlap] = diagonal_chunked_attention_scores[
            :, :, -(window_overlap + 1): -1, window_overlap + 1:
        ]

        diagonal_attention_scores[:, 0, 1:window_overlap, 1:window_overlap] = diagonal_chunked_attention_scores[
            :, 0, : window_overlap - 1, 1 - window_overlap:
        ]

        diagonal_attention_scores = diagonal_attention_scores.view(
            batch_size, num_heads, seq_len, 2 * window_overlap + 1
        ).transpose(2, 1)

        self._mask_invalid_locations(diagonal_attention_scores, window_overlap)
        return diagonal_attention_scores

    def _sliding_chunks_matmul_attn_probs_value(self, attn_probs, value, num_heads, window_overlap):
        bnh, seq_len, head_dim = value.size()
        batch_size = bnh // num_heads
        assert seq_len % (window_overlap * 2) == 0
        assert attn_probs.size(3) == 2 * window_overlap + 1
        chunks_count = seq_len // window_overlap - 1

        chunked_attn_probs = attn_probs.transpose(1, 2).reshape(
            batch_size * num_heads, seq_len // window_overlap, window_overlap, 2 * window_overlap + 1
        )

        padded_value = nn.functional.pad(value, (0, 0, window_overlap, window_overlap), value=-1)

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
        B, C, T = x.size()

        q, qx_mask = self.query_conv(x, mask)
        q = self.query_norm(q.permute(0, 2, 1)).permute(0, 2, 1)
        k, kv_mask = self.key_conv(x, mask)
        k = self.key_norm(k.permute(0, 2, 1)).permute(0, 2, 1)
        v, _ = self.value_conv(x, mask)
        v = self.value_norm(v.permute(0, 2, 1)).permute(0, 2, 1)

        q = self.query(q)
        k = self.key(k)
        v = self.value(v)

        q = q.view(B, self.n_head, self.n_channels, -1).transpose(2, 3)
        k = k.view(B, self.n_head, self.n_channels, -1).transpose(2, 3)
        v = v.view(B, self.n_head, self.n_channels, -1).transpose(2, 3)

        q = q.view(B * self.n_head, -1, self.n_channels).contiguous()
        k = k.view(B * self.n_head, -1, self.n_channels).contiguous()
        v = v.view(B * self.n_head, -1, self.n_channels).contiguous()

        q *= self.scale
        att = self._sliding_chunks_query_key_matmul(q, k, self.n_head, self.window_overlap)

        if self.use_rel_pe:
            att += self.rel_pe

        inverse_kv_mask = torch.logical_not(kv_mask[:, None, :, None].view(B, -1, 1))

        float_inverse_kv_mask = inverse_kv_mask.type_as(q).masked_fill(inverse_kv_mask, -1e4)

        diagonal_mask = self._sliding_chunks_query_key_matmul(
            float_inverse_kv_mask.new_ones(size=float_inverse_kv_mask.size()),
            float_inverse_kv_mask,
            1,
            self.window_overlap,
        )
        att += diagonal_mask

        att = nn.functional.softmax(att, dim=-1)
        att = att.masked_fill(torch.logical_not(kv_mask[:, :, None, None]), 0.0)
        att = self.attn_drop(att)

        out = self._sliding_chunks_matmul_attn_probs_value(att, v, self.n_head, self.window_overlap)
        out = out.transpose(2, 3).contiguous().view(B, C, -1)
        out = self.proj_drop(self.proj(out)) * qx_mask.unsqueeze(1).to(out.dtype)
        return out, qx_mask


def drop_path(x, drop_prob=0.0, training=False):
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    mask = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    mask.floor_()
    output = x.div(keep_prob) * mask
    return output


class AffineDropPath(nn.Module):
    def __init__(self, num_dim, drop_prob=0.0, init_scale_value=1e-4, transpose=False):
        super().__init__()
        self.scale = nn.Parameter(init_scale_value * torch.ones((1, num_dim, 1)), requires_grad=True)
        self.drop_prob = drop_prob
        self.transpose = transpose

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
    def __init__(
            self,
            in_channels,
            n_head,
            n_ds_strides=(1, 1),
            n_out=None,
            n_hidden=None,
            act_layer=nn.GELU,
            attn_pdrop=0.0,
            proj_pdrop=0.0,
            path_pdrop=0.0,
            mha_win_size=-1,
    ):
        super().__init__()
        assert len(n_ds_strides) == 2

        self.ln1 = nn.LayerNorm(in_channels)
        self.ln2 = nn.LayerNorm(in_channels)

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

        if n_ds_strides[0] > 1:
            kernel_size, stride, padding = n_ds_strides[0] + 1, n_ds_strides[0], (n_ds_strides[0] + 1) // 2
            self.pool_skip = nn.MaxPool1d(kernel_size, stride=stride, padding=padding)
        else:
            self.pool_skip = nn.Identity()

        if n_hidden is None:
            n_hidden = 4 * in_channels
        if n_out is None:
            n_out = in_channels

        self.mlp = nn.Sequential(
            nn.Conv1d(in_channels, n_hidden, 1),
            act_layer(),
            nn.Dropout(proj_pdrop, inplace=True),
            nn.Conv1d(n_hidden, n_out, 1),
            nn.Dropout(proj_pdrop, inplace=True),
        )

        if path_pdrop > 0.0:
            self.drop_path_attn = AffineDropPath(in_channels, drop_prob=path_pdrop)
            self.drop_path_mlp = AffineDropPath(n_out, drop_prob=path_pdrop)
        else:
            self.drop_path_attn = nn.Identity()
            self.drop_path_mlp = nn.Identity()

    def forward(self, x, mask):
        out, out_mask = self.attn(self.ln1(x.permute(0, 2, 1)).permute(0, 2, 1), mask)
        out_mask_float = out_mask.to(out.dtype)
        out = self.pool_skip(x) * out_mask_float.unsqueeze(1) + self.drop_path_attn(out)

        out = out + self.drop_path_mlp(
            self.mlp(self.ln2(out.permute(0, 2, 1)).permute(0, 2, 1)) * out_mask_float.unsqueeze(1)
        )
        return out, out_mask


def get_sinusoid_encoding(n_position, d_hid):
    def get_position_angle_vec(position):
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    return torch.FloatTensor(sinusoid_table).unsqueeze(0).transpose(1, 2)
