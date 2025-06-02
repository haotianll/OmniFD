from typing import List, Optional, Sequence

import torch
import torch.nn as nn
from mmengine.model import BaseModel
from mmengine.structures import BaseDataElement as DataSample

from mmpretrain.registry import MODELS


@MODELS.register_module()
class UnifiedDetector(BaseModel):
    def __init__(self,
                 backbone: dict,
                 neck: Optional[dict] = None,
                 head: Optional[dict] = None,
                 pretrained: Optional[str] = None,
                 train_cfg: Optional[dict] = None,
                 data_preprocessor: Optional[dict] = None,
                 init_cfg: Optional[dict] = None,
                 ):
        if pretrained is not None:
            init_cfg = dict(type='Pretrained', checkpoint=pretrained)

        data_preprocessor = data_preprocessor or {}

        if isinstance(data_preprocessor, dict):
            data_preprocessor = MODELS.build(data_preprocessor)
        elif not isinstance(data_preprocessor, nn.Module):
            raise TypeError('data_preprocessor should be a `dict` or '
                            f'`nn.Module` instance, but got {type(data_preprocessor)}')

        super().__init__(init_cfg=init_cfg, data_preprocessor=data_preprocessor)

        if not isinstance(backbone, nn.Module):
            backbone = MODELS.build(backbone)
        if neck is not None and not isinstance(neck, nn.Module):
            neck = MODELS.build(neck)
        if head is not None and not isinstance(head, nn.Module):
            head = MODELS.build(head)

        self.backbone = backbone
        self.neck = neck
        self.head = head

    @property
    def with_neck(self) -> bool:
        return hasattr(self, 'neck') and self.neck is not None

    @property
    def with_head(self) -> bool:
        return hasattr(self, 'head') and self.head is not None

    def forward(self,
                inputs: torch.Tensor,
                data_samples: Optional[List[DataSample]] = None,
                mode: str = 'tensor',
                **kwargs):

        if mode == 'tensor':
            feats = self.extract_feat(inputs, data_samples=data_samples)
            return self.head(feats) if self.with_head else feats
        elif mode == 'loss':
            return self.loss(inputs, data_samples)
        elif mode == 'predict':
            return self.predict(inputs, data_samples)
        else:
            raise RuntimeError(f'Invalid mode "{mode}".')

    def extract_feat_origin(self, x):
        x = self.backbone(x)
        if self.with_neck:
            x = self.neck(x)
        out = x
        if isinstance(out, list):
            out = tuple(out)
        return out

    def extract_feat_image(self, image, data_samples=None):
        image = self.backbone(image)
        if self.with_neck:
            image = self.neck(image, data_samples=data_samples)

        if isinstance(image, list):
            image = tuple(image)
        return image

    def extract_feat_video(self, video, data_samples=None):
        video = self.backbone(video)
        if self.with_neck:
            video = self.neck(video, data_samples=data_samples)

        if isinstance(video, list):
            video = tuple(video)
        return video

    def extract_feat(self, inputs, data_samples=None, **kwargs):
        if not isinstance(inputs, dict):
            return self.extract_feat_origin(inputs)

        if data_samples is None:
            raise NotImplementedError()

        outs = []

        for x, data_type, data_sample in zip(inputs['data'], inputs['data_types'], data_samples):
            if data_type == 'image':
                out = self.extract_feat_image(x, data_samples=data_sample)
            elif data_type == 'video':
                out = self.extract_feat_video(x, data_samples=data_sample)
            else:
                raise NotImplementedError()
            outs.append(out)
        return outs

    def extract_feats(self, multi_inputs: Sequence[torch.Tensor], **kwargs) -> list:
        assert isinstance(multi_inputs, Sequence), \
            '`extract_feats` is used for a sequence of inputs tensor. If you ' \
            'want to extract on single inputs tensor, use `extract_feat`.'
        return [self.extract_feat(inputs, **kwargs) for inputs in multi_inputs]

    def loss(self, inputs: torch.Tensor, data_samples: List[DataSample], **kwargs) -> dict:
        feats = self.extract_feat(inputs, data_samples=data_samples)
        loss = self.head.loss(feats, data_samples)
        return loss

    def predict(self,
                inputs: torch.Tensor,
                data_samples: Optional[List[DataSample]] = None,
                **kwargs) -> List[DataSample]:
        feats = self.extract_feat(inputs, data_samples=data_samples)
        return self.head.predict(feats, data_samples, **kwargs)

    def train_step(self, data, optim_wrapper):
        with optim_wrapper.optim_context(self):
            data = self.data_preprocessor(data, True)
            losses = self._run_forward(data, mode='loss')  # type: ignore
        parsed_losses, log_vars = self.parse_losses(losses)  # type: ignore
        optim_wrapper.update_params(parsed_losses)
        return log_vars
