from typing import List, Optional, Dict

from mmengine.logging import MMLogger
from mmengine.model import BaseModule, ModuleDict
from mmengine.structures import BaseDataElement as DataSample

from mmpretrain.registry import MODELS
from mmpretrain.models.omnifd.base import UnifiedDataSample


@MODELS.register_module()
class UnifiedHead(BaseModule):
    def __init__(self,
                 in_channels=None,
                 decoder_dict: dict = dict(),
                 auxiliary_dict: dict = dict(),
                 init_cfg: Optional[dict] = None,

                 train_tasks=['video'],
                 test_tasks=['video'],
                 ):
        super().__init__(init_cfg=init_cfg)

        self.train_tasks = train_tasks
        self.test_tasks = test_tasks

        self.decoder_modules = ModuleDict()
        self.auxiliary_modules = ModuleDict()

        for task_name, decoder_module in decoder_dict.items():
            if task_name not in self.train_tasks:
                logger = MMLogger.get_current_instance()
                logger.info(f'Based on train_tasks {train_tasks}, skip building module: {task_name}')
                continue

            if in_channels is not None and decoder_module.get('in_channels') is None:
                decoder_module['in_channels'] = in_channels

            self.decoder_modules.register_module(task_name, MODELS.build(decoder_module))

        for task_name, auxiliary_module in auxiliary_dict.items():
            if auxiliary_module is None or task_name not in self.train_tasks:
                continue
            self.auxiliary_modules.register_module(task_name, MODELS.build(auxiliary_module))

    def forward(self, feats: Dict, data_samples=None):
        results = []

        for _feats, _data_samples in zip(feats, data_samples):
            feat_tasks = _data_samples[0].training_tasks()

            _results = dict()

            for task_name, decoder in self.decoder_modules.items():
                if task_name not in feat_tasks:
                    continue

                sub_results = decoder(_feats, data_samples=_data_samples)
                _results[task_name] = sub_results

            results.append(_results)
        return results

    def loss(self, feats: Dict, data_samples: List[DataSample], **kwargs) -> dict:
        results = self(feats, data_samples)

        losses = dict()

        for i, (_results, _data_samples) in enumerate(zip(results, data_samples)):
            feat_tasks = _data_samples[0].training_tasks()

            for task_name, decoder in self.decoder_modules.items():
                if task_name not in feat_tasks:
                    continue

                pred = _results[task_name]

                loss = decoder.loss(pred, _data_samples, task_name=task_name)

                loss = {k.replace(task_name, f'{task_name}.{i}'): v for k, v in loss.items()}
                losses.update(loss)

            for name, auxiliary_module in self.auxiliary_modules.items():
                loss = auxiliary_module.loss(feats, data_samples, self.decoder_modules)
                losses.update(loss)
        return losses

    def predict(self,
                feats: Dict,
                data_samples: Optional[List[Optional[UnifiedDataSample]]] = None) -> List[UnifiedDataSample]:
        if data_samples is None:
            raise NotImplementedError()

        results = self(feats, data_samples)

        for i, (_results, _data_samples) in enumerate(zip(results, data_samples)):
            feat_tasks = _data_samples[0].training_tasks()

            for task_name, decoder in self.decoder_modules.items():
                if task_name not in self.test_tasks or task_name not in feat_tasks:
                    continue

                pred = _results[task_name]

                new_data_samples = decoder.predict(pred, _data_samples, task_name=task_name)
                data_samples[i] = new_data_samples

        return data_samples
