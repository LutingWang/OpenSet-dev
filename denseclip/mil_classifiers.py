from abc import abstractmethod
from collections import namedtuple
from typing import Dict, List, Tuple, Union

import einops
import einops.layers.torch
import todd.schedulers
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv import ConfigDict
from mmcv.runner import BaseModule, Sequential
from mmcv.utils import Registry
from todd.losses import LOSSES

from .mmdet_patch import DyHeadBlock
from .model import Classifier


MIL_CLASSIFIERS = Registry('MIL_CLASSIFIER')
ClassificationResult = namedtuple('ClassificationResult', ['class_embeddings', 'image_features', 'class_logits', 'logits_weight', 'indices'])


class BaseMILClassifier(Classifier):
    def __init__(
        self,
        *args,
        channels: int,
        embedding_dim: int,
        logits_weight: bool = False,
        kappa: int = 35, 
        loss_mil: ConfigDict = None,
        loss_image_kd: ConfigDict = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        assert kappa >= 10, f'kappa must be >= 10, but got {kappa}.'
        self._channels = channels
        self._embedding_dim = embedding_dim
        self._logits_weight = logits_weight
        self._kappa = kappa
        self._adapt = self._build_adapt()
        self._loss_mil = None if loss_mil is None else LOSSES.build(loss_mil)
        self._loss_image_kd = None if loss_image_kd is None else LOSSES.build(loss_image_kd)
        self._loss_scheduler = todd.schedulers.WarmupScheduler(iter_=1000)

    @abstractmethod
    def _build_adapt(self) -> BaseModule:
        pass

    def forward(
        self, 
        x: torch.Tensor, 
        class_embeddings: torch.Tensor,
    ) -> ClassificationResult:
        image_features = self._adapt(x)
        image_features = F.normalize(image_features)
        self.set_weight(class_embeddings, norm=False)
        class_logits = super().forward(image_features, norm=False)
        values, indices = torch.topk(class_logits, k=self._kappa, dim=-1)
        if class_embeddings.ndim == 2:
            class_embeddings = class_embeddings[indices]
        elif class_embeddings.ndim == 3:
            gather_indices = einops.repeat(
                indices, 'b n -> b n c', c=class_embeddings.shape[-1],
            )
            class_embeddings = torch.gather(
                class_embeddings, 1, gather_indices,
            )
        else:
            raise ValueError(f'class_embeddings.ndim must be 2 or 3, but got {class_embeddings.shape}')
        logits_weight = values.detach().sigmoid() if self._logits_weight else None
        return ClassificationResult(class_embeddings, image_features, class_logits, logits_weight, indices)

    def losses(
        self,
        result: ClassificationResult,
        mil_labels: torch.Tensor,
        clip_image_features: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        loss_mil = self._loss_mil(
            result.class_logits, 
            mil_labels,
        ) * self._loss_scheduler
        loss_image_kd = self._loss_image_kd(
            result.image_features, 
            F.normalize(clip_image_features), 
        )
        losses = dict(loss_mil=loss_mil, loss_image_kd=loss_image_kd)
        return losses


@MIL_CLASSIFIERS.register_module()
class DyHeadClassifier(BaseMILClassifier):
    def _build_adapt(self) -> Sequential:
        return Sequential(
            DyHeadBlock(self._channels, self._embedding_dim),
            einops.layers.torch.Reduce('b c h w -> b c', reduction='mean'),
        )


@MIL_CLASSIFIERS.register_module()
class GAPClassifier(BaseMILClassifier):
    def _build_adapt(self) -> Sequential:
        return Sequential(
            einops.layers.torch.Reduce('b c h w -> b c', reduction='mean'),
            nn.Linear(self._channels, self._embedding_dim),
        )
