from abc import abstractmethod
from collections import namedtuple
from typing import List, Tuple, Union

import einops
import einops.layers.torch
import todd.utils
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.runner import BaseModule
from mmcv.utils import Registry

from .mmdet_patch import DyHeadBlock


MIL_CLASSIFIER = Registry('MIL_CLASSIFIER')
ClassificationResult = namedtuple('ClassificationResult', ['class_embeddings', 'image_features', 'class_logits', 'logits_weight'])


class BaseMILClassifier(BaseModule):
    def __init__(
        self,
        *args,
        channels: int,
        embedding_dim: int,
        logits_weight: bool = False,
        kappa: int = 35, 
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._channels = channels
        self._embedding_dim = embedding_dim
        self._logits_weight = logits_weight
        self._kappa = kappa

    @abstractmethod
    def adapt(self, x: torch.Tensor) -> torch.Tensor:
        pass

    def forward(
        self, 
        x: torch.Tensor, 
        class_embeddings: torch.Tensor,
    ) -> ClassificationResult:
        image_features = self.adapt(x)
        image_features = F.normalize(image_features)
        class_logits: torch.Tensor = torch.einsum('b d, c d -> b c', image_features, class_embeddings)
        values, indices = torch.topk(class_logits, k=self._kappa, dim=-1)
        class_embeddings = class_embeddings[indices]
        logits_weight = values.detach().sigmoid() if self._logits_weight else None
        return ClassificationResult(class_embeddings, image_features, class_logits, logits_weight)


@MIL_CLASSIFIER.register_module()
class DyHeadClassifier(BaseMILClassifier):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._adapter = nn.Sequential(
            DyHeadBlock(self._channels, self._embedding_dim),
            einops.layers.torch.Reduce('b c h w -> b c', reduction='mean'),
        )

    def adapt(self, x: torch.Tensor) -> torch.Tensor:
        return self._adapter(x)


@MIL_CLASSIFIER.register_module()
class GAPClassifier(BaseMILClassifier):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._adapter = nn.Sequential(
            einops.layers.torch.Reduce('b c h w -> b c', reduction='mean'),
            nn.Linear(self._channels, self._embedding_dim),
        )

    def adapt(self, x: torch.Tensor) -> torch.Tensor:
        return self._adapter(x)
