from abc import abstractmethod
from collections import namedtuple
from typing import List, Tuple, Union

import einops
import einops.layers.torch
import todd.utils
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.runner import BaseModule, Sequential
from mmcv.utils import Registry

from .mmdet_patch import DyHeadBlock


MIL_CLASSIFIERS = Registry('MIL_CLASSIFIER')
ClassificationResult = namedtuple('ClassificationResult', ['class_embeddings', 'image_features', 'class_logits', 'logits_weight', 'indices'])


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
        assert kappa >= 10, f'kappa must be >= 10, but got {kappa}.'
        self._channels = channels
        self._embedding_dim = embedding_dim
        self._logits_weight = logits_weight
        self._kappa = kappa
        self._adapt = self._build_adapt()

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
        if class_embeddings.ndim == 2:
            class_embeddings_shape = 'c d'
        elif class_embeddings.ndim == 3:
            class_embeddings_shape = 'b c d'
        else:
            raise ValueError(f'class_embeddings.ndim must be 2 or 3, but got {class_embeddings.shape}')
        class_logits: torch.Tensor = torch.einsum(
            f'b d, {class_embeddings_shape} -> b c', 
            image_features, class_embeddings,
        )
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


@MIL_CLASSIFIERS.register_module()
class DyHeadClassifier(BaseMILClassifier):
    def _build_adapt(self) -> Sequential:
        return Sequential(
            DyHeadBlock(self._channels, self._embedding_dim),
            einops.layers.torch.Reduce('b c h w -> b c', reduction='mean'),
        )


@MIL_CLASSIFIERS.register_module()
class GAPClassifier(BaseMILClassifier):
    def adapt(self) -> Sequential:
        return Sequential(
            einops.layers.torch.Reduce('b c h w -> b c', reduction='mean'),
            nn.Linear(self._channels, self._embedding_dim),
        )
