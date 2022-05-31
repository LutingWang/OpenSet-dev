from abc import abstractmethod
from collections import namedtuple
from typing import Any, Dict, List, Tuple, Union, Optional
import numbers

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
import todd.reproduction

from .mmdet_patch import DyHeadBlock


MIL_CLASSIFIERS = Registry('MIL_CLASSIFIER')
ClassificationResult = namedtuple('ClassificationResult', ['class_embeddings', 'image_features', 'class_logits', 'logits_weight', 'indices'])


class Classifier(BaseModule):
    def __init__(self, tau: Tuple[float, float] = (0.07, 0.07), bias: Optional[float] = None):
        super().__init__()
        if isinstance(tau, numbers.Number):
            tau = (tau, tau)
        self._tau = tau
        self._bias = (
            None if bias is None else 
            nn.Parameter(torch.FloatTensor(data=[bias]))
        )

    @property
    def tau(self) -> float:
        return self._tau[self.training]

    def set_weight(self, weight: Optional[torch.Tensor], norm: bool = True):
        if isinstance(weight, nn.Parameter):
            weight = weight.data
        if norm:
            weight = F.normalize(weight)
        self._weight = weight

    def forward_hook(
        self, module: Any, input_: Any, output: torch.Tensor, 
    ) -> torch.Tensor:
        return self.forward(output)

    def forward(self, x: torch.Tensor, norm: bool = True) -> torch.Tensor:
        if norm:
            x = F.normalize(x)
        if self._weight is None:
            return x
        if x.ndim == 2:
            input_pattern = 'b c'
            output_pattern = 'b k'
        elif x.ndim == 4:
            input_pattern = 'b c h w'
            output_pattern = 'b k h w'
        if self._weight.ndim == 2:
            weight_pattern = 'k c'
        elif self._weight.ndim == 3:
            weight_pattern = 'b k c'
        x = torch.einsum(
            f'{input_pattern}, {weight_pattern} -> {output_pattern}', 
            x, self._weight,
        )
        x = x / self.tau
        if self._bias is not None:
            x = x + self._bias
        return x


class BaseMILClassifier(Classifier):
    def __init__(
        self,
        *args,
        channels: int,
        embedding_dim: int,
        kappa: int = 35, 
        loss_mil: ConfigDict = None,
        loss_image_kd: ConfigDict = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        assert kappa >= 10, f'kappa must be >= 10, but got {kappa}.'
        self._channels = channels
        self._embedding_dim = embedding_dim
        self._kappa = kappa
        self._adapt = self._build_adapt()
        self._loss_mil = None if loss_mil is None else LOSSES.build(loss_mil)
        self._loss_image_kd = None if loss_image_kd is None else LOSSES.build(loss_image_kd)
        self._loss_scheduler = todd.schedulers.WarmupScheduler(iter_=1000)

    @todd.reproduction.set_seed_temp('mil_classifier')
    def init_weights(self):
        return super().init_weights()

    @abstractmethod
    def _build_adapt(self) -> BaseModule:
        pass

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        image_features = self._adapt(x)
        image_features = F.normalize(image_features)
        class_logits = super().forward(image_features, norm=False)
        return image_features, class_logits

    def _topk(
        self,
        class_embeddings: torch.Tensor,
        class_logits: torch.Tensor,
    ) -> Tuple[torch.Tensor, ...]:
        class_logits, indices = torch.topk(class_logits, k=self._kappa, dim=-1)
        if class_embeddings.ndim == 2:
            return class_embeddings[indices], class_logits, indices
        if class_embeddings.ndim == 3:
            gather_indices = einops.repeat(
                indices, 'b n -> b n c', c=class_embeddings.shape[-1],
            )
            class_embeddings = torch.gather(
                class_embeddings, 1, gather_indices,
            )
            return class_embeddings, class_logits, indices
        raise ValueError(f'class_embeddings.ndim must be 2 or 3, but got {class_embeddings.shape}')

    def _add_gts(
        self,
        topk_class_embeddings: torch.Tensor,
        topk_class_logits: torch.Tensor,
        topk_indices: torch.Tensor,
        class_embeddings: torch.Tensor,
        class_logits: torch.Tensor,
        mil_labels: torch.Tensor, 
    ):
        for topk_class_embedding, topk_class_logit, topk_index, class_logit, mil_label in zip(
            topk_class_embeddings, topk_class_logits, topk_indices, class_logits, mil_labels,
        ):
            gt_index, = mil_label.index_put((topk_index,), mil_label.new_zeros([])).nonzero(as_tuple=True)
            gt_index = gt_index[:topk_index.shape[0] // 10]
            num_gts = gt_index.shape[0]
            if num_gts == 0:
                continue
            topk_class_embedding[-num_gts:] = class_embeddings[gt_index]
            topk_class_logit[-num_gts:] = class_logit[gt_index]
            topk_index[-num_gts:] = gt_index

    def forward_train(
        self, 
        x: torch.Tensor, 
        class_embeddings: torch.Tensor,
        mil_labels: torch.Tensor,
        gt_image_features: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        self.set_weight(class_embeddings, norm=False)
        image_features, class_logits = self.forward(x)
        topk_class_embeddings, topk_class_logits, topk_indices = self._topk(
            class_embeddings, class_logits.detach(),
        )
        self._add_gts(topk_class_embeddings, topk_class_logits, topk_indices, class_embeddings, class_logits.detach(), mil_labels)
        loss_mil = self._loss_mil(class_logits, mil_labels) * self._loss_scheduler
        loss_image_kd = self._loss_image_kd(image_features, F.normalize(gt_image_features))
        losses = dict(loss_mil=loss_mil, loss_image_kd=loss_image_kd)
        return topk_class_embeddings, topk_class_logits, losses, topk_indices

    def forward_test(
        self, 
        x: torch.Tensor, 
        class_embeddings: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        self.set_weight(class_embeddings, norm=False)
        _, class_logits = self.forward(x)
        topk_class_embeddings, topk_class_logits, _ = self._topk(
            class_embeddings, class_logits,
        )
        return topk_class_embeddings, topk_class_logits


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