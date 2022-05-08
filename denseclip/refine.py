from abc import abstractmethod
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import einops
import einops.layers.torch
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv import ConfigDict
from mmcv.runner import BaseModule, ModuleList
from mmcv.utils import Registry
from timm.models.layers import DropPath
from todd.losses import LOSSES

from .mil_classifiers import BaseMILClassifier, MIL_CLASSIFIERS, ClassificationResult
from .mmdet_patch import DyHeadBlock


REFINE_LAYERS = Registry('REFINE')


class Fusion(BaseModule):
    def __init__(
        self, 
        *args, 
        num_heads: int = 8,
        embed_dim: int = 2048,
        v_dim: int = 256, 
        l_dim: int = 512,
        avg_factor: int,
        dropout: float = 0.1, 
        drop_path: float = 0.0,
        bi_direct: bool = True,
        **kwargs,
    ):
        super().__init__(
            *args,
            init_cfg=dict(
                type='Xavier', layer='Conv2d', distribution='uniform',
            ),
            **kwargs,
        )

        head_dim = embed_dim // num_heads
        assert head_dim * num_heads == embed_dim

        self._num_heads = num_heads
        self._head_dim = head_dim
        self._scale = head_dim ** (-0.5)

        self._v_layer_norm = nn.LayerNorm(v_dim)
        self._v_proj = nn.Linear(v_dim, embed_dim)
        self._values_l_proj = nn.Linear(l_dim, embed_dim)
        self._out_v_proj = nn.Linear(embed_dim, v_dim)
        self._gamma_v = nn.Parameter(torch.ones((v_dim)) / avg_factor, requires_grad=True)

        self._l_layer_norm = nn.LayerNorm(l_dim)
        self._l_proj = nn.Linear(l_dim, embed_dim)
        if bi_direct:
            self._values_v_proj = nn.Linear(v_dim, embed_dim)
            self._out_l_proj = nn.Linear(embed_dim, l_dim)
            self._gamma_l = nn.Parameter(torch.ones((l_dim)) / avg_factor, requires_grad=True)

        self._dropout = dropout
        self._drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self._bi_direct = bi_direct

        self._stable_softmax_2d = False
        self._clamp_min_for_underflow = True
        self._clamp_max_for_overflow = True

    def forward(self, v: torch.Tensor, l: torch.Tensor, l_weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        h, w = v.shape[-2:]
        v = einops.rearrange(v, 'b c h w -> b (h w) c')

        v = self._v_layer_norm(v)
        l = self._l_layer_norm(l)

        query_states: torch.Tensor = einops.rearrange(
            self._v_proj(v) * self._scale, 
            'b hw (num_heads head_dim) -> (b num_heads) hw head_dim', 
            num_heads=self._num_heads, head_dim=self._head_dim,
        )
        key_states: torch.Tensor = einops.rearrange(
            self._l_proj(l), 
            'b l (num_heads head_dim) -> (b num_heads) l head_dim', 
            num_heads=self._num_heads, head_dim=self._head_dim,
        )

        attn_weights = torch.einsum('b n c, b l c -> b n l', query_states, key_states)
        if self._stable_softmax_2d:
            attn_weights = attn_weights - attn_weights.max()
        attn_weights = torch.clamp(
            attn_weights, 
            min=-50000 if self._clamp_min_for_underflow else None,
            max=50000 if self._clamp_max_for_overflow else None,
        )  # Do not increase 50000, data type half has quite limited range
        attn_weights_v = attn_weights.softmax(dim=-1)
        if l_weights is not None:
            l_weights = einops.repeat(l_weights, 'b num_heads -> (b num_heads) 1 l', num_heads=self._num_heads)
            attn_weights_v = attn_weights_v * l_weights
            attn_weights_v = attn_weights_v / attn_weights_v.sum(dim=-1, keepdim=True)
        attn_probs_v = F.dropout(attn_weights_v, p=self._dropout, training=self.training)
        value_l_states = einops.rearrange(self._values_l_proj(l), 'b l (num_heads head_dim) -> (b num_heads) l head_dim', num_heads=self._num_heads, head_dim=self._head_dim)
        attn_output_v = torch.einsum('b n l, b l c -> b n c', attn_probs_v, value_l_states)
        attn_output_v = einops.rearrange(attn_output_v, '(b num_heads) n head_dim -> b n (num_heads head_dim)', num_heads=self._num_heads, head_dim=self._head_dim)
        delta_v = self._out_v_proj(attn_output_v)

        if self._bi_direct:
            attn_weights = einops.rearrange(attn_weights, 'b hw l -> b l hw')
            attn_weights = attn_weights - torch.max(attn_weights, dim=-1, keepdim=True)[0]
            attn_weights = torch.clamp(
                attn_weights, 
                min=-50000 if self._clamp_min_for_underflow else None,
                max=50000 if self._clamp_max_for_overflow else None,
            )  # Do not increase 50000, data type half has quite limited range
            attn_weights_l = attn_weights.softmax(dim=-1)
            attn_probs_l = F.dropout(attn_weights_l, p=self._dropout, training=self.training)
            value_v_states = einops.rearrange(self._values_v_proj(v), 'b hw (num_heads head_dim) -> (b num_heads) hw head_dim', num_heads=self._num_heads, head_dim=self._head_dim)
            attn_output_l = torch.einsum('b l n, b n c -> b l c', attn_probs_l, value_v_states)
            attn_output_l = einops.rearrange(attn_output_l, '(b num_heads) l head_dim -> b l (num_heads head_dim)', num_heads=self._num_heads, head_dim=self._head_dim)
            delta_l = self._out_l_proj(attn_output_l)

        v = v + self._drop_path(self._gamma_v * delta_v)
        v = einops.rearrange(v, 'b (h w) c -> b c h w', h=h, w=w)
        if self._bi_direct:
            l = l + self._drop_path(self._gamma_l * delta_l)
            return v, l
        else:
            return v, None


class BaseFusionDyHead(BaseModule):
    def __init__(
        self, 
        *args, 
        channels: int, 
        embedding_dim: int,
        mil_classifier,
        num_layers: int = 6, 
        loss_mil: ConfigDict,
        loss_image_kd: ConfigDict,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._channels = channels
        self._embedding_dim = embedding_dim
        self._num_layers = num_layers
        self._loss_mil = LOSSES.build(loss_mil)
        self._loss_image_kd = LOSSES.build(loss_image_kd)

        self._fuse_layers = ModuleList(
            Fusion(
                avg_factor=num_layers, 
                bi_direct=(l != num_layers - 1),
            ) for l in range(num_layers)
        )
        self._dyhead_layers = ModuleList(
            DyHeadBlock(channels, channels) for _ in range(num_layers)
        )

        self._build_mil_classifiers(mil_classifier)
        self.init_weights()

    @abstractmethod
    def _build_mil_classifiers(self, *args, **kwargs):
        pass


@REFINE_LAYERS.register_module()
class StandardFusionDyHead(BaseFusionDyHead):
    def _build_mil_classifiers(self, config: ConfigDict):
        self._mil_classifier: BaseMILClassifier = MIL_CLASSIFIERS.build(
            config, 
            default_args=dict(
                channels=self._channels,
                embedding_dim=self._embedding_dim,
            ),
        )

    def forward(
        self, 
        bsf: torch.Tensor, 
        class_embeddings: torch.Tensor, 
        mil_labels: Optional[torch.Tensor] = None,
        clip_image_features: Optional[torch.Tensor] = None,
    ):
        classification_result: ClassificationResult = self._mil_classifier(bsf, class_embeddings)
        class_embeddings = classification_result.class_embeddings
        for fuse_layer, dyhead_layer in zip(self._fuse_layers, self._dyhead_layers):
            bsf, class_embeddings = fuse_layer(
                bsf, class_embeddings, 
                classification_result.logits_weight,
            )
            bsf = dyhead_layer(bsf)
        assert class_embeddings is None
        if not self.training:
            assert mil_labels is None
            return bsf

        loss_mil = self._loss_mil(
            classification_result.class_logits, 
            mil_labels,
        )

        loss_image_kd = self._loss_image_kd(
            classification_result.image_features, 
            F.normalize(clip_image_features), 
        )
        return bsf, dict(loss_mil=loss_mil, loss_image_kd=loss_image_kd)


@REFINE_LAYERS.register_module()
class CascadeFusionDyHead(BaseFusionDyHead):
    def _build_mil_classifiers(self, configs: List[ConfigDict]):
        assert len(configs) == self._num_layers
        self._mil_classifiers = ModuleList([MIL_CLASSIFIERS.build(
            config, 
            default_args=dict(
                channels=self._channels,
                embedding_dim=self._embedding_dim,
            ),
        ) for config in configs])

    def forward(
        self, 
        bsf: torch.Tensor, 
        class_embeddings: torch.Tensor, 
        mil_labels: Optional[torch.Tensor] = None,
        clip_image_features: Optional[torch.Tensor] = None,
    ):
        if self.training:
            assert mil_labels is not None
            losses = defaultdict(list)
            gt_class_embeddings = [class_embeddings[mil_label.bool()] for mil_label in mil_labels]
        else:
            assert mil_labels is None

        for layer, (fuse_layer, dyhead_layer) in enumerate(zip(self._fuse_layers, self._dyhead_layers)):
            mil_classifier = self._mil_classifiers[layer]
            classification_result: ClassificationResult = mil_classifier(bsf, class_embeddings)

            bsf, class_embeddings = fuse_layer(
                bsf, 
                classification_result.class_embeddings, 
                classification_result.logits_weight,
            )
            bsf = dyhead_layer(bsf)

            if not self.training:
                continue

            loss_mil = self._loss_mil(
                classification_result.class_logits, 
                mil_labels,
            )
            losses['loss_mil'].append(loss_mil)
            loss_image_kd = self._loss_image_kd(
                classification_result.image_features, 
                F.normalize(clip_image_features), 
            )
            losses['loss_image_kd'].append(loss_image_kd)

            if class_embeddings is None:
                continue

            mil_labels = torch.gather(mil_labels, 1, classification_result.indices)
            max_num_gts = max(class_embeddings.shape[1] // 10, 1)
            for i, gt_class_embedding in enumerate(gt_class_embeddings):
                gt_class_embedding = gt_class_embedding[:max_num_gts]
                num_gts = gt_class_embedding.shape[0]
                class_embeddings[i, -num_gts:] = gt_class_embedding
                mil_labels[i, -num_gts:] = 1

        assert class_embeddings is None
        if self.training:
            return bsf, {k: sum(v) / len(v) for k, v in losses.items()}
        return bsf

