from abc import abstractmethod
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple, cast

import einops
import einops.layers.torch
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv import ConfigDict
from mmcv.runner import BaseModule, ModuleList
from mmcv.utils import Registry
from timm.models.layers import DropPath

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
            l_weights = einops.repeat(l_weights, 'b l -> (b num_heads) 1 l', num_heads=self._num_heads)
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


def _add_gts(
    classification_result: ClassificationResult, 
    mil_labels: torch.Tensor, 
    all_class_embeddings: torch.Tensor,
):
    for i in range(mil_labels.shape[0]):
        mil_label: torch.Tensor = mil_labels[i]
        class_embeddings: torch.Tensor = classification_result.class_embeddings[i]
        indices: torch.Tensor = classification_result.indices[i]
        gt_inds, = mil_label.index_put((indices,), mil_label.new_zeros([])).nonzero(as_tuple=True)
        gt_inds = gt_inds[:indices.shape[0] // 10]
        num_gts = gt_inds.shape[0]
        if num_gts == 0:
            continue
        if all_class_embeddings.ndim == 2:
            gt_class_embeddings = all_class_embeddings[gt_inds]
        elif all_class_embeddings.ndim == 3:
            gt_class_embeddings = all_class_embeddings[i, gt_inds]
        else:
            raise ValueError('all_class_embeddings.ndim must be 2 or 3')
        class_embeddings[-num_gts:] = gt_class_embeddings
        indices[-num_gts:] = gt_inds
        if classification_result.logits_weight is not None:
            logits_weight: torch.Tensor = classification_result.logits_weight[i]
            class_logits: torch.Tensor = classification_result.class_logits[i]
            logits_weight[-num_gts:] = class_logits[gt_inds].detach().sigmoid()


@REFINE_LAYERS.register_module()
class BaseFusionDyHead(BaseModule):
    def __init__(
        self, 
        *args, 
        channels: int, 
        embedding_dim: int,
        num_layers: int = 6, 
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._channels = channels
        self._embedding_dim = embedding_dim
        self._num_layers = num_layers

        self._fuse_layers = ModuleList(
            Fusion(
                avg_factor=num_layers, 
                bi_direct=(l != num_layers - 1),
            ) for l in range(num_layers)
        )
        self._dyhead_layers = ModuleList(
            DyHeadBlock(channels, channels) for _ in range(num_layers)
        )

        self.init_weights()

    def forward(
        self, 
        bsf: torch.Tensor, 
        class_embeddings: torch.Tensor, 
        logits_weight: Optional[torch.Tensor] = None,
    ):
        for fuse_layer, dyhead_layer in zip(self._fuse_layers, self._dyhead_layers):
            bsf, class_embeddings = fuse_layer(
                bsf, class_embeddings, logits_weight,
            )
            bsf = dyhead_layer(bsf)
        assert class_embeddings is None
        if self.training:
            return bsf, dict()
        return bsf


@REFINE_LAYERS.register_module()
class StandardFusionDyHead(BaseFusionDyHead):
    def __init__(self, *args, mil_classifier: ConfigDict, **kwargs):
        super().__init__(*args, **kwargs)
        self._mil_classifier: BaseMILClassifier = MIL_CLASSIFIERS.build(
            mil_classifier, 
            default_args=dict(
                channels=self._channels,
                embedding_dim=self._embedding_dim,
            ),
        )

    def forward(
        self, 
        bsf: torch.Tensor, 
        class_embeddings: torch.Tensor, 
        **kwargs,
    ):
        classification_result: ClassificationResult = self._mil_classifier(bsf, class_embeddings)
        results = super().forward(
            bsf, 
            classification_result.class_embeddings, 
            classification_result.logits_weight,
        )
        if not self.training:
            assert len(kwargs) == 0
            return results

        losses = self._mil_classifier.losses(classification_result, **kwargs)
        return results[0], losses


@REFINE_LAYERS.register_module()
class CascadeFusionDyHead(BaseFusionDyHead):
    def __init__(self, *args, mil_classifiers: List[ConfigDict], **kwargs):
        super().__init__(*args, **kwargs)
        assert len(mil_classifiers) == self._num_layers
        self._mil_classifiers = ModuleList([MIL_CLASSIFIERS.build(
            mil_classifier, 
            default_args=dict(
                channels=self._channels,
                embedding_dim=self._embedding_dim,
            ),
        ) for mil_classifier in mil_classifiers])

    def forward(
        self, 
        bsf: torch.Tensor, 
        class_embeddings: torch.Tensor, 
        mil_labels: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        if self.training:
            assert mil_labels is not None
            losses = defaultdict(list)
        else:
            assert mil_labels is None

        updated_class_embeddings = einops.repeat(class_embeddings, 'c d -> b c d', b=bsf.shape[0])
        for mil_classifier, fuse_layer, dyhead_layer in zip(
            self._mil_classifiers, self._fuse_layers, self._dyhead_layers,
        ):
            mil_classifier = cast(BaseMILClassifier, mil_classifier)
            classification_result: ClassificationResult = mil_classifier(bsf, class_embeddings)
            
            if self.training:
                losses_ = mil_classifier.losses(classification_result, mil_labels=mil_labels, **kwargs)
                for k, v in losses_.items():
                    losses[k].append(v)
                _add_gts(classification_result, mil_labels, updated_class_embeddings)
                mil_labels = torch.gather(mil_labels, 1, classification_result.indices)

            indices = einops.repeat(
                classification_result.indices, 'b n -> b n c', 
                c=updated_class_embeddings.shape[-1],
            )
            updated_class_embeddings = torch.gather(updated_class_embeddings, 1, indices)
            class_embeddings = classification_result.class_embeddings
            bsf, updated_class_embeddings = fuse_layer(
                bsf, updated_class_embeddings, 
                classification_result.logits_weight,
            )
            bsf = dyhead_layer(bsf)

        assert updated_class_embeddings is None
        if self.training:
            return bsf, losses
        return bsf


class PLV(BaseModule):
    def __init__(
        self, *args, v_dim: int, l_dim: int, hidden_dim: int, **kwargs,
    ):
        super().__init__(
            *args, 
            init_cfg=dict(
                type='Xavier', layer='Conv2d', 
                distribution='uniform',
            ),
            **kwargs,
        )
        self._v_proj = nn.Sequential(
            nn.Conv2d(v_dim, hidden_dim, 1),
            nn.Tanh(),
        )
        self._l_proj = nn.Sequential(
            nn.Linear(l_dim, hidden_dim),
            nn.Tanh(),
        )
        self._out_v_proj = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(hidden_dim, v_dim, 1),
            nn.BatchNorm2d(v_dim),
            nn.ReLU()
        )

    def forward(self, v: torch.Tensor, l: torch.Tensor, logits_weight: Optional[torch.Tensor] = None) -> torch.Tensor:
        v_feats = self._v_proj(v)
        if l.ndim == 2:
            l_feats = self._l_proj(l)
            l_feats = einops.reduce(l_feats, 'c d -> 1 d 1 1', reduction='mean')
        elif l.ndim == 3:
            b, c, d = l.shape
            l_feats = einops.rearrange(l, 'b c d -> (b c) d')
            l_feats = self._l_proj(l_feats)
            l_feats = einops.reduce(l_feats, '(b c) d -> b d 1 1', b=b, c=c, reduction='mean')
        v_feats = self._out_v_proj(v_feats * l_feats)
        # v_feats = F.normalize(v + v_feats)
        return v + v_feats


class PLVRefine(BaseModule):
    def __init__(
        self, 
        *args, 
        channels: List[int], 
        embedding_dim: int,
        hidden_dim: int,
        mil_classifier,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._channels = channels
        self._embedding_dim = embedding_dim

        self._mil_classifier: BaseMILClassifier = MIL_CLASSIFIERS.build(
            mil_classifier, 
            default_args=dict(
                channels=channels[-1],
                embedding_dim=self._embedding_dim,
            ),
        )
        self._plvs = ModuleList([
            PLV(v_dim=channel, l_dim=embedding_dim, hidden_dim=hidden_dim)
            for channel in channels
        ])
        self.init_weights()

    def forward(
        self, 
        x: Tuple[torch.Tensor], 
        class_embeddings: torch.Tensor, 
        mil_labels: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        classification_result: ClassificationResult = self._mil_classifier(x[-1], class_embeddings)
            
        if self.training:
            losses = self._mil_classifier.losses(classification_result, mil_labels=mil_labels, **kwargs)
            losses = {f'{k}_plv': v for k, v in losses.items()}
            _add_gts(classification_result, mil_labels, class_embeddings)
            mil_labels = torch.gather(mil_labels, 1, classification_result.indices)

        class_embeddings = classification_result.class_embeddings
        logits_weight = classification_result.logits_weight
        x = tuple(
            plv(feat, class_embeddings, classification_result.logits_weight) 
            for plv, feat in zip(self._plvs, x)
        )

        if self.training:
            return x, class_embeddings, logits_weight, mil_labels, losses
        return x, class_embeddings, logits_weight
