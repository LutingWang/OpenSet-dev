from typing import Dict, List, Optional, Tuple

import einops
import einops.layers.torch
import todd.reproduction
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv import ConfigDict
from mmcv.runner import BaseModule, ModuleList
from timm.models.layers import DropPath

from .mmdet_patch import BFP, DyHeadBlock


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

        self._dropout = nn.Dropout(dropout)
        self._drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self._bi_direct = bi_direct

        # Do not increase 50000, data type half has quite limited range
        self._clamp_min: Optional[int] = -50000
        self._clamp_max: Optional[int] = 50000

    def forward(
        self, 
        v: torch.Tensor, 
        l: torch.Tensor, 
        *,
        v_weights: Optional[torch.Tensor] = None,
        l_weights: Optional[torch.Tensor] = None,
        with_masks: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        h, w = v.shape[-2:]
        v = einops.rearrange(v, 'b c h w -> b (h w) c')

        v = self._v_layer_norm(v)
        l = self._l_layer_norm(l)

        attn_weights = self._attn_weights(v, l)
        delta_v = self._attn_v(attn_weights, l, l_weights)

        if with_masks:
            masks = einops.reduce(
                attn_weights, '(b num_heads) (h w) l -> b l h w', 
                num_heads=self._num_heads, h=h, w=w, reduction='mean',
            )
        else:
            masks = None

        if self._bi_direct:
            attn_weights = einops.rearrange(attn_weights, 'b hw l -> b l hw')
            attn_weights = attn_weights - torch.max(attn_weights, dim=-1, keepdim=True).values
            delta_l = self._attn_l(attn_weights, v, v_weights)
            l = l + self._drop_path(self._gamma_l * delta_l)
        else:
            l = None

        v = v + self._drop_path(self._gamma_v * delta_v)
        v = einops.rearrange(v, 'b (h w) c -> b c h w', h=h, w=w)
        return v, l, masks

    def _attn_l(
        self, 
        attn_weights: torch.Tensor, 
        v: torch.Tensor, 
        v_weights: Optional[torch.Tensor],
    ) -> torch.Tensor:
        attn_weights = torch.clamp(
            attn_weights, min=self._clamp_min, max=self._clamp_max,
        )  
        attn_weights_l = attn_weights.softmax(dim=-1)
        if v_weights is not None:
            raise NotImplementedError
        attn_probs_l = self._dropout(attn_weights_l)
        value_v_states = einops.rearrange(self._values_v_proj(v), 'b hw (num_heads head_dim) -> (b num_heads) hw head_dim', num_heads=self._num_heads, head_dim=self._head_dim)
        attn_output_l = torch.einsum('b l n, b n c -> b l c', attn_probs_l, value_v_states)
        attn_output_l = einops.rearrange(attn_output_l, '(b num_heads) l head_dim -> b l (num_heads head_dim)', num_heads=self._num_heads, head_dim=self._head_dim)
        delta_l = self._out_l_proj(attn_output_l)
        return delta_l

    def _attn_v(
        self, 
        attn_weights: torch.Tensor, 
        l: torch.Tensor, 
        l_weights: Optional[torch.Tensor],
    ) -> torch.Tensor:
        attn_weights = torch.clamp(
            attn_weights, min=self._clamp_min, max=self._clamp_max,
        )  
        attn_weights_v = attn_weights.softmax(dim=-1)
        if l_weights is not None:
            l_weights = einops.repeat(l_weights, 'b l -> (b num_heads) 1 l', num_heads=self._num_heads)
            attn_weights_v = attn_weights_v * l_weights
            attn_weights_v = attn_weights_v / attn_weights_v.sum(dim=-1, keepdim=True)
        attn_probs_v = self._dropout(attn_weights_v)
        value_l_states = einops.rearrange(self._values_l_proj(l), 'b l (num_heads head_dim) -> (b num_heads) l head_dim', num_heads=self._num_heads, head_dim=self._head_dim)
        attn_output_v = torch.einsum('b n l, b l c -> b n c', attn_probs_v, value_l_states)
        attn_output_v = einops.rearrange(attn_output_v, '(b num_heads) n head_dim -> b n (num_heads head_dim)', num_heads=self._num_heads, head_dim=self._head_dim)
        delta_v = self._out_v_proj(attn_output_v)
        return delta_v

    def _attn_weights(self, v: torch.Tensor, l: torch.Tensor) -> torch.Tensor:
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
        return attn_weights


class GLIP(BaseModule):
    def __init__(
        self, 
        *args, 
        channels: int, 
        num_layers: int = 6, 
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._channels = channels
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

    def forward_train(
        self, 
        bsf: torch.Tensor, 
        class_embeddings: torch.Tensor, 
        class_weights: Optional[torch.Tensor] = None,
        gt_masks: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, List[torch.Tensor]]]:
        multi_layer_masks = []
        for fuse_layer, dyhead_layer in zip(self._fuse_layers, self._dyhead_layers):
            bsf, class_embeddings, masks = fuse_layer(
                bsf, class_embeddings, l_weights=class_weights, with_masks=True,
            )
            multi_layer_masks.append(masks)
            bsf = dyhead_layer(bsf)
        assert class_embeddings is None
        return bsf, {}

    def forward_test(
        self, 
        bsf: torch.Tensor, 
        class_embeddings: torch.Tensor, 
        class_weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        for fuse_layer, dyhead_layer in zip(self._fuse_layers, self._dyhead_layers):
            bsf, class_embeddings, _ = fuse_layer(
                bsf, class_embeddings, l_weights=class_weights, with_masks=False,
            )
            bsf = dyhead_layer(bsf)
        assert class_embeddings is None
        return bsf


class GLIPNeck(BaseModule):
    def __init__(
        self, 
        *args,
        channels: int,
        refine_level: int,
        refine_layers: int, 
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._refine_level = refine_level
        self._refine = GLIP(
            channels=channels,
            num_layers=refine_layers,
        )

    @todd.reproduction.set_seed_temp('GLIPNeck')
    def init_weights(self):
        return super().init_weights()

    def _gather(self, feats: Tuple[torch.Tensor]) -> torch.Tensor:
        bsf = feats[self._refine_level]
        bsf = bsf.new_empty((len(feats),) + bsf.shape)
        for i in range(self._refine_level):
            bsf[i] = F.adaptive_max_pool2d(feats[i], bsf.shape[-2:])
        bsf[self._refine_level] = feats[self._refine_level]
        for i in range(self._refine_level + 1, len(feats)):
            bsf[i] = F.interpolate(feats[i], bsf.shape[-2:], mode='nearest')
        return einops.reduce(bsf, 'l b c h w -> b c h w', reduction='mean')

    def _scatter(self, feats: Tuple[torch.Tensor], bsf: torch.Tensor) -> List[torch.Tensor]:
        feats = list(feats)
        for i in range(self._refine_level):
            feats[i] = feats[i] + F.interpolate(bsf, feats[i].shape[-2:], mode='nearest')
        feats[self._refine_level] = feats[self._refine_level] + bsf
        for i in range(self._refine_level + 1, len(feats)):
            feats[i] = feats[i] + F.adaptive_max_pool2d(bsf, feats[i].shape[-2:])
        return feats

    def forward_train(
        self, 
        feats: Tuple[torch.Tensor], 
        class_embeddings: torch.Tensor, 
        class_weights: Optional[torch.Tensor] = None, 
        gt_masks: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, List[torch.Tensor]]]:
        bsf = self._gather(feats)
        bsf, losses = self._refine.forward_train(bsf, class_embeddings, class_weights, gt_masks)
        feats = self._scatter(feats, bsf)
        return feats, losses

    def forward_test(
        self, 
        feats: Tuple[torch.Tensor],
        class_embeddings: torch.Tensor, 
        class_weights: Optional[torch.Tensor] = None, 
    ) -> Tuple[torch.Tensor]:
        bsf = self._gather(feats)
        bsf = self._refine.forward_test(bsf, class_embeddings, class_weights)
        feats = self._scatter(feats, bsf)
        return feats
