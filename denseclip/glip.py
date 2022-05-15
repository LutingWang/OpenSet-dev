from typing import List, Optional, Tuple

import einops
import einops.layers.torch
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.runner import BaseModule, ModuleList
from timm.models.layers import DropPath

from .mmdet_patch import DyHeadBlock


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

    def forward(
        self, 
        v: torch.Tensor, 
        l: torch.Tensor, 
        l_weights: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
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
        masks = einops.reduce(
            attn_weights, '(b num_heads) (h w) l -> b l h w', 
            num_heads=self._num_heads, h=h, w=w, reduction='mean',
        ) if self.training else None
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
            return v, l, masks
        else:
            return v, None, masks


class GLIP(BaseModule):
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

    def _forward(
        self, 
        bsf: torch.Tensor, 
        class_embeddings: torch.Tensor, 
        logits_weight: Optional[torch.Tensor] = None,
    ):
        multi_layer_masks = []
        for fuse_layer, dyhead_layer in zip(self._fuse_layers, self._dyhead_layers):
            bsf, class_embeddings, masks = fuse_layer(
                bsf, class_embeddings, logits_weight,
            )
            multi_layer_masks.append(masks)
            bsf = dyhead_layer(bsf)
        assert class_embeddings is None
        return bsf, multi_layer_masks

    def forward(self, *args, **kwargs):
        bsf, multi_layer_masks = self._forward(*args, **kwargs)
        if self.training:
            return bsf, multi_layer_masks
        assert all(masks is None for masks in multi_layer_masks)
        return bsf
