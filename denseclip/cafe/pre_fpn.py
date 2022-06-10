from typing import List, Optional, Tuple

import einops
import einops.layers.torch
import todd.reproduction
import torch
import torch.nn as nn
from mmcv.runner import BaseModule, ModuleList
from todd import Registry


PRE_FPNS = Registry('pre fpns')


@PRE_FPNS.register_module()
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

    def forward(
        self, 
        v: torch.Tensor, 
        l: torch.Tensor, 
        *,
        v_weights: Optional[torch.Tensor] = None,
        l_weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        v_feats = self._v_proj(v)
        if l.ndim > 2:
            b, *_, d = l.shape
            l = l.view(-1, d)
        else:
            b = 1
        l_feats = self._l_proj(l)
        l_feats = einops.reduce(l_feats, '(b c) d -> b d 1 1', b=b, reduction='mean')
        v_feats = self._out_v_proj(v_feats * l_feats)
        # v_feats = F.normalize(v + v_feats)
        return v + v_feats


class PreFPN(BaseModule):
    def __init__(
        self, 
        *args, 
        channels: List[int], 
        embedding_dim: int,
        hidden_dim: int,
        type: str,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._refines = ModuleList([
            PRE_FPNS.build(
                dict(
                    type=type,
                    v_dim=channel, 
                    l_dim=embedding_dim, 
                    hidden_dim=hidden_dim,
                ),
            ) for channel in channels
        ])

    @todd.reproduction.set_seed_temp('PreFPN')
    def init_weights(self):
        return super().init_weights()

    def forward(
        self, 
        x: Tuple[torch.Tensor], 
        class_embeddings: torch.Tensor, 
        class_weights: torch.Tensor,
    ):
        x = tuple(
            refine(feat, class_embeddings, l_weights=class_weights) 
            for refine, feat in zip(self._refines, x)
        )
        return x
