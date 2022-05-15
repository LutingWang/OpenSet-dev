from typing import Optional

import einops
import einops.layers.torch
import torch
import torch.nn as nn
from mmcv.runner import BaseModule


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
