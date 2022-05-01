import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.runner import BaseModule
from mmdet.models import NECKS, BFP
from mmdet.models.necks.dyhead import DyHeadBlock
from timm.models.layers import DropPath


class Fusion(BaseModule):
    def __init__(
        self, 
        *args, 
        num_dyconv: int,
        v_dim: int = 256, 
        l_dim: int = 64,  # TODO: 512
        embed_dim: int = 256,  # TODO: 2048
        num_heads: int = 8,
        dropout: float = 0.1, 
        drop_path: float = 0.0,
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

        self._head_dim = head_dim
        self._num_heads = num_heads
        self._scale = head_dim ** (-0.5)
        self._dropout = dropout

        self.layer_norm_v = nn.LayerNorm(v_dim)
        self.v_proj = nn.Linear(v_dim, embed_dim)
        self.values_v_proj = nn.Linear(v_dim, embed_dim)
        self.out_v_proj = nn.Linear(embed_dim, v_dim)

        self.layer_norm_l = nn.LayerNorm(l_dim)
        self.l_proj = nn.Linear(l_dim, embed_dim)
        self.values_l_proj = nn.Linear(l_dim, embed_dim)
        self.out_l_proj = nn.Linear(embed_dim, l_dim)

        self.stable_softmax_2d = False
        self.clamp_min_for_underflow = True
        self.clamp_max_for_overflow = True

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.gamma_v = nn.Parameter(torch.ones((v_dim)) / num_dyconv, requires_grad=True)
        self.gamma_l = nn.Parameter(torch.ones((l_dim)) / num_dyconv, requires_grad=True)

        self.init_weights()

    def forward(self, v: torch.Tensor, l: torch.Tensor) -> torch.Tensor:
        h, w = v.shape[-2:]
        v = einops.rearrange(v, 'b c h w -> b (h w) c')
        v = self.layer_norm_v(v)
        l = self.layer_norm_l(l)

        query_states: torch.Tensor = einops.rearrange(
            self.v_proj(v) * self._scale, 
            'b n (num_heads head_dim) -> (b num_heads) n head_dim', 
            num_heads=self._num_heads, head_dim=self._head_dim,
        )
        key_states: torch.Tensor = einops.rearrange(
            self.l_proj(l), 
            'b c (num_heads head_dim) -> (b num_heads) c head_dim', 
            num_heads=self._num_heads, head_dim=self._head_dim,
        )

        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
        if self.stable_softmax_2d:
            attn_weights = attn_weights - attn_weights.max()
        attn_weights = torch.clamp(
            attn_weights, 
            min=-50000 if self.clamp_min_for_underflow else None,
            max=50000 if self.clamp_max_for_overflow else None,
        )  # Do not increase 50000, data type half has quite limited range
        attn_weights_v = attn_weights.softmax(dim=-1)
        attn_probs_v = F.dropout(attn_weights_v, p=self._dropout, training=self.training)
        value_l_states = einops.rearrange(self.values_l_proj(l), 'b n (num_heads head_dim) -> (b num_heads) n head_dim', num_heads=self._num_heads, head_dim=self._head_dim)
        attn_output_v = torch.bmm(attn_probs_v, value_l_states)
        attn_output_v = einops.rearrange(attn_output_v, '(b num_heads) n head_dim -> b n (num_heads head_dim)', num_heads=self._num_heads, head_dim=self._head_dim)
        delta_v = self.out_v_proj(attn_output_v)

        attn_weights = attn_weights.transpose(1, 2)
        attn_weights = attn_weights - torch.max(attn_weights, dim=-1, keepdim=True)[0]
        attn_weights = torch.clamp(
            attn_weights, 
            min=-50000 if self.clamp_min_for_underflow else None,
            max=50000 if self.clamp_max_for_overflow else None,
        )  # Do not increase 50000, data type half has quite limited range
        attn_weights_l = attn_weights.softmax(dim=-1)
        attn_probs_l = F.dropout(attn_weights_l, p=self._dropout, training=self.training)
        value_v_states = einops.rearrange(self.values_v_proj(v), 'b n (num_heads head_dim) -> (b num_heads) n head_dim', num_heads=self._num_heads, head_dim=self._head_dim)
        attn_output_l = torch.bmm(attn_probs_l, value_v_states)
        attn_output_l = einops.rearrange(attn_output_l, '(b num_heads) n head_dim -> b n (num_heads head_dim)', num_heads=self._num_heads, head_dim=self._head_dim)
        delta_l = self.out_l_proj(attn_output_l)

        v = v + self.drop_path(self.gamma_v * delta_v)
        l = l + self.drop_path(self.gamma_l * delta_l)
        v = einops.rearrange(v, 'b (h w) c -> b c h w', h=h, w=w)
        return v, l


class FusionDyHead(BaseModule):
    def __init__(self, *args, num_layers: int, class_embeddings: str, **kwargs):
        super().__init__(*args, **kwargs)
        class_embeddings = torch.load(class_embeddings)[:, :64].unsqueeze(0)  # TODO: [1,1203,512]
        self._class_embeddings = nn.Parameter(class_embeddings.float(), requires_grad=False)

        self._fuse_layers = nn.ModuleList(Fusion(num_dyconv=num_layers) for _ in range(num_layers))
        self._dyhead_layers = nn.ModuleList(DyHeadBlock(256, 256) for _ in range(num_layers))

    def forward(self, bsf: torch.Tensor) -> torch.Tensor:
        hidden = self._class_embeddings.repeat((bsf.shape[0], 1, 1))
        for fuse_layer, dyhead_layer in zip(self._fuse_layers, self._dyhead_layers):
            bsf, hidden = fuse_layer(bsf, hidden)
            bsf = dyhead_layer([bsf])[0]
        return bsf


@NECKS.register_module()
class GLIP(BFP):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, refine_type=None, **kwargs)
        self.refine_type = 'fuson_dyhead'
        self.refine = FusionDyHead(
            num_layers = 2, 
            class_embeddings = 'data/lvis_v1/prompt/detpro_vild_ViT-B-32.pt'
        )
