import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.runner import BaseModule
from mmdet.models import NECKS, BFP
from mmdet.models.necks.dyhead import DyHeadBlock
from timm.models.layers import DropPath



class BiMultiHeadAttention(BaseModule):
    def __init__(self, v_dim, l_dim, embed_dim, num_heads, dropout=0.1):
        super().__init__(init_cfg=dict(type='Xavier', layer='Conv2d', distribution='uniform'))

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.v_dim = v_dim
        self.l_dim = l_dim

        assert (
                self.head_dim * self.num_heads == self.embed_dim
        ), f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`: {self.num_heads})."
        self.scale = self.head_dim ** (-0.5)
        self.dropout = dropout

        self.v_proj = nn.Linear(self.v_dim, self.embed_dim)
        self.l_proj = nn.Linear(self.l_dim, self.embed_dim)
        self.values_v_proj = nn.Linear(self.v_dim, self.embed_dim)
        self.values_l_proj = nn.Linear(self.l_dim, self.embed_dim)

        self.out_v_proj = nn.Linear(self.embed_dim, self.v_dim)
        self.out_l_proj = nn.Linear(self.embed_dim, self.l_dim)

        self.stable_softmax_2d = False
        self.clamp_min_for_underflow = True
        self.clamp_max_for_overflow = True

        self.init_weights()

    def forward(self, v, l):
        # import ipdb;ipdb.set_trace()
        query_states = einops.rearrange(self.v_proj(v) * self.scale, 'b n (num_heads head_dim) -> (b num_heads) n head_dim', num_heads=self.num_heads, head_dim=self.head_dim)
        key_states = einops.rearrange(self.l_proj(l), 'b c (num_heads head_dim) -> (b num_heads) c head_dim', num_heads=self.num_heads, head_dim=self.head_dim)

        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
        if self.stable_softmax_2d:
            attn_weights = attn_weights - attn_weights.max()
        if self.clamp_min_for_underflow:
            attn_weights = torch.clamp(attn_weights, min=-50000) # Do not increase -50000, data type half has quite limited range
        if self.clamp_max_for_overflow:
            attn_weights = torch.clamp(attn_weights, max=50000) # Do not increase 50000, data type half has quite limited range

        attn_weights_T = attn_weights.transpose(1, 2)
        attn_weights_l = attn_weights_T - torch.max(attn_weights_T, dim=-1, keepdim=True)[0]
        if self.clamp_min_for_underflow:
            attn_weights_l = torch.clamp(attn_weights_l, min=-50000) # Do not increase -50000, data type half has quite limited range
        if self.clamp_max_for_overflow:
            attn_weights_l = torch.clamp(attn_weights_l, max=50000) # Do not increase 50000, data type half has quite limited range

        attn_weights_v = attn_weights.softmax(dim=-1)
        attn_probs_v = F.dropout(attn_weights_v, p=self.dropout, training=self.training)
        value_l_states = einops.rearrange(self.values_l_proj(l), 'b n (num_heads head_dim) -> (b num_heads) n head_dim', num_heads=self.num_heads, head_dim=self.head_dim)
        attn_output_v = torch.bmm(attn_probs_v, value_l_states)
        attn_output_v = einops.rearrange(attn_output_v, '(b num_heads) n head_dim -> b n (num_heads head_dim)', num_heads=self.num_heads, head_dim=self.head_dim)
        attn_output_v = self.out_v_proj(attn_output_v)

        attn_weights_l = attn_weights_l.softmax(dim=-1)
        attn_probs_l = F.dropout(attn_weights_l, p=self.dropout, training=self.training)
        value_v_states = einops.rearrange(self.values_v_proj(v), 'b n (num_heads head_dim) -> (b num_heads) n head_dim', num_heads=self.num_heads, head_dim=self.head_dim)
        attn_output_l = torch.bmm(attn_probs_l, value_v_states)
        attn_output_l = einops.rearrange(attn_output_l, '(b num_heads) n head_dim -> b n (num_heads head_dim)', num_heads=self.num_heads, head_dim=self.head_dim)
        attn_output_l = self.out_l_proj(attn_output_l)

        return attn_output_v, attn_output_l


class VLFuse(BaseModule):
    def __init__(
        self, 
        *args, 
        num_dyconv: int, 
        num_heads: int = 8, 
        v_dim: int = 256,
        embed_dim: int = 256,  # TODO: 2048
        l_dim: int = 64,  # TODO: 512
        dropout: float = 0.1,
        drop_path: float = 0.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.layer_norm_v = nn.LayerNorm(v_dim)
        self.layer_norm_l = nn.LayerNorm(l_dim)
        self.attn = BiMultiHeadAttention(v_dim=v_dim,
                                         l_dim=l_dim,
                                         embed_dim=embed_dim,
                                         num_heads=num_heads,
                                         dropout=dropout,
                                         )

        # add layer scale for training stability
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.gamma_v = nn.Parameter(torch.ones((v_dim)) / num_dyconv, requires_grad=True)
        self.gamma_l = nn.Parameter(torch.ones((l_dim)) / num_dyconv, requires_grad=True)

    def forward(self, visual: torch.Tensor, textual: torch.Tensor):
        h, w = visual.shape[-2:]
        visual = einops.rearrange(visual, 'b c h w -> b (h w) c')
        visual = self.layer_norm_v(visual)
        textual = self.layer_norm_l(textual)
        delta_v, delta_l = self.attn(visual, textual)
        visual = visual + self.drop_path(self.gamma_v * delta_v)
        textual = textual + self.drop_path(self.gamma_l * delta_l)
        visual = einops.rearrange(visual, 'b (h w) c -> b c h w', h=h, w=w)
        return visual, textual


class FuseDyHead(BaseModule):
    def __init__(self, *args, num_layers: int, class_embeddings: str, **kwargs):
        super().__init__(*args, **kwargs)
        class_embeddings = torch.load(class_embeddings)[:, :64].unsqueeze(0)  #[1,1203,512]
        self._class_embeddings = nn.Parameter(class_embeddings.float(), requires_grad=False)

        self._fuse_layers = nn.ModuleList(VLFuse(num_dyconv=num_layers) for _ in range(num_layers))
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
        self.refine_type = 'fuse_dy_head'
        self.refine = FuseDyHead(
            num_layers = 2, 
            class_embeddings = 'data/lvis_v1/prompt/detpro_vild_ViT-B-32.pt'
        )
