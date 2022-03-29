from typing import Any, List, Tuple

import clip.model
import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.models.builder import BACKBONES


class AttentionPool2d(clip.model.AttentionPool2d):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3]).permute(2, 0, 1)  # NCHW -> (HW)NC
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
        x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC
        x, _ = F.multi_head_attention_forward(
            query=x, key=x, value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False
        )

        return x


class CLIPResNet(clip.model.ModifiedResNet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.layer1.register_forward_hook(self._layer_forward_hook)
        self.layer2.register_forward_hook(self._layer_forward_hook)
        self.layer3.register_forward_hook(self._layer_forward_hook)
        self.layer4.register_forward_hook(self._layer_forward_hook)

    def _layer_forward_hook(self, module: nn.Module, input_: Any, output: torch.Tensor):
        self._outs.append(output)

    def forward(self, x: torch.Tensor) -> Tuple[List[torch.Tensor], Any]:
        self._outs = []
        x = super().forward(x)
        return self._outs, x


@BACKBONES.register_module()
class CLIPResNetWithoutAttention(CLIPResNet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.attnpool = nn.Identity()

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        outs, _ = super().forward(x)
        return outs


@BACKBONES.register_module()
class CLIPResNetWithAttention(CLIPResNet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.attnpool.__class__ = AttentionPool2d


class ContextDecoder(nn.TransformerDecoder):
    def __init__(
        self,
        in_features=1024,
        hidden_features=256,
        num_heads=4,
        num_layers=6,
        dropout=0.1,
    ):
        decoder_layer = nn.TransformerDecoderLayer(
            hidden_features, nhead=num_heads, 
            dim_feedforward=hidden_features * 4, 
            dropout=dropout, activation='gelu',
            # NOTE: compat pytorch 1.8.0
            # batch_first=True,
        )
        super().__init__(
            decoder_layer, num_layers, 
            norm=nn.LayerNorm(hidden_features),
        )
        self._visual_proj = nn.Sequential(
            nn.LayerNorm(in_features),
            nn.Linear(in_features, hidden_features),
            nn.LayerNorm(hidden_features),
        )
        self._textual_proj = nn.Sequential(
            nn.LayerNorm(in_features),
            nn.Linear(in_features, hidden_features),
            nn.LayerNorm(hidden_features),
        )
        self._out_proj = nn.Linear(hidden_features, in_features)

    def init_weights(self):

        def initialize(module):
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=.02)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.bias, 0)
                nn.init.constant_(module.weight, 1.0)

        self.apply(initialize)

    def forward(self, text: torch.Tensor, visual: torch.Tensor) -> torch.Tensor:
        visual = self._visual_proj(visual)
        text = self._textual_proj(text)
        visual = einops.rearrange(visual, 'b n c -> n b c')
        text = einops.rearrange(text, 'b n c -> n b c')
        x = super().forward(text, visual)
        x = einops.rearrange(x, 'n b c -> b n c')
        return self._out_proj(x)
