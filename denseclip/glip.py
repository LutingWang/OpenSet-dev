from typing import Any, Dict, List

import einops
import einops.layers.torch
import todd.datasets
import todd.distillers
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv import ConfigDict
from mmcv.runner import BaseModule
from mmdet.models import DETECTORS, NECKS, BFP, FasterRCNN, MaskRCNN, build_neck
from mmdet.models.necks.dyhead import DyHeadBlock
from timm.models.layers import DropPath

from .datasets import COCO_INDEX_SEEN_48_17, COCO_ALL_48_17, LVIS_V1_SEEN_866_337
from .denseclip import CLIPDistiller


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

        self.layer_norm_v = nn.LayerNorm(v_dim)
        self.v_proj = nn.Linear(v_dim, embed_dim)
        self.values_l_proj = nn.Linear(l_dim, embed_dim)
        self.out_v_proj = nn.Linear(embed_dim, v_dim)
        self.gamma_v = nn.Parameter(torch.ones((v_dim)) / avg_factor, requires_grad=True)

        self.layer_norm_l = nn.LayerNorm(l_dim)
        self.l_proj = nn.Linear(l_dim, embed_dim)
        if bi_direct:
            self.values_v_proj = nn.Linear(v_dim, embed_dim)
            self.out_l_proj = nn.Linear(embed_dim, l_dim)
            self.gamma_l = nn.Parameter(torch.ones((l_dim)) / avg_factor, requires_grad=True)

        self._dropout = dropout
        self._drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self._bi_direct = bi_direct

        self.stable_softmax_2d = False
        self.clamp_min_for_underflow = True
        self.clamp_max_for_overflow = True

    def forward(self, v: torch.Tensor, l: torch.Tensor) -> torch.Tensor:
        h, w = v.shape[-2:]
        v = einops.rearrange(v, 'b c h w -> b (h w) c')
        v = self.layer_norm_v(v)
        l = self.layer_norm_l(l)

        query_states: torch.Tensor = einops.rearrange(
            self.v_proj(v) * self._scale, 
            'b hw (num_heads head_dim) -> (b num_heads) hw head_dim', 
            num_heads=self._num_heads, head_dim=self._head_dim,
        )
        key_states: torch.Tensor = einops.rearrange(
            self.l_proj(l), 
            'b l (num_heads head_dim) -> (b num_heads) l head_dim', 
            num_heads=self._num_heads, head_dim=self._head_dim,
        )

        attn_weights = torch.einsum('b n c, b l c -> b n l', query_states, key_states)
        if self.stable_softmax_2d:
            attn_weights = attn_weights - attn_weights.max()
        attn_weights = torch.clamp(
            attn_weights, 
            min=-50000 if self.clamp_min_for_underflow else None,
            max=50000 if self.clamp_max_for_overflow else None,
        )  # Do not increase 50000, data type half has quite limited range
        attn_weights_v = attn_weights.softmax(dim=-1)
        attn_probs_v = F.dropout(attn_weights_v, p=self._dropout, training=self.training)
        value_l_states = einops.rearrange(self.values_l_proj(l), 'b l (num_heads head_dim) -> (b num_heads) l head_dim', num_heads=self._num_heads, head_dim=self._head_dim)
        attn_output_v = torch.einsum('b n l, b l c -> b n c', attn_probs_v, value_l_states)
        attn_output_v = einops.rearrange(attn_output_v, '(b num_heads) n head_dim -> b n (num_heads head_dim)', num_heads=self._num_heads, head_dim=self._head_dim)
        delta_v = self.out_v_proj(attn_output_v)

        if self._bi_direct:
            attn_weights = einops.rearrange(attn_weights, 'b hw l -> b l hw')
            attn_weights = attn_weights - torch.max(attn_weights, dim=-1, keepdim=True)[0]
            attn_weights = torch.clamp(
                attn_weights, 
                min=-50000 if self.clamp_min_for_underflow else None,
                max=50000 if self.clamp_max_for_overflow else None,
            )  # Do not increase 50000, data type half has quite limited range
            attn_weights_l = attn_weights.softmax(dim=-1)
            attn_probs_l = F.dropout(attn_weights_l, p=self._dropout, training=self.training)
            value_v_states = einops.rearrange(self.values_v_proj(v), 'b hw (num_heads head_dim) -> (b num_heads) hw head_dim', num_heads=self._num_heads, head_dim=self._head_dim)
            attn_output_l = torch.einsum('b l n, b n c -> b l c', attn_probs_l, value_v_states)
            attn_output_l = einops.rearrange(attn_output_l, '(b num_heads) l head_dim -> b l (num_heads head_dim)', num_heads=self._num_heads, head_dim=self._head_dim)
            delta_l = self.out_l_proj(attn_output_l)

        v = v + self._drop_path(self.gamma_v * delta_v)
        v = einops.rearrange(v, 'b (h w) c -> b c h w', h=h, w=w)
        if self._bi_direct:
            l = l + self._drop_path(self.gamma_l * delta_l)
            return v, l
        else:
            return v, None


class DyHead(DyHeadBlock):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.spatial_conv_high = None
        self.spatial_conv_low = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward([x])[0]


class FusionDyHead(BaseModule):
    def __init__(
        self, 
        *args, 
        channels: int, 
        num_layers: int = 6, 
        class_embeddings: str = 'data/lvis_v1/prompt/detpro_ViT-B-32.pt', 
        kappa: int = 35, 
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._channels = channels
        self._num_layers = num_layers
        self._kappa = kappa

        class_embeddings: torch.Tensor = torch.load(class_embeddings, map_location='cpu')
        if class_embeddings.shape[0] == 80:
            class_embeddings = class_embeddings[COCO_ALL_48_17]
            seen_ids = COCO_INDEX_SEEN_48_17
        elif class_embeddings.shape[0] == 1203:
            seen_ids = LVIS_V1_SEEN_866_337
        else:
            raise ValueError(f'Unknown number of classes: {class_embeddings.shape[0]}')
        self._seen_ids = seen_ids
        self._seen_ids_mapper = {c: i for i, c in enumerate(seen_ids)}

        self._class_embeddings = nn.Parameter(class_embeddings.float(), requires_grad=False)
        self._adapter = nn.Sequential(
            DyHead(channels, class_embeddings.shape[1]),
            einops.layers.torch.Reduce('b c h w -> b c', reduction='mean'),
        )

        self._fuse_layers = nn.ModuleList(
            Fusion(
                avg_factor=num_layers, 
                bi_direct=(l != num_layers - 1),
            ) for l in range(num_layers)
        )
        self._dyhead_layers = nn.ModuleList(
            DyHead(channels, channels) for _ in range(num_layers)
        )

    @property
    def class_embeddings(self) -> torch.Tensor:
        if self.training:
            return self._class_embeddings[self._seen_ids]
        return self._class_embeddings

    def forward(self, bsf: torch.Tensor) -> torch.Tensor:
        image_feat = self._adapter(bsf)
        image_feat = F.normalize(image_feat)
        logits = torch.einsum('b d, c d -> b c', image_feat, self.class_embeddings)
        if self.training:
            self.logits: torch.Tensor = logits
        inds = torch.topk(logits, k=self._kappa, dim=-1).indices
        hidden = self._class_embeddings[inds]

        for fuse_layer, dyhead_layer in zip(self._fuse_layers, self._dyhead_layers):
            bsf, hidden = fuse_layer(bsf, hidden)
            bsf = dyhead_layer(bsf)
        assert hidden is None
        return bsf

    def loss(self, labels: List[torch.Tensor]) -> torch.Tensor:
        logits = self.logits
        self.logits = None
        onehot_labels = torch.zeros_like(logits, dtype=bool)
        for i, label in enumerate(labels):
            onehot_labels[i, [self._seen_ids_mapper[l] for l in set(label.tolist())]] = True
        return F.binary_cross_entropy_with_logits(logits, onehot_labels.float())


@NECKS.register_module()
class GLIP(BFP):
    def __init__(self, *args, refine: ConfigDict, **kwargs):
        super().__init__(*args, refine_type=None, **kwargs)
        self.refine_type = 'fuson_dyhead'
        self.refine = FusionDyHead(
            channels=self.in_channels, **refine,
        )
        self.init_weights()

    # def _load_from_state_dict(
    #     self, 
    #     state_dict: Dict[str, torch.Tensor], 
    #     prefix: str, 
    #     local_metadata: dict, 
    #     strict: bool,
    #     missing_keys: List[str], 
    #     unexpected_keys: List[str], 
    #     error_msgs: List[str],
    # ):
    #     if any(k.startswith(prefix) for k in state_dict):
    #         super()._load_from_state_dict(
    #             state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs,
    #         )
    #     else:
    #         self.init_weights()


@todd.distillers.SelfDistiller.wrap()
class GLIPMixin:
    distiller: todd.distillers.SelfDistiller

    def __init__(self, *args, glip_neck: ConfigDict, **kwargs):
        super().__init__(*args, **kwargs)
        self._glip_neck: GLIP = build_neck(glip_neck)

    def extract_feat(self, *args, **kwargs) -> List[torch.Tensor]:
        x = super().extract_feat(*args, **kwargs)
        x = self._glip_neck(x)
        return x

    # def forward_train(self, *args, raw_image: torch.Tensor, **kwargs) -> Dict[str, Any]:
    def forward_train(
        self, 
        img: torch.Tensor, 
        img_metas: List[dict], 
        gt_bboxes: List[torch.Tensor],
        gt_labels: List[torch.Tensor],
        *args, 
        image_embeddings: torch.Tensor, 
        **kwargs,
    ) -> Dict[str, Any]:
        losses = super().forward_train(img, img_metas, gt_bboxes, gt_labels, *args, **kwargs)
        mil_loss = self._glip_neck.refine.loss(gt_labels)
        # clip_image_features = self.distiller.teacher.encode_image(raw_image)
        kd_losses = self.distiller.distill(dict(
            # clip_image_features=clip_image_features,
            clip_image_features=image_embeddings,
        ))
        return {**losses, **dict(loss_mil=mil_loss), **kd_losses}

    def simple_test(self, *args, **kwargs) -> Any:
        results = super().simple_test(*args, **kwargs)
        self.distiller.reset()
        return results


@DETECTORS.register_module()
class GLIPFasterRCNN(GLIPMixin, FasterRCNN):
    pass


@DETECTORS.register_module()
class GLIPMaskRCNN(GLIPMixin, MaskRCNN):
    pass
