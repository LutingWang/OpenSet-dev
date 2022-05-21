import numbers
import os
from typing import Any, List, Optional, Tuple

import clip.model
import einops
import todd.reproduction
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv import ConfigDict
from mmcv.ops import batched_nms
from mmcv.runner import BaseModule
from mmdet.core import DistancePointBBoxCoder
from mmdet.models import BACKBONES, HEADS, RetinaHead, RPNHead

# from .cafe.mmdet_patch import AnchorGenerator
from ..utils import has_debug_flag


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
    def __init__(self, *args, frozen_stages: int = 1, norm_eval: bool = True, **kwargs):
        super().__init__(*args, **kwargs)
        self._frozen_stages = frozen_stages
        self._norm_eval = norm_eval

        self.layer1.register_forward_hook(self._layer_forward_hook)
        self.layer2.register_forward_hook(self._layer_forward_hook)
        self.layer3.register_forward_hook(self._layer_forward_hook)
        self.layer4.register_forward_hook(self._layer_forward_hook)
        self._freeze_stages()
        self._eval_norms()

    def _layer_forward_hook(self, module: nn.Module, input_: Any, output: torch.Tensor):
        self._outs.append(output)

    def _freeze_stages(self):
        if self._frozen_stages == 0:
            return
        for i in range(3):
            todd.reproduction.freeze_model(getattr(self, f'conv{i + 1}'))
            todd.reproduction.freeze_model(getattr(self, f'bn{i + 1}'))
        for i in range(self._frozen_stages):
            todd.reproduction.freeze_model(getattr(self, f'layer{i + 1}'))

    def _eval_norms(self):
        if not self.training or not self._norm_eval:
            return
        for module in self.modules():
            # trick: eval have effect on BatchNorm only
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.eval()

    def forward(self, x: torch.Tensor) -> Tuple[List[torch.Tensor], Any]:
        self._outs = []
        x = super().forward(x)
        return self._outs, x

    def train(self, mode: bool = True):
        super().train(mode)
        self._freeze_stages()
        self._eval_norms()


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


class RPNHeadWithPos(RPNHead):
    prior_generator: AnchorGenerator

    def _bbox_post_process(
        self, 
        mlvl_scores: List[torch.Tensor], 
        mlvl_bboxes: List[torch.Tensor], 
        mlvl_valid_anchors: List[torch.Tensor],
        level_ids: int, 
        cfg: ConfigDict, 
        img_shape: Tuple[int], 
        **kwargs,
    ):
        scores = torch.cat(mlvl_scores)
        anchors = torch.cat(mlvl_valid_anchors)
        rpn_bbox_pred = torch.cat(mlvl_bboxes)

        if hasattr(self.prior_generator, '_with_pos') and self.prior_generator._with_pos:
            if isinstance(self.bbox_coder, DistancePointBBoxCoder):
                poses = anchors[:, 2:]
                anchors = anchors[:, :2]
            else:
                poses = anchors[:, 4:]
                anchors = anchors[:, :4]
        else:
            poses = None

        proposals = self.bbox_coder.decode(
            anchors, rpn_bbox_pred, max_shape=img_shape)
        ids = torch.cat(level_ids)

        if cfg.min_bbox_size >= 0:
            w = proposals[:, 2] - proposals[:, 0]
            h = proposals[:, 3] - proposals[:, 1]
            valid_mask = (w > cfg.min_bbox_size) & (h > cfg.min_bbox_size)
            if not valid_mask.all():
                proposals = proposals[valid_mask]
                scores = scores[valid_mask]
                ids = ids[valid_mask]
                if poses is not None:
                    poses = poses[valid_mask]

        if proposals.numel() > 0:
            dets, keep = batched_nms(proposals, scores, ids, cfg.nms)
            if poses is not None:
                dets = torch.cat([dets, poses[keep]], dim=-1)
        else:
            return proposals.new_zeros(0, 9)

        return dets[:cfg.max_per_img]


@HEADS.register_module()
class RetinaRPNHead(RetinaHead):
    prior_generator: AnchorGenerator

    def forward_train(self, x, img_metas, gt_bboxes, gt_labels=None, gt_bboxes_ignore=None, proposal_cfg=None, **kwargs):
        proposal_cfg = ConfigDict(
            nms_pre=2000,
            max_per_img=2 if has_debug_flag(3) else 8,
            nms=dict(type='nms', iou_threshold=0.4),
            min_bbox_size=224,
        )
        return super().forward_train(x, img_metas, gt_bboxes, gt_labels, gt_bboxes_ignore, proposal_cfg, **kwargs)

    def get_bboxes(self, cls_scores: List[torch.Tensor], *args, **kwargs):
        if not self.training:
            return super().get_bboxes(cls_scores, *args, **kwargs)
        cls_scores = [einops.reduce(
            cls_score,
            'b k h w -> b 1 h w',
            reduction='max',
        ) for cls_score in cls_scores]
        with self.prior_generator.with_pos():
            return super().get_bboxes(cls_scores, *args, **kwargs)

    def _get_bboxes_single(self, *args, **kwargs):
        if self.training:
            return RPNHead._get_bboxes_single(self, *args, **kwargs)
        return super()._get_bboxes_single(*args, **kwargs)

    def _bbox_post_process(self, *args, **kwargs):
        if self.training:
            return RPNHeadWithPos._bbox_post_process(self, *args, **kwargs)
        return super()._bbox_post_process(*args, **kwargs)
