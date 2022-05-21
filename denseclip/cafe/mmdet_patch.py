from abc import abstractmethod
from typing import List, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.batchnorm import _BatchNorm

from mmcv import ConfigDict
from mmdet.core import PRIOR_GENERATORS, AnchorGenerator as _AnchorGenerator
from mmdet.models import DETECTORS, BFP as _BFP, ResNet as _ResNet, TwoStageDetector as _TwoStageDetector, RPNHead, BaseRoIHead
from mmdet.models.necks.dyhead import DyHeadBlock as _DyHeadBlock
import todd.reproduction


class DyHeadBlock(_DyHeadBlock):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.spatial_conv_high = None
        self.spatial_conv_low = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward([x])[0]


class BFP(_BFP):
    def forward(self, inputs: Tuple[torch.Tensor], *args, **kwargs):
        assert len(inputs) == self.num_levels

        # step 1: gather multi-level features by resize and average
        feats = []
        gather_size = inputs[self.refine_level].size()[2:]
        for i in range(self.num_levels):
            if i < self.refine_level:
                gathered = F.adaptive_max_pool2d(
                    inputs[i], output_size=gather_size)
            else:
                gathered = F.interpolate(
                    inputs[i], size=gather_size, mode='nearest')
            feats.append(gathered)

        bsf = sum(feats) / len(feats)

        # step 2: refine gathered features
        assert self.refine_type is not None
        bsf = self.refine(bsf, *args, **kwargs)
        if isinstance(bsf, tuple):
            args = bsf[1:]
            bsf = bsf[0]
        else:
            args = None

        # step 3: scatter refined features to multi-levels by a residual path
        outs = []
        for i in range(self.num_levels):
            out_size = inputs[i].size()[2:]
            if i < self.refine_level:
                residual = F.interpolate(bsf, size=out_size, mode='nearest')
            else:
                residual = F.adaptive_max_pool2d(bsf, output_size=out_size)
            outs.append(residual + inputs[i])

        if args is None:
            return tuple(outs)
        else:
            return tuple(outs), args


# class ResNet(_ResNet):
#     def __init__(self, *args, custom_plugins: Optional[ConfigDict] = None, **kwargs):
#         super().__init__(*args, **kwargs)
#         self._custom_plugins = (
#             None if custom_plugins is None else
#             self._make_custom_plugins(**custom_plugins)
#         )

#     @abstractmethod
#     def _make_custom_plugins(self, *args, **kwargs) -> nn.Module:
#         pass

#     def forward(self, x: torch.Tensor, *args) -> torch.Tensor:
#         if self.deep_stem:
#             x = self.stem(x)
#         else:
#             x = self.conv1(x)
#             x = self.norm1(x)
#             x = self.relu(x)
#         x = self.maxpool(x)
#         outs = []
#         for i, layer_name in enumerate(self.res_layers):
#             res_layer = getattr(self, layer_name)
#             x = res_layer(x)
#             if self._custom_plugins is not None:
#                 plugin = self._custom_plugins[i]
#                 x = plugin(x, *args)
#             if i in self.out_indices:
#                 outs.append(x)
#         return tuple(outs)


@DETECTORS.register_module(force=True)
class TwoStageDetector(
    todd.reproduction.FrozenMixin,
    _TwoStageDetector,
):
    pass


@PRIOR_GENERATORS.register_module(force=True)
class AnchorGenerator(_AnchorGenerator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._with_pos = False

    def with_pos(self, mode: bool = True) -> todd.utils.setattr_temp:
        return todd.utils.setattr_temp(self, '_with_pos', mode)

    # def grid_anchors(
    #     self, 
    #     featmap_sizes: Tuple[int], 
    #     device: str = 'cuda',
    # ) -> List[torch.Tensor]:
    #     anchors: List[torch.Tensor] = super().grid_anchors(featmap_sizes, device)
    #     if self._with_pos:
    #         self.anchors = [
    #             anchor.reshape(featmap_size + (8,))
    #             for featmap_size, anchor in zip(featmap_sizes, anchors)
    #         ]
    #     return anchors

    def single_level_grid_priors(
        self,
        featmap_size: Tuple[int],
        level_idx: int,
        dtype: torch.dtype = torch.float32,
        device: str = 'cuda',
    ) -> torch.Tensor:
        if not self._with_pos:
            return super().single_level_grid_priors(featmap_size, level_idx, dtype, device)
        base_anchors = self.base_anchors[level_idx].to(device).to(dtype)
        feat_h, feat_w = featmap_size
        stride_w, stride_h = self.strides[level_idx]
        shift_x = torch.arange(0, feat_w, device=device).to(dtype) * stride_w
        shift_y = torch.arange(0, feat_h, device=device).to(dtype) * stride_h

        shift_xx, shift_yy = self._meshgrid(shift_x, shift_y)
        shifts = torch.stack([shift_xx, shift_yy, shift_xx, shift_yy], dim=-1)

        all_anchors: torch.Tensor = base_anchors[None, :, :] + shifts[:, None, :]

        all_pos = torch.zeros_like(all_anchors)
        # assert all_pos.shape[1] == self.num_base_anchors[level_idx]
        all_pos[..., 0] = level_idx
        all_pos[..., 1] = shift_yy[:, None] // stride_h
        all_pos[..., 2] = shift_xx[:, None] // stride_w
        all_pos[..., 3] = torch.arange(self.num_base_anchors[level_idx], device=device).unsqueeze(0)
        # for i in range(self.num_base_anchors[level_idx]):
        #     all_pos[:, i, 3] = i

        all_anchors = torch.cat((all_anchors, all_pos), dim=-1)
        all_anchors = all_anchors.view(-1, 8)
        return all_anchors
