from abc import abstractmethod
from typing import Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmcv import ConfigDict
from mmdet.models import BFP as _BFP
from mmdet.models import ResNet as _ResNet
from mmdet.models import TwoStageDetector as _TwoStageDetector
from mmdet.models import RPNHead, BaseRoIHead
from mmdet.models.builder import DETECTORS
from mmdet.models.necks.dyhead import DyHeadBlock as _DyHeadBlock
import todd.utils


class DyHeadBlock(_DyHeadBlock):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.spatial_conv_high = None
        self.spatial_conv_low = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward([x])[0]


class BFP(_BFP):
    def forward(self, inputs: Tuple[torch.Tensor], *args):
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
        bsf = self.refine(bsf, *args)
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


class ResNet(_ResNet):
    def __init__(self, *args, custom_plugins: Optional[ConfigDict] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self._custom_plugins = (
            None if custom_plugins is None else
            self._make_custom_plugins(**custom_plugins)
        )

    @abstractmethod
    def _make_custom_plugins(self, *args, **kwargs) -> nn.Module:
        pass

    def forward(self, x: torch.Tensor, *args) -> torch.Tensor:
        if self.deep_stem:
            x = self.stem(x)
        else:
            x = self.conv1(x)
            x = self.norm1(x)
            x = self.relu(x)
        x = self.maxpool(x)
        outs = []
        for i, layer_name in enumerate(self.res_layers):
            res_layer = getattr(self, layer_name)
            x = res_layer(x)
            if self._custom_plugins is not None:
                plugin = self._custom_plugins[i]
                x = plugin(x, *args)
            if i in self.out_indices:
                outs.append(x)
        return tuple(outs)


@DETECTORS.register_module(force=True)
class TwoStageDetector(_TwoStageDetector):
    rpn_head: RPNHead
    roi_head: BaseRoIHead

    def __init__(self, *args, freeze_neck: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        if freeze_neck:
            todd.utils.freeze_model(self.neck)
