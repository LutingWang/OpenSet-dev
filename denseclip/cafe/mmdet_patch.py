from typing import Tuple
import torch

from mmdet.core import PRIOR_GENERATORS, AnchorGenerator as _AnchorGenerator
from mmdet.models import DETECTORS, TwoStageDetector as _TwoStageDetector
from mmdet.models.necks.dyhead import DyHeadBlock as _DyHeadBlock
import todd
import todd.reproduction


class DyHeadBlock(_DyHeadBlock):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.spatial_conv_high = None
        self.spatial_conv_low = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward([x])[0]


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

    def with_pos(self, mode: bool = True) -> todd.setattr_temp:
        return todd.setattr_temp(self, '_with_pos', mode)

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
