from typing import List

import numpy as np

from mmdet.models import DETECTORS, MaskRCNN
import einops
import mmcv
import todd
import torch

from .datasets import LVIS_V1_UNSEEN_866_337


@DETECTORS.register_module()
@todd.distillers.SelfDistiller.wrap()
class Visualizer(MaskRCNN):
    distiller: todd.distillers.SelfDistiller

    def forward_test(
        self, 
        imgs: List[torch.Tensor], 
        img_metas: List[dict], 
        gt_bboxes: List[torch.Tensor],
        gt_labels: List[torch.Tensor],
        **kwargs,
    ):
        tag = [
            label in LVIS_V1_UNSEEN_866_337
            for label in gt_labels[0].tolist()
        ]
        if any(tag):
            img = imgs[0]
            img_meta = img_metas[0]

            self.extract_feat(img.unsqueeze(0))
            img = einops.rearrange(img, 'c h w -> h w c').detach().cpu().numpy()
            img = [mmcv.imdenormalize(
                img,
                mean=img_meta['img_norm_cfg']['mean'],
                std=img_meta['img_norm_cfg']['std'],
                to_bgr=img_meta['img_norm_cfg']['to_rgb'],
            )]

            gt_bboxes = [gt_bbox.detach().cpu().numpy() for gt_bbox in gt_bboxes]

            tag = [np.array(tag, dtype=np.int32)]
            self.distiller.visualize(custom_tensors=dict(
                img=img,
                imgs=[img] * 5,
                bboxes=gt_bboxes,
                labels=tag,
                classes=['seen', 'unseen'],
            ))
        todd.utils.inc_iter()
        return [[
            np.zeros((0, 5), dtype=np.float32)
        ] * len(self.CLASSES)] * len(imgs)
