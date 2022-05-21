from typing import Any, Dict, List, Optional, Tuple, Union, cast

import todd
import todd.distillers
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv import ConfigDict
from mmdet.core import bbox2roi, MaxIoUAssigner
from mmdet.models import DETECTORS, HEADS, LOSSES, RetinaHead, ConvFCBBoxHead, StandardRoIHead, KnowledgeDistillationKLDivLoss

from ..datasets import COCO_INDEX_SEEN_48_17, COCO_ALL_48_17, CocoGZSLDataset, LVIS_V1_SEEN_866_337
from ..cafe import Classifier


@DETECTORS.register_module()
class RetinaHeadZSL(RetinaHead):
    def __init__(self, *args, class_embeddings: str, **kwargs):
        super().__init__(*args, **kwargs, init_cfg=dict(
            type='Normal', layer='Conv2d', std=0.01,
        ))
        class_embeddings: torch.Tensor = torch.load(class_embeddings, map_location='cpu')
        self._seen_class_embeddings = nn.Parameter(class_embeddings[COCO_SEEN_48_17], requires_grad=False)
        self._all_class_embeddings = nn.Parameter(class_embeddings[COCO_ALL_48_17], requires_grad=False)
        self._classifier = Classifier(bias=-4.18)  # TODO: bias need to be specified as -4.18
        self.retina_cls = nn.Conv2d(
            self.feat_channels, 
            class_embeddings.shape[1],  # NOTE: assumes single anchor
            kernel_size=3, padding=1,
            bias=False,
        )

    def forward_single(self, *args, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        cls_score, bbox_pred = super().forward_single(*args, **kwargs)
        self._classifier.set_weight(
            self._seen_class_embeddings if self.training else self._all_class_embeddings, 
            norm=False,
        )
        cls_score = self._classifier(cls_score)
        return cls_score, bbox_pred

    def simple_test(self, *args, **kwargs) -> Any:
        num_classes = len(CocoGZSLDataset.CLASSES)
        cls_out_channels = self.cls_out_channels - self.num_classes + num_classes
        with todd.utils.setattr_temp(self, 'num_classes', num_classes):
            with todd.utils.setattr_temp(self, 'cls_out_channels', cls_out_channels):
                return super().simple_test(*args, **kwargs)

    def aug_test(self, *args, **kwargs) -> Any:
        num_classes = len(CocoGZSLDataset.CLASSES)
        cls_out_channels = self.cls_out_channels - self.num_classes + num_classes
        with todd.utils.setattr_temp(self, 'num_classes', num_classes):
            with todd.utils.setattr_temp(self, 'cls_out_channels', cls_out_channels):
                return super().aug_test(*args, **kwargs)


@LOSSES.register_module()
class KLDivLossZSL(KnowledgeDistillationKLDivLoss):
    def set_valid_ids(self, valid_ids: List[int]):
        self._valid_ids = valid_ids

    def set_class_similarities(self, class_similarities: torch.Tensor):
        # assert not hasattr('_class_similarities', self)
        # class_similarities: torch.Tensor = class_embeddings @ class_embeddings.T
        # class_similarities = class_similarities.softmax(dim=-1)
        # class_similarities = torch.cat([class_similarities, torch.zeros_like(class_similarities[[0]])], dim=0)
        # class_similarities = torch.cat([class_similarities, torch.zeros_like(class_similarities[:, [0]])], dim=-1)
        # class_similarities[-1][-1] = 1
        # self._class_similarities = nn.Parameter(
        #     class_similarities, requires_grad=False,
        # )
        self._class_similarities = class_similarities[:, self._valid_ids]

    def forward(
        self,
        cls_score: torch.Tensor,
        label: torch.Tensor,
        weight: Optional[torch.Tensor] = None,
        avg_factor: Optional[int] = None,
        reduction_override: Optional[str] = None,
        **kwargs,
    ) -> torch.Tensor:
        pred = cls_score[:, self._valid_ids]
        soft_label = self._class_similarities[label]
        return super().forward(
            pred, soft_label, weight, avg_factor, reduction_override,
        )

