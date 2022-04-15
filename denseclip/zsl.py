from typing import Any, Tuple

import todd
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.models import DETECTORS, RetinaHead, Shared2FCBBoxHead

from .datasets import SEEN_48_17, ALL_48_17, CocoGZSLDataset
from .prompt import Classifier


@DETECTORS.register_module()
class RetinaHeadZSL(RetinaHead):
    def __init__(self, *args, class_embeddings: str, **kwargs):
        super().__init__(*args, **kwargs, init_cfg=dict(
            type='Normal', layer='Conv2d', std=0.01,
        ))
        class_embeddings: torch.Tensor = torch.load(class_embeddings, map_location='cpu')
        self._seen_class_embeddings = nn.Parameter(class_embeddings[SEEN_48_17], requires_grad=False)
        self._all_class_embeddings = nn.Parameter(class_embeddings[ALL_48_17], requires_grad=False)
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


@DETECTORS.register_module()
class Shared2FCBBoxHeadZSL(Shared2FCBBoxHead):
    def __init__(self, *args, class_embeddings: str, **kwargs):
        super().__init__(*args, **kwargs)
        class_embeddings: torch.Tensor = torch.load(class_embeddings, map_location='cpu')
        self._seen_class_embeddings = nn.Parameter(class_embeddings[SEEN_48_17], requires_grad=False)
        self._all_class_embeddings = nn.Parameter(class_embeddings[ALL_48_17], requires_grad=False)
        self._bg_class_embeddings = nn.Parameter(torch.zeros_like(class_embeddings[[0]]), requires_grad=True)
        self.fc_cls = Classifier()  # TODO: bias need to be specified as -4.18

    def forward(self, *args, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        class_embeddings = self._seen_class_embeddings if self.training else self._all_class_embeddings
        class_embeddings = torch.cat([class_embeddings, F.normalize(self._bg_class_embeddings)], dim=0)
        self.fc_cls.set_weight(class_embeddings, norm=False)
        return super().forward(*args, **kwargs)
