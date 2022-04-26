from typing import List, Union

import torch
import torch.nn as nn
from mmcv.runner import BaseModule
from mmdet.core import bbox2result
from mmdet.models import DETECTORS

from .prompt import Classifier


@DETECTORS.register_module()
class CLIPDetector(BaseModule):
    def __init__(self, *args, class_embeddings: Union[str, torch.Tensor], **kwargs):
        super().__init__()
        if isinstance(class_embeddings, str):
            class_embeddings = torch.load(class_embeddings, map_location='cpu')
        self._num_classes = class_embeddings.shape[0]

        class_embeddings = class_embeddings.float()
        class_embeddings = nn.Parameter(
            class_embeddings.float(), 
            # requires_grad=False,
        )
        classifier = Classifier(0.01)
        classifier.set_weight(class_embeddings, norm=False)
        self._classifier = classifier
    
    def forward(self, *args, bboxes: List[torch.Tensor], bbox_embeddings: List[torch.Tensor], **kwargs):
        bbox_results = []
        for bbox, bbox_embedding in zip(bboxes, bbox_embeddings):
            cls_: torch.Tensor = self._classifier(bbox_embedding, norm=False)
            score, label = cls_.max(-1, keepdim=True)
            bbox = torch.cat([bbox, score], dim=-1)
            bbox_result = bbox2result(bbox, label.flatten(), self._num_classes)
            bbox_results.append(bbox_result)
        return bbox_results
