from typing import Dict, List, Tuple

import clip
import clip.model
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv import ConfigDict
from mmdet.core import AssignResult, MaxIoUAssigner, bbox2result, multiclass_nms
from mmdet.models import DETECTORS, BaseDetector

from .datasets import COCO_SEEN_48_17, LVIS_V1_SEEN_866_337, LVISV1GZSLDataset
from .denseclip import CLIPDistiller
from .model import Classifier


@DETECTORS.register_module()
@CLIPDistiller.wrap()
class PromptTrainer(BaseDetector):
    CLASSES: Tuple[str]
    distiller: CLIPDistiller
    
    def __init__(self, *args, num_classes: int, test_cfg: ConfigDict, **kwargs):
        self._num_classes = num_classes
        self._test_cfg = test_cfg

        if num_classes == 80:
            seen_ids = COCO_SEEN_48_17
            assert NotImplementedError
        elif num_classes == 1203:
            seen_ids = LVIS_V1_SEEN_866_337
            self.CLASSES = LVISV1GZSLDataset.CLASSES
        else:
            raise ValueError(f'Unknown number of classes: {num_classes}')
        self._seen_ids = seen_ids
        self._unseen_ids = [
            i for i in range(num_classes) 
            if i not in seen_ids
        ]

        super().__init__()

        self._bbox_assigner = MaxIoUAssigner(0.5, 0.4, match_low_quality=False)
        self._classifier = Classifier(tau=(0.007, 0.01))
    
    def _init_with_distiller(self):
        teacher_cfg = self.distiller._teacher_cfg
        tokens = clip.tokenize(self.CLASSES, context_length=teacher_cfg.context_length - teacher_cfg.prompt_length)
        tokens = torch.cat([
            tokens[:, [0]], 
            tokens.new_zeros(self._num_classes, teacher_cfg.prompt_length), 
            tokens[:, 1:],
        ], dim=-1)
        self._tokens = nn.Parameter(tokens, requires_grad=False)

    def init_weights(self):
        print('init weights')

    def extract_feat(self, bbox_embeddings: torch.Tensor) -> torch.Tensor:
        label_embeddings = self.distiller.teacher.encode_text(self._tokens)
        self._classifier.set_weight(label_embeddings, norm=True)
        logits = self._classifier(bbox_embeddings, norm=False)
        if self.training:
            logits[..., self._unseen_ids] = float('-inf')
        return logits

    def simple_test(
        self, 
        img: torch.Tensor,
        img_metas: List[dict],
        bboxes: List[torch.Tensor],
        bbox_embeddings: List[torch.Tensor],
        **kwargs,
    ) -> list:
        bbox_results = []
        for bbox, bbox_embedding in zip(bboxes, bbox_embeddings):
            logit = self.extract_feat(bbox_embedding).softmax(-1)
            bbox, label = multiclass_nms(
                bbox, logit, self._test_cfg.score_thr, self._test_cfg.nms, self._test_cfg.max_per_img,
            )
            bbox_result = bbox2result(bbox, label, self._num_classes)
            bbox_results.append(bbox_result)
        return bbox_results

    def aug_test(self, *args, **kwargs):
        raise NotImplementedError

    def forward_train(
        self, 
        imgs: torch.Tensor,
        img_metas: List[dict],
        gt_bboxes: List[torch.Tensor],
        gt_labels: List[torch.Tensor],
        bboxes: List[torch.Tensor],
        bbox_embeddings: List[torch.Tensor],
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        labels: List[torch.Tensor] = []
        for bbox, gt_bbox, gt_label in zip(bboxes, gt_bboxes, gt_labels):
            assign_result: AssignResult = self._bbox_assigner.assign(
                bbox, gt_bbox, gt_bboxes_ignore=None, gt_labels=gt_label,
            )
            labels.append(assign_result.labels)
        bbox_embeddings = torch.cat(bbox_embeddings)
        logits = self.extract_feat(bbox_embeddings)

        labels = torch.cat(labels)
        pos_inds = labels >= 0
        loss_pos = F.cross_entropy(logits[pos_inds], labels[pos_inds], reduction="sum")
        neg_inds = labels == -1
        loss_neg = -F.log_softmax(logits[neg_inds][:, self._seen_ids], dim=1).sum() / len(self.CLASSES)
        return dict(loss_pos=loss_pos / len(img_metas), loss_neg=loss_neg / len(img_metas))
