from typing import Any, Dict, List, Tuple, Union, cast
from denseclip.utils import has_debug_flag
from mmdet.core.bbox.transforms import bbox2roi
from mmdet.models.builder import HEADS

import todd
import todd.distillers
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv import ConfigDict
from mmdet.models import DETECTORS, RetinaHead, Shared4Conv1FCBBoxHead, StandardRoIHead

from .datasets import COCO_INDEX_SEEN_48_17, COCO_ALL_48_17, CocoGZSLDataset, LVIS_V1_SEEN_866_337
from .prompt import Classifier


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


class ViLDBaseBBoxHead(Shared4Conv1FCBBoxHead):
    def __init__(
        self, 
        *args, 
        class_embeddings: Union[str, torch.Tensor], 
        **kwargs,
    ):
        if isinstance(class_embeddings, str):
            class_embeddings = torch.load(class_embeddings, map_location='cpu')
        class_embeddings = class_embeddings.float()
        if class_embeddings.shape[0] == 80:
            class_embeddings = class_embeddings[COCO_ALL_48_17]
            seen_ids = COCO_INDEX_SEEN_48_17
        elif class_embeddings.shape[0] == 1203:
            seen_ids = LVIS_V1_SEEN_866_337
        else:
            raise ValueError(f'Unknown number of classes: {class_embeddings.shape[0]}')
        self._seen_ids = seen_ids

        super().__init__(*args, **kwargs)
        self._unseen_ids = [
            i for i in range(self.num_classes) 
            if i not in seen_ids
        ]
        self._class_embeddings = nn.Parameter(
            class_embeddings, requires_grad=False,
        )

        self.fc_cls = nn.Sequential(
            nn.Linear(self.fc_out_channels, self.embedding_dim),
            Classifier(tau=(0.007, 0.01)),
        )
        
        # do not put into init_weight, since PretrainedInit will skip it
        nn.init.xavier_uniform_(self.fc_cls[0].weight)
        nn.init.constant_(self.fc_cls[0].bias, 0)

    @property
    def embedding_dim(self) -> int:
        return self._class_embeddings.shape[1]

    @property
    def classifier(self) -> Classifier:
        return self.fc_cls[-1]

    def _set_classifier_weight(self):
        self.classifier.set_weight(self._class_embeddings, norm=False)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        self._set_classifier_weight()
        cls_, reg = super().forward(x)
        cls_ = cast(torch.Tensor, cls_)
        if cls_.shape[-1] == self.num_classes:
            bg_cls = torch.zeros_like(cls_[:, [0]])
            cls_ = torch.cat([cls_, bg_cls], dim=1)
        else:
            assert cls_.shape[-1] == self.num_classes + 1, cls_.shape
        if self.training:
            cls_[..., self._unseen_ids] = float('-inf')
        return cls_, reg


@DETECTORS.register_module()
class ViLDTextBBoxHead(ViLDBaseBBoxHead):
    def __init__(self, *args, bg_class_embedding: bool = True, **kwargs):
        super().__init__(*args, **kwargs)
        if not bg_class_embedding:
            self._bg_class_embedding = None
            return
        self._bg_class_embedding = nn.Parameter(
            self._class_embeddings[[0]], requires_grad=True,
        )
        nn.init.xavier_uniform_(self._bg_class_embedding)

    def _set_classifier_weight(self):
        if self._bg_class_embedding is None:
            super()._set_classifier_weight()
            return
        class_embeddings = torch.cat([
            self._class_embeddings, 
            F.normalize(self._bg_class_embedding),
        ], dim=0)
        self.classifier.set_weight(class_embeddings, norm=False)


@DETECTORS.register_module()
@todd.distillers.SelfDistiller.wrap()
class ViLDImageBBoxHead(ViLDBaseBBoxHead):
    distiller: todd.distillers.SelfDistiller

    def __init__(self, *args, **kwargs):
        kwargs.pop('with_reg', None)
        kwargs.pop('bg_class_embedding', None)
        super().__init__(*args, with_reg=False, **kwargs)

    def forward(self, *args, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        if not self.training:
            cls_, _ = super().forward(*args, **kwargs)
            self.distiller.reset()
            return cls_, None
        self.classifier.set_weight(None, norm=False)
        return super(ViLDBaseBBoxHead, self).forward(*args, **kwargs)

    def loss(self, preds: torch.Tensor, targets: torch.Tensor) -> Dict[str, torch.Tensor]:
        return self.distiller.distill(dict(
            preds=preds, targets=targets,
        ))


@DETECTORS.register_module()
class ViLDEnsembleRoIHead(StandardRoIHead):
    def __init__(self, *args, bbox_head: ConfigDict, ensemble_head: ConfigDict, **kwargs):
        super().__init__(*args, bbox_head=bbox_head, **kwargs)
        ensemble_head = HEADS.build(ensemble_head, default_args=bbox_head)
        ensemble_mask = nn.Parameter(
            torch.ones(ensemble_head.num_classes + 1) / 3, 
            requires_grad=False,
        )
        ensemble_mask[ensemble_head._seen_ids] *= 2
        self._ensemble_head = ensemble_head
        self._ensemble_mask = ensemble_mask

        self._ensemble_head.init_weights()
        if self.with_mask:
            self.mask_head.init_weights()

    # def _load_from_state_dict(self, state_dict: Dict[str, torch.Tensor], prefix: str, *args, **kwargs):
    #     if not any('_ensemble_head' in k for k in state_dict):
    #         state_dict.update({
    #             '_ensemble_head'.join(k.split('bbox_head', 1)): v 
    #             for k, v in state_dict.items() 
    #             if k.startswith(prefix + 'bbox_head.')
    #         })
    #     return super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def _bbox_forward(
        self, 
        x: List[torch.Tensor], 
        rois: torch.Tensor, 
    ) -> Dict[str, torch.Tensor]:
        assert not self.with_shared_head
        bbox_results: Dict[str, torch.Tensor] = super()._bbox_forward(x, rois)
        if not self.training:
            bbox_feats = bbox_results['bbox_feats']
            cls_score = bbox_results['cls_score']
            cls_score = cls_score.softmax(-1) ** self._ensemble_mask
            cls_score_, _ = self._ensemble_head(bbox_feats)
            cls_score_ = cast(torch.Tensor, cls_score_)
            cls_score_ = cls_score_.softmax(-1) ** (1 - self._ensemble_mask)
            ensemble_score = cls_score * cls_score_
            ensemble_score[:, -1] = 1 - ensemble_score[:, :-1].sum(-1)
            bbox_results['cls_score'] = ensemble_score.log()
        return bbox_results

    def forward_train(
        self, 
        x: List[torch.Tensor], 
        *args, 
        bboxes: List[torch.Tensor], 
        bbox_embeddings: List[torch.Tensor], 
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        losses = super().forward_train(x, *args, **kwargs)
        rois = bbox2roi(bboxes)
        bbox_feats = self.bbox_roi_extractor(
            x[:self.bbox_roi_extractor.num_inputs], rois,
        )
        if self.with_shared_head:
            bbox_feats = self.shared_head(bbox_feats)
        cls_, _ = self._ensemble_head(bbox_feats)
        loss_bbox = self._ensemble_head.loss(
            cls_, torch.cat(bbox_embeddings),
        )
        losses.update(loss_bbox)
        return losses
