from typing import Any, Dict, List, Optional, Tuple, Union, cast

import todd
import todd.distillers
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv import ConfigDict
from mmdet.core import bbox2roi
from mmdet.models import DETECTORS, HEADS, ConvFCBBoxHead, StandardRoIHead

from ..datasets import COCO_INDEX_SEEN_48_17, COCO_ALL_48_17, LVIS_V1_SEEN_866_337
from .classifiers import Classifier


class ViLDBaseBBoxHead(ConvFCBBoxHead):
    def __init__(
        self, 
        *args, 
        class_embeddings: Union[str, torch.Tensor], 
        num_shared_convs: int = 4,
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

        super().__init__(
            *args, 
            num_shared_convs=num_shared_convs,
            num_shared_fcs=1,
            num_cls_convs=0,
            num_cls_fcs=0,
            num_reg_convs=0,
            num_reg_fcs=0,
            **kwargs,
        )
        self.register_buffer('_class_embeddings', class_embeddings, persistent=False)

        self._unseen_ids = [
            i for i in range(self.num_classes) 
            if i not in seen_ids
        ]
        self.fc_cls = nn.Sequential(
            nn.Linear(self.fc_out_channels, self.embedding_dim),
            Classifier(tau=(0.007, 0.01)),
        )

    @property
    def embedding_dim(self) -> int:
        return self._class_embeddings.shape[1]

    @property
    def classifier(self) -> Classifier:
        return self.fc_cls[-1]

    @property
    def class_embeddings(self) -> torch.Tensor:
        return self._class_embeddings


@DETECTORS.register_module()
class ViLDTextBBoxHead(ViLDBaseBBoxHead):
    def __init__(
        self, 
        *args, 
        bg_class_embedding: bool = True, 
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if not bg_class_embedding:
            self._bg_class_embedding = None
            return
        self._bg_class_embedding = nn.Parameter(
            self._class_embeddings[[0]], requires_grad=True,
        )
    
    def init_weights(self):
        super().init_weights()
        nn.init.xavier_uniform_(self._bg_class_embedding)

    @property
    def class_embeddings(self) -> torch.Tensor:
        class_embeddings = super().class_embeddings 
        if self._bg_class_embedding is None:
            return class_embeddings
        bg_class_embeddings = F.normalize(self._bg_class_embedding)
        return torch.cat([class_embeddings, bg_class_embeddings], dim=0)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        class_embeddings = self.class_embeddings
        self.classifier.set_weight(class_embeddings, norm=False)

        cls_, reg = super().forward(x)
        cls_: torch.Tensor
        assert cls_.shape[-1] == self.num_classes + 1, cls_.shape
        if self.training:
            # setting score=-inf for unseen class
            # not including background
            cls_[..., self._unseen_ids] = float('-inf')
        return cls_, reg


@DETECTORS.register_module()
@todd.distillers.SelfDistiller.wrap()
class ViLDImageBBoxHead(ViLDBaseBBoxHead):
    distiller: todd.distillers.SelfDistiller

    def __init__(self, *args, **kwargs):
        kwargs.pop('with_reg', None)
        kwargs.pop('bg_class_embedding', None)
        super().__init__(*args, with_reg=False, **kwargs)

    @property
    def class_embeddings(self) -> Optional[torch.Tensor]:
        return None if self.training else super().class_embeddings

    def forward(self, *args, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        self.classifier.set_weight(self.class_embeddings, norm=False)
        cls_, reg = super().forward(*args, **kwargs)
        if not self.training:
            bg_cls = torch.zeros_like(cls_[:, [0]]) - float('inf')
            cls_ = torch.cat([cls_, bg_cls], dim=1)
            self.distiller.reset()
        return cls_, reg

    def loss(self, preds: torch.Tensor, targets: torch.Tensor) -> Dict[str, torch.Tensor]:
        return self.distiller.distill(dict(
            preds=preds, targets=targets,
        ))


@DETECTORS.register_module()
class ViLDEnsembleRoIHead(StandardRoIHead):
    def __init__(self, *args, bbox_head: ConfigDict, ensemble_head: ConfigDict, **kwargs):
        super().__init__(*args, bbox_head=bbox_head, **kwargs)
        ensemble_head = HEADS.build(ensemble_head, default_args=bbox_head)
        ensemble_head = cast(ViLDImageBBoxHead, ensemble_head)
        self._ensemble_head = ensemble_head
        ensemble_mask = torch.ones(ensemble_head.num_classes + 1) / 3
        ensemble_mask[ensemble_head._seen_ids] *= 2
        self.register_buffer('_ensemble_mask', ensemble_mask, persistent=False)
        self.init_weights()

    @property
    def with_mask(self):
        return self.training and super().with_mask

    def _load_from_state_dict(
        self, 
        state_dict: Dict[str, torch.Tensor], 
        prefix: str, 
        local_metadata: dict, 
        strict: bool,
        missing_keys: List[str], 
        unexpected_keys: List[str], 
        error_msgs: List[str],
    ):
        # if not any('_ensemble_head' in k for k in state_dict):
        #     state_dict.update({
        #         '_ensemble_head'.join(k.split('bbox_head', 1)): v 
        #         for k, v in state_dict.items() 
        #         if k.startswith(prefix + 'bbox_head.')
        #     })
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs,
        )

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