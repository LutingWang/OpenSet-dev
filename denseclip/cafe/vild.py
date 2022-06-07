from abc import abstractclassmethod, abstractmethod
from typing import Any, Dict, List, NoReturn, Optional, Tuple, Union, cast

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
    _class_embeddings: torch.Tensor

    def __init__(
        self, 
        *args, 
        class_embeddings: Union[str, torch.Tensor], 
        bg_class_embedding_trainable: bool,
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
            num_shared_convs=4,
            num_shared_fcs=1,
            num_cls_convs=0,
            num_cls_fcs=0,
            num_reg_convs=0,
            num_reg_fcs=0,
            **kwargs,
        )
        self.register_buffer('_class_embeddings', class_embeddings, persistent=False)

        self._bg_class_embedding_trainable = bg_class_embedding_trainable
        if bg_class_embedding_trainable:
            self._bg_class_embedding = nn.Parameter(
                self._class_embeddings[[0]], requires_grad=True,
            )
        else:
            self.register_buffer(
                '_bg_class_embedding', 
                torch.zeros_like(self._class_embeddings[[0]]), 
                persistent=False,
            )

        self._unseen_ids = [
            i for i in range(self.num_classes) 
            if i not in seen_ids
        ]
        self.fc_cls = nn.Sequential(
            nn.Linear(self.fc_out_channels, self.embedding_dim),
            Classifier(tau=(0.007, 0.01)),
        )

    @todd.reproduction.set_seed_temp('ViLDBaseBBoxHead')
    def init_weights(self) -> None:
        super().init_weights()
        if self._bg_class_embedding_trainable:
            nn.init.xavier_uniform_(self._bg_class_embedding)

    @property
    def embedding_dim(self) -> int:
        return self._class_embeddings.shape[1]

    @property
    def classifier(self) -> Classifier:
        return self.fc_cls[-1]

    @property
    def class_embeddings(self) -> torch.Tensor:
        bg_class_embeddings = self._bg_class_embedding
        if self._bg_class_embedding_trainable:
            bg_class_embeddings = F.normalize(bg_class_embeddings)
        return torch.cat([self._class_embeddings, bg_class_embeddings], dim=0)


@DETECTORS.register_module()
class ViLDTextBBoxHead(ViLDBaseBBoxHead):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, bg_class_embedding_trainable=True, **kwargs)

    @todd.reproduction.set_seed_temp('ViLDTextBBoxHead')
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
class ViLDImageBBoxHead(ViLDBaseBBoxHead):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, bg_class_embedding_trainable=False, **kwargs)

    def forward(self, *args, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        self.classifier.set_weight(None if self.training else self.class_embeddings, norm=False)
        cls_, reg = super().forward(*args, **kwargs)
        if not self.training:
            cls_[:, -1] = float('-inf')
        return cls_, reg

    def get_targets(self, *args, **kwargs) -> NoReturn:
        raise AssertionError

    def loss(self, *args, **kwargs) -> NoReturn:
        raise AssertionError


class ViLDDistiller(todd.distillers.SelfDistiller):
    def __init__(self, student: nn.Module, **kwargs):
        super().__init__(
            student,
            student_hooks=dict(
                preds=dict(
                    type='MultiTensorsHook', 
                    path='image_bbox_head.classifier', 
                    on_input=True,
                    tensor_names=('', ),
                ),
            ),
            adapts=dict(
                preds_adapted=dict(
                    type='Linear',
                    in_features=512,
                    out_features=512,
                    tensor_names=['preds_'],
                ),
            ),
            losses=dict(
                vild_image_kd=dict(
                    type='L1Loss',
                    tensor_names=['preds_adapted', 'targets'],
                    weight=256,
                    norm=True,
                ),
            ),
            schedulers=[
                dict(
                    type='WarmupScheduler',
                    tensor_names=['loss_vild_image_kd'],
                    iter_=200,
                ),
            ],
            **kwargs,
        )


@ViLDDistiller.wrap()
class ViLDBaseRoIHead(StandardRoIHead):
    distiller: ViLDDistiller

    @property
    def with_mask(self):
        return self.training and super().with_mask

    @property
    @abstractmethod
    def image_bbox_head(self) -> ViLDBaseBBoxHead:
        pass

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
        self.image_bbox_head(bbox_feats)
        loss_bbox = self.distiller.distill(dict(
            targets=torch.cat(bbox_embeddings),
        ))
        losses.update(loss_bbox)
        return losses

    def simple_test(self, *args, **kwargs):
        results = super().simple_test(*args, **kwargs)
        self.distiller.reset()
        return results

    def aug_test(self, *args, **kwargs):
        results = super().aug_test(*args, **kwargs)
        self.distiller.reset()
        return results


@DETECTORS.register_module()
class ViLDRoIHead(ViLDBaseRoIHead):
    @property
    def image_bbox_head(self) -> ViLDImageBBoxHead:
        return self.bbox_head


@DETECTORS.register_module()
class ViLDEnsembleRoIHead(ViLDBaseRoIHead):
    _ensemble_mask: torch.Tensor

    def __init__(self, *args, text_bbox_head: ConfigDict, image_bbox_head: ConfigDict, **kwargs):
        super().__init__(*args, bbox_head=text_bbox_head, **kwargs)
        self._ensemble_head: ViLDImageBBoxHead = HEADS.build(image_bbox_head)
        ensemble_mask = torch.ones(self._ensemble_head.num_classes + 1) / 3
        ensemble_mask[self._ensemble_head._seen_ids] *= 2
        self.register_buffer('_ensemble_mask', ensemble_mask, persistent=False)
        self.init_weights()

    @property
    def image_bbox_head(self) -> ViLDImageBBoxHead:
        return self._ensemble_head

    def _bbox_forward(
        self, 
        x: List[torch.Tensor], 
        rois: torch.Tensor, 
    ) -> Dict[str, torch.Tensor]:
        assert not self.with_shared_head
        bbox_results: Dict[str, torch.Tensor] = super()._bbox_forward(x, rois)
        if not self.training:
            cls_score, _ = self._ensemble_head(bbox_results['bbox_feats'])
            ensemble_score: torch.Tensor = (
                bbox_results['cls_score'].softmax(-1) ** self._ensemble_mask
                * cls_score.softmax(-1) ** (1 - self._ensemble_mask)
            )
            ensemble_score[:, -1] = 1 - ensemble_score[:, :-1].sum(-1)
            bbox_results['cls_score'] = ensemble_score.log()
        return bbox_results
