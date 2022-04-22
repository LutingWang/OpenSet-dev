from typing import Any, Dict, List, Optional, Tuple, Union, cast
from denseclip.utils import has_debug_flag
from mmdet.core.bbox.transforms import bbox2roi
from mmdet.models.builder import HEADS, build_head

import todd
import todd.distillers
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv import ConfigDict
from mmdet.models import DETECTORS, RetinaHead, Shared4Conv1FCBBoxHead, StandardRoIHead

from .datasets import COCO_INDEX_SEEN_48_17, COCO_ALL_48_17, CocoGZSLDataset, LVIS_V1_SEEN
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


@DETECTORS.register_module()
class ViLDTextBBoxHead(Shared4Conv1FCBBoxHead):
    def __init__(self, *args, class_embeddings: Union[str, torch.Tensor], bg_class_embedding: bool = True, **kwargs):
        if isinstance(class_embeddings, str):
            class_embeddings = torch.load(class_embeddings, map_location='cpu')
        class_embeddings = class_embeddings.float()
        if class_embeddings.shape[0] == 80:
            class_embeddings = class_embeddings[COCO_ALL_48_17]
            seen_ids = COCO_INDEX_SEEN_48_17
        elif class_embeddings.shape[0] == 1203:
            seen_ids = LVIS_V1_SEEN
        else:
            raise ValueError(f'Unknown number of classes: {class_embeddings.shape[0]}')
        
        # enable self.num_classes
        self._class_embeddings = class_embeddings
        self._seen_ids = seen_ids

        super().__init__(*args, **kwargs)
        self.fc_cls = nn.Sequential(
            nn.Linear(self.fc_out_channels, self.embedding_dim),
            Classifier(tau=(0.007, 0.01)),
        )
        nn.init.xavier_uniform_(self.fc_cls[0].weight)
        nn.init.constant_(self.fc_cls[0].bias, 0)

        self._class_embeddings = nn.Parameter(
            class_embeddings, requires_grad=False,
        )
        if bg_class_embedding is None:
            self._bg_class_embedding = None
        else:
            bg_class_embedding = nn.Parameter(
                torch.zeros_like(class_embeddings[[0]]), requires_grad=True,
            )
            nn.init.xavier_uniform_(bg_class_embedding)
            self._bg_class_embedding = bg_class_embedding


    @property
    def num_classes(self) -> int:
        if self.training:
            return len(self._seen_ids)
        return self._class_embeddings.shape[0]

    @num_classes.setter
    def num_classes(self, value: Any):
        return

    @property
    def embedding_dim(self) -> int:
        return self._class_embeddings.shape[1]

    @property
    def classifier(self) -> Classifier:
        return self.fc_cls[-1]

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        class_embeddings = self._class_embeddings
        if self._bg_class_embedding is not None:
            bg_class_embedding = F.normalize(self._bg_class_embedding)
            class_embeddings = torch.cat([class_embeddings, bg_class_embedding], dim=0)
        self.classifier.set_weight(class_embeddings, norm=False)
        cls_, reg = super().forward(x)
        if self._bg_class_embedding is None:
            bg_cls = torch.zeros_like(cls_[:, [0]])
            cls_ = torch.cat([cls_, bg_cls], dim=1)
        return cls_, reg


@DETECTORS.register_module()
@todd.distillers.SelfDistiller.wrap()
class ViLDImageBBoxHead(ViLDTextBBoxHead):
    distiller: todd.distillers.SelfDistiller

    def __init__(self, *args, bg_class_embedding: bool = False, with_reg: bool = False, **kwargs):
        super().__init__(*args, bg_class_embedding=False, with_reg=False, **kwargs)

    def forward(self, *args, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        cls_, _ = super().forward(*args, **kwargs)
        if not self.training:
            self.distiller.reset()
        return cls_, None

    def loss(self, targets: torch.Tensor) -> Dict[str, torch.Tensor]:
        return self.distiller.distill(dict(
            targets=targets,
        ))


@DETECTORS.register_module()
class ViLDEnsembleRoIHead(StandardRoIHead):
    def __init__(self, *args, bbox_head: ConfigDict, ensemble_head: ConfigDict, **kwargs):
        super().__init__(*args, bbox_head=bbox_head, **kwargs)
        ensemble_head = HEADS.build(ensemble_head, default_args=bbox_head)
        ensemble_head = cast(ViLDImageBBoxHead, ensemble_head)
        ensemble_mask = nn.Parameter(
            torch.ones(ensemble_head._class_embeddings.shape[0] + 1) / 3, 
            requires_grad=False,
        )
        ensemble_mask[ensemble_head._seen_ids] *= 2
        self._ensemble_head = ensemble_head
        self._ensemble_mask = ensemble_mask

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
        data = [x, args[1], args[3], bboxes, bbox_embeddings, self.bbox_head._bg_class_embedding, self.bbox_head.fc_cls[0].state_dict(), self._ensemble_head.fc_cls[0].state_dict(), self.bbox_head.fc_reg.state_dict(), self._ensemble_head.state_dict()]
        torch.save(data, 'data.pth')
        losses = super().forward_train(x, *args, **kwargs)
        rois = bbox2roi(bboxes)
        bbox_feats = self.bbox_roi_extractor(
            x[:self.bbox_roi_extractor.num_inputs], rois,
        )
        import ipdb; ipdb.set_trace()
        if self.with_shared_head:
            bbox_feats = self.shared_head(bbox_feats)
        self._ensemble_head(bbox_feats)
        loss_bbox = self._ensemble_head.loss(
            torch.cat(bbox_embeddings),
        )
        losses.update(loss_bbox)
        return losses
