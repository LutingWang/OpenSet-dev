from typing import List, Optional, Tuple

from mmdet.core.mask.structures import BitmapMasks
import torch
import todd.reproduction
from mmcv import ConfigDict
from mmcv.runner import BaseModule, ModuleList
from mmdet.models import DETECTORS, NECKS, FPN, TwoStageDetector

from ..datasets import LVIS_V1_SEEN_866_337
from .classifiers import BaseMILClassifier, MIL_CLASSIFIERS, ClassificationResult
from .mmdet_patch import TwoStageDetector
from .post_fpn import PostFPN
from .pre_fpn import PreFPN


@NECKS.register_module()
class CAFENeck(BaseModule):
    _seen_ids_mapper: torch.Tensor
    _class_embeddings: torch.Tensor

    def __init__(
        self, 
        *,
        in_channels: List[int],
        plv_channels: int,
        out_channels: int,
        num_outs: int,
        mil_classifier: ConfigDict,
        glip_refine_level: int,
        glip_refine_layers: int,
        class_embeddings: str = 'data/lvis_v1/prompt/detpro_ViT-B-32.pt', 
        norm_cfg: Optional[ConfigDict] = None,
        init_cfg: Optional[ConfigDict] = None,
    ):
        if init_cfg is None:
            init_cfg = dict(type='Xavier', layer='Conv2d', distribution='uniform')
        super().__init__(init_cfg)

        class_embeddings: torch.Tensor = torch.load(
            class_embeddings, map_location='cpu',
        )
        assert class_embeddings.shape[0] == 1203
        embedding_dim = class_embeddings.shape[1]

        self._mil_classifier: BaseMILClassifier = MIL_CLASSIFIERS.build(
            mil_classifier, 
            default_args=dict(
                channels=in_channels[-1],
                embedding_dim=embedding_dim,
            ),
        )

        self._pre = PreFPN(
            channels=in_channels, 
            embedding_dim=embedding_dim,
            hidden_dim=plv_channels,
        )

        self._fpn = FPN(
            in_channels, 
            out_channels, 
            num_outs,
            norm_cfg=norm_cfg,
            init_cfg=init_cfg,
        )

        self._post = PostFPN(
            channels=out_channels,
            refine_level=glip_refine_level,
            refine_layers=glip_refine_layers,
            init_cfg=init_cfg,
        )

        seen_ids = LVIS_V1_SEEN_866_337
        seen_ids_mapper = torch.zeros(class_embeddings.shape[0], dtype=torch.long) - 1
        seen_ids_mapper[seen_ids] = torch.arange(len(seen_ids))
        self._seen_ids = seen_ids
        self.register_buffer('_seen_ids_mapper', seen_ids_mapper, persistent=False)
        self.register_buffer('_class_embeddings', class_embeddings, persistent=False)

        self.init_weights()

    @property
    def class_embeddings(self) -> torch.Tensor:
        if self.training:
            return self._class_embeddings[self._seen_ids]
        return self._class_embeddings

    def forward_train(
        self, 
        x: List[torch.Tensor], 
        gt_labels: List[torch.Tensor], 
        gt_masks: List[BitmapMasks],
        gt_image_features: torch.Tensor,
    ):
        mil_labels = gt_image_features.new_zeros((len(gt_labels), len(self._seen_ids)))
        for i, gt_label in enumerate(gt_labels):
            mil_label = self._seen_ids_mapper[gt_label]
            assert mil_label.ge(0).all(), gt_label
            mil_labels[i, mil_label] = 1.0

        class_embeddings, class_logits, mil_losses, indices = self._mil_classifier.forward_train(
            x[-1], self.class_embeddings, mil_labels=mil_labels, 
            gt_image_features=gt_image_features,
        )
        class_weights = class_logits.detach().sigmoid()

        # gt_masks = gt_masks[indices]

        x = self._pre(x, class_embeddings, class_weights)
        x = self._fpn(x)
        x, post_fpn_losses = self._post.forward_train(
            x, class_embeddings, class_weights, gt_masks,
        )
        return x, {**mil_losses, **post_fpn_losses}

    def forward_test(self, x: List[torch.Tensor]) -> List[torch.Tensor]:
        class_embeddings, class_logits = self._mil_classifier.forward_test(
            x[-1], self.class_embeddings,
        )
        class_weights = class_logits.detach().sigmoid()
        x = self._pre(x, class_embeddings, class_weights)
        x = self._fpn(x)
        x = self._post.forward_test(
            x, class_embeddings, class_weights,
        )
        return x


@DETECTORS.register_module()
class CAFE(TwoStageDetector):
    neck: CAFENeck

    def extract_feat(self, img: torch.Tensor) -> List[torch.Tensor]:
        x: List[torch.Tensor] = self.backbone(img)
        x: List[torch.Tensor] = self.neck.forward_test(x)
        return x

    def forward_train(
        self, 
        img: torch.Tensor, 
        img_metas: List[dict], 
        gt_bboxes: List[torch.Tensor], 
        gt_labels: List[torch.Tensor], 
        *,
        gt_masks: List[BitmapMasks], 
        image_embeddings: torch.Tensor,
        **kwargs,
    ):
        x = self.backbone(img)
        x, cafe_losses = self.neck.forward_train(x, gt_labels, gt_masks, image_embeddings)

        proposal_cfg = self.train_cfg.get('rpn_proposal', self.test_cfg.rpn)
        with todd.reproduction.set_seed_temp(3407):
            rpn_losses, proposal_list = self.rpn_head.forward_train(
                x, img_metas, gt_bboxes, gt_labels=None,
                gt_bboxes_ignore=None,
                proposal_cfg=proposal_cfg, **kwargs,
            )

            roi_losses = self.roi_head.forward_train(
                x, img_metas, proposal_list, gt_bboxes, gt_labels,
                None, gt_masks, **kwargs,
            )

        return {**cafe_losses, **rpn_losses, **roi_losses}
