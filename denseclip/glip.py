from typing import Any, Dict, List, Optional, Tuple, cast

import einops
import einops.layers.torch
import todd.datasets
import todd.distillers
import todd.utils
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv import ConfigDict
from mmcv.runner import BaseModule, ModuleList
from mmdet.core import AssignResult, MaxIoUAssigner
from mmdet.models import BACKBONES, DETECTORS, NECKS, FasterRCNN, MaskRCNN, TwoStageDetector, build_neck
from todd.losses import LOSSES, L1Loss
from todd.utils import ListTensor

from .datasets import COCO_INDEX_SEEN_48_17, COCO_ALL_48_17, LVIS_V1_SEEN_866_337
from .mil_classifiers import BaseMILClassifier, MIL_CLASSIFIERS, ClassificationResult
from .mmdet_patch import DyHeadBlock, BFP, ResNet, TwoStageDetector, AnchorGenerator
from .refine import REFINE_LAYERS


# TODO: change class_embeddings to buffer with persistent=False


class BackboneFusion(BaseModule):
    def __init__(
        self, *args, v_dim: int, l_dim: int, hidden_dim: int, **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._v_proj = nn.Sequential(
            nn.Conv2d(v_dim, hidden_dim, 1),
            nn.Tanh(),
        )
        self._l_proj = nn.Sequential(
            nn.Linear(l_dim, hidden_dim),
            nn.Tanh(),
        )
        self._out_v_proj = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(hidden_dim, v_dim, 1),
            nn.BatchNorm2d(v_dim),
            nn.ReLU()
        )

    def forward(self, v: torch.Tensor, l: torch.Tensor) -> torch.Tensor:
        v_feats = self._v_proj(v)
        l_feats = self._l_proj(l)
        l_feats = einops.reduce(l_feats, 'b c -> 1 c 1 1', reduction='mean')
        v_feats = self._out_v_proj(v_feats * l_feats)
        # v_feats = F.normalize(v + v_feats)
        return v + v_feats


@BACKBONES.register_module()
class GLIPResNet(ResNet):
    def _make_custom_plugins(
        self, 
        in_channels: List[int],
        embedding_dim: int,
        hidden_dim: int, 
    ) -> nn.ModuleList:
        fusions = [BackboneFusion(
            v_dim=channel, 
            l_dim=embedding_dim, 
            hidden_dim=hidden_dim,
        ) for channel in in_channels]
        return nn.ModuleList(fusions)


class GLIPNeck(BFP):
    def __init__(
        self, 
        *args, 
        refine: ConfigDict, 
        **kwargs,
    ):
        super().__init__(*args, refine_type=None, **kwargs)
        self.refine_type = 'fusion_dyhead'
        self.refine = REFINE_LAYERS.build(
            refine,
            default_args=dict(
                channels=self.in_channels,
            )
        )


@LOSSES.register_module()
class DSLoss(L1Loss):
    def __init__(
        self, *args, pred_features: int, target_features: int, **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._adapt = nn.Linear(pred_features, target_features)

    def forward(
        self, 
        x: Tuple[torch.Tensor], 
        bboxes: List[torch.Tensor], 
        bbox_embeddings: List[torch.Tensor], 
        prior_generator: AnchorGenerator, 
        assigner: MaxIoUAssigner,
    ) -> Dict[str, torch.Tensor]:
        with prior_generator.with_pos():
            anchors = prior_generator.grid_priors(
                [featmap.shape[-2:] for featmap in x],
                device=x[0].device,
            )
        anchors = torch.cat(anchors)
        poses = anchors[:, [4, 7, 5, 6]]  # level, anchor_id(batch), h, w
        anchors = anchors[:, :4]
        preds = []  # position of preds
        targets = []
        for i, (bbox, bbox_embedding) in enumerate(zip(bboxes, bbox_embeddings)):
            with todd.utils.setattr_temp(assigner, 'match_low_quality', False):
                assign_result: AssignResult = assigner.assign(anchors, bbox)
            indices = assign_result.gt_inds > 0
            pred = poses[indices].clone()
            pred[:, 1] = i
            bbox_indices = assign_result.gt_inds[indices] - 1
            target = bbox_embedding[bbox_indices]
            preds.append(pred)
            targets.append(target)
        x = ListTensor.apply(x, lambda t: einops.rearrange(t, 'b c h w -> b h w c'))
        preds = ListTensor.index(
            x, torch.cat(preds),
        )
        preds = self._adapt(preds)
        preds = F.normalize(preds)
        targets = torch.cat(targets)
        loss = super().forward(preds, targets)
        return dict(loss_ds=loss)


class ClassEmbeddingsMixin(TwoStageDetector):
    def __init__(
        self, 
        *args, 
        class_embeddings: str = 'data/lvis_v1/prompt/detpro_ViT-B-32.pt', 
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        class_embeddings: torch.Tensor = torch.load(
            class_embeddings, map_location='cpu',
        )
        if class_embeddings.shape[0] == 80:
            class_embeddings = class_embeddings[COCO_ALL_48_17]
            seen_ids = COCO_INDEX_SEEN_48_17
        elif class_embeddings.shape[0] == 1203:
            seen_ids = LVIS_V1_SEEN_866_337
        else:
            raise ValueError(f'Unknown number of classes: {class_embeddings.shape[0]}')
        self._class_embeddings = nn.Parameter(class_embeddings.float(), requires_grad=False)
        self._seen_ids = seen_ids

    @property
    def class_embeddings(self) -> torch.Tensor:
        if self.training:
            return self._class_embeddings[self._seen_ids]
        return self._class_embeddings


class GLIPBackboneMixin(ClassEmbeddingsMixin):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert isinstance(self.backbone, GLIPResNet), type(self.backbone)

    def extract_feat(self, image: torch.Tensor) -> torch.Tensor:
        x = self.backbone(image, self.class_embeddings)
        x = self.neck(x)
        return x


class GLIPNeckMixin(ClassEmbeddingsMixin):
    def __init__(
        self, 
        *args, 
        glip_neck: ConfigDict, 
        loss_ds: Optional[ConfigDict] = None, 
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        glip_neck.refine.embedding_dim = self._class_embeddings.shape[1]
        self._glip_neck = GLIPNeck(**glip_neck)
        # self._seen_ids_mapper = {c: i for i, c in enumerate(self._seen_ids)}
        seen_ids_mapper = torch.zeros(self._class_embeddings.shape[0], dtype=torch.long) - 1
        seen_ids_mapper[self._seen_ids] = torch.arange(len(self._seen_ids))
        self.register_buffer('_seen_ids_mapper', seen_ids_mapper, persistent=False)

        self._loss_ds = None if loss_ds is None else LOSSES.build(loss_ds)

    def extract_feat(
        self, 
        image: torch.Tensor, 
        gt_labels: Optional[List[torch.Tensor]] = None, 
        clip_image_features: Optional[torch.Tensor] = None,
    ):
        x = super().extract_feat(image)
        if self.training:
            # mil_labels = []
            # for gt_label in gt_labels:
            #     mil_label = gt_label.clone()
            #     mil_label.apply_(self._seen_ids_mapper.__getitem__)
            #     mil_labels.append(mil_label)
            mil_labels = image.new_zeros((len(gt_labels), len(self._seen_ids)))
            for i, gt_label in enumerate(gt_labels):
                mil_label = self._seen_ids_mapper[gt_label]
                assert mil_label.ge(0).all(), gt_label
                mil_labels[i, mil_label] = 1.0
        else:
            mil_labels = None
        return self._glip_neck(x, self.class_embeddings, mil_labels, clip_image_features)

    def forward_train(
        self, 
        img: torch.Tensor, 
        img_metas: List[dict], 
        gt_bboxes: List[torch.Tensor],
        gt_labels: List[torch.Tensor],
        gt_bboxes_ignore: Optional[List[torch.Tensor]] = None,
        gt_masks: Optional[torch.Tensor] = None,
        proposals: Optional[List[torch.Tensor]] = None,
        *,
        image_embeddings: torch.Tensor, 
        bboxes: List[torch.Tensor],
        bbox_embeddings: List[torch.Tensor],
        **kwargs,
    ) -> Dict[str, Any]:
        x, (glip_losses,) = self.extract_feat(img, gt_labels, image_embeddings)
        x = cast(List[torch.Tensor], x)

        proposal_cfg = self.train_cfg.get('rpn_proposal', self.test_cfg.rpn)
        rpn_losses, proposal_list = self.rpn_head.forward_train(
            x, img_metas, gt_bboxes, proposal_cfg=proposal_cfg, **kwargs,
        )
        roi_losses = self.roi_head.forward_train(
            x, img_metas, proposal_list, gt_bboxes, gt_labels,
            gt_bboxes_ignore, gt_masks, bboxes=bboxes, 
            bbox_embeddings=bbox_embeddings, **kwargs,
        )

        losses = dict(**rpn_losses, **roi_losses, **glip_losses)
        if self._loss_ds is not None:
            ds_losses = self._loss_ds(
                x, bboxes, bbox_embeddings, 
                self.rpn_head.prior_generator, 
                self.rpn_head.assigner,
            )
            losses.update(ds_losses)
        return losses


class GLIPMixin(GLIPNeckMixin, GLIPBackboneMixin):
    pass


@DETECTORS.register_module()
class GLIPBackboneFasterRCNN(GLIPBackboneMixin, FasterRCNN):
    pass


@DETECTORS.register_module()
class GLIPNeckFasterRCNN(GLIPNeckMixin, FasterRCNN):
    pass


@DETECTORS.register_module()
class GLIPFasterRCNN(GLIPMixin, FasterRCNN):
    pass


@DETECTORS.register_module()
class GLIPBackboneMaskRCNN(GLIPBackboneMixin, MaskRCNN):
    pass


@DETECTORS.register_module()
class GLIPNeckMaskRCNN(GLIPNeckMixin, MaskRCNN):
    pass


@DETECTORS.register_module()
class GLIPMaskRCNN(GLIPMixin, MaskRCNN):
    pass
