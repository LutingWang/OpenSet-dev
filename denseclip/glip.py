from typing import Any, Dict, List, Optional, Tuple

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
from mmdet.models import BACKBONES, DETECTORS, NECKS, FasterRCNN, MaskRCNN, TwoStageDetector, build_neck
from timm.models.layers import DropPath

from .datasets import COCO_INDEX_SEEN_48_17, COCO_ALL_48_17, LVIS_V1_SEEN_866_337
from .mil_classifiers import BaseMILClassifier, MIL_CLASSIFIER, ClassificationResult
from .mmdet_patch import DyHeadBlock, BFP, ResNet, TwoStageDetector


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
        v_feats = F.normalize(v + v_feats)
        return v_feats


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


class Fusion(BaseModule):
    def __init__(
        self, 
        *args, 
        num_heads: int = 8,
        embed_dim: int = 2048,
        v_dim: int = 256, 
        l_dim: int = 512,
        avg_factor: int,
        dropout: float = 0.1, 
        drop_path: float = 0.0,
        bi_direct: bool = True,
        **kwargs,
    ):
        super().__init__(
            *args,
            init_cfg=dict(
                type='Xavier', layer='Conv2d', distribution='uniform',
            ),
            **kwargs,
        )

        head_dim = embed_dim // num_heads
        assert head_dim * num_heads == embed_dim

        self._num_heads = num_heads
        self._head_dim = head_dim
        self._scale = head_dim ** (-0.5)

        self._v_layer_norm = nn.LayerNorm(v_dim)
        self._v_proj = nn.Linear(v_dim, embed_dim)
        self._values_l_proj = nn.Linear(l_dim, embed_dim)
        self._out_v_proj = nn.Linear(embed_dim, v_dim)
        self._gamma_v = nn.Parameter(torch.ones((v_dim)) / avg_factor, requires_grad=True)

        self._l_layer_norm = nn.LayerNorm(l_dim)
        self._l_proj = nn.Linear(l_dim, embed_dim)
        if bi_direct:
            self._values_v_proj = nn.Linear(v_dim, embed_dim)
            self._out_l_proj = nn.Linear(embed_dim, l_dim)
            self._gamma_l = nn.Parameter(torch.ones((l_dim)) / avg_factor, requires_grad=True)

        self._dropout = dropout
        self._drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self._bi_direct = bi_direct

        self._stable_softmax_2d = False
        self._clamp_min_for_underflow = True
        self._clamp_max_for_overflow = True

    def forward(self, v: torch.Tensor, l: torch.Tensor, l_weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        h, w = v.shape[-2:]
        v = einops.rearrange(v, 'b c h w -> b (h w) c')

        v = self._v_layer_norm(v)
        l = self._l_layer_norm(l)

        query_states: torch.Tensor = einops.rearrange(
            self._v_proj(v) * self._scale, 
            'b hw (num_heads head_dim) -> (b num_heads) hw head_dim', 
            num_heads=self._num_heads, head_dim=self._head_dim,
        )
        key_states: torch.Tensor = einops.rearrange(
            self._l_proj(l), 
            'b l (num_heads head_dim) -> (b num_heads) l head_dim', 
            num_heads=self._num_heads, head_dim=self._head_dim,
        )

        attn_weights = torch.einsum('b n c, b l c -> b n l', query_states, key_states)
        if self._stable_softmax_2d:
            attn_weights = attn_weights - attn_weights.max()
        attn_weights = torch.clamp(
            attn_weights, 
            min=-50000 if self._clamp_min_for_underflow else None,
            max=50000 if self._clamp_max_for_overflow else None,
        )  # Do not increase 50000, data type half has quite limited range
        attn_weights_v = attn_weights.softmax(dim=-1)
        if l_weights is not None:
            l_weights = einops.repeat(l_weights, 'b num_heads -> (b num_heads) 1 l', num_heads=self._num_heads)
            attn_weights_v = attn_weights_v * l_weights
            attn_weights_v = attn_weights_v / attn_weights_v.sum(dim=-1, keepdim=True)
        attn_probs_v = F.dropout(attn_weights_v, p=self._dropout, training=self.training)
        value_l_states = einops.rearrange(self._values_l_proj(l), 'b l (num_heads head_dim) -> (b num_heads) l head_dim', num_heads=self._num_heads, head_dim=self._head_dim)
        attn_output_v = torch.einsum('b n l, b l c -> b n c', attn_probs_v, value_l_states)
        attn_output_v = einops.rearrange(attn_output_v, '(b num_heads) n head_dim -> b n (num_heads head_dim)', num_heads=self._num_heads, head_dim=self._head_dim)
        delta_v = self._out_v_proj(attn_output_v)

        if self._bi_direct:
            attn_weights = einops.rearrange(attn_weights, 'b hw l -> b l hw')
            attn_weights = attn_weights - torch.max(attn_weights, dim=-1, keepdim=True)[0]
            attn_weights = torch.clamp(
                attn_weights, 
                min=-50000 if self._clamp_min_for_underflow else None,
                max=50000 if self._clamp_max_for_overflow else None,
            )  # Do not increase 50000, data type half has quite limited range
            attn_weights_l = attn_weights.softmax(dim=-1)
            attn_probs_l = F.dropout(attn_weights_l, p=self._dropout, training=self.training)
            value_v_states = einops.rearrange(self._values_v_proj(v), 'b hw (num_heads head_dim) -> (b num_heads) hw head_dim', num_heads=self._num_heads, head_dim=self._head_dim)
            attn_output_l = torch.einsum('b l n, b n c -> b l c', attn_probs_l, value_v_states)
            attn_output_l = einops.rearrange(attn_output_l, '(b num_heads) l head_dim -> b l (num_heads head_dim)', num_heads=self._num_heads, head_dim=self._head_dim)
            delta_l = self._out_l_proj(attn_output_l)

        v = v + self._drop_path(self._gamma_v * delta_v)
        v = einops.rearrange(v, 'b (h w) c -> b c h w', h=h, w=w)
        if self._bi_direct:
            l = l + self._drop_path(self._gamma_l * delta_l)
            return v, l
        else:
            return v, None


class FusionDyHead(BaseModule):
    def __init__(
        self, 
        *args, 
        channels: int, 
        embedding_dim: int,
        mil_classifier: ConfigDict,
        num_layers: int = 6, 
        image_kd_loss_weight: int = 256,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._num_layers = num_layers
        self._image_kd_loss_weight = image_kd_loss_weight

        self._mil_classifier: BaseMILClassifier = MIL_CLASSIFIER.build(
            mil_classifier, 
            default_args=dict(
                channels=channels,
                embedding_dim=embedding_dim,
            ),
        )
        self._fuse_layers = ModuleList(
            Fusion(
                avg_factor=num_layers, 
                bi_direct=(l != num_layers - 1),
            ) for l in range(num_layers)
        )
        self._dyhead_layers = ModuleList(
            DyHeadBlock(channels, channels) for _ in range(num_layers)
        )
        self.init_weights()

    def forward(
        self, 
        bsf: torch.Tensor, 
        class_embeddings: torch.Tensor, 
        mil_labels: Optional[List[torch.Tensor]] = None,
        clip_image_features: Optional[torch.Tensor] = None,
    ):
        classification_result: ClassificationResult = self._mil_classifier(bsf, class_embeddings)
        class_embeddings = classification_result.class_embeddings
        image_features = classification_result.image_features
        class_logits = classification_result.class_logits
        logits_weight = classification_result.logits_weight
        for fuse_layer, dyhead_layer in zip(self._fuse_layers, self._dyhead_layers):
            bsf, class_embeddings = fuse_layer(bsf, class_embeddings, logits_weight)
            bsf = dyhead_layer(bsf)
        assert class_embeddings is None
        if not self.training:
            assert mil_labels is None
            return bsf

        onehot_labels = torch.zeros_like(class_logits)
        for i, labels in enumerate(mil_labels):
            onehot_labels[i, labels] = 1.0
        loss_mil = F.binary_cross_entropy_with_logits(class_logits, onehot_labels.float())

        clip_image_features = F.normalize(clip_image_features)
        loss_image_kd = F.l1_loss(image_features, clip_image_features, reduction='mean') * self._image_kd_loss_weight
        return bsf, dict(loss_mil=loss_mil, loss_image_kd=loss_image_kd)


class GLIPNeck(BFP):
    def __init__(
        self, 
        *args, 
        refine: ConfigDict, 
        **kwargs,
    ):
        super().__init__(*args, refine_type=None, **kwargs)
        self.refine_type = 'fusion_dyhead'
        self.refine = FusionDyHead(
            channels=self.in_channels, **refine,
        )


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
    def __init__(self, *args, glip_neck: ConfigDict, **kwargs):
        super().__init__(*args, **kwargs)
        glip_neck.refine.embedding_dim = self._class_embeddings.shape[1]
        self._glip_neck = GLIPNeck(**glip_neck)
        self._seen_ids_mapper = {c: i for i, c in enumerate(self._seen_ids)}

    def extract_feat(
        self, 
        image: torch.Tensor, 
        gt_labels: Optional[List[torch.Tensor]] = None, 
        clip_image_features: Optional[torch.Tensor] = None,
    ):
        x = super().extract_feat(image)
        if self.training:
            mil_labels = []
            for gt_label in gt_labels:
                mil_label = gt_label.clone()
                mil_label.apply_(self._seen_ids_mapper.__getitem__)
                mil_labels.append(mil_label)
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
        **kwargs,
    ) -> Dict[str, Any]:
        x, (glip_losses,) = self.extract_feat(img, gt_labels, image_embeddings)
        proposal_cfg = self.train_cfg.get('rpn_proposal', self.test_cfg.rpn)
        rpn_losses, proposal_list = self.rpn_head.forward_train(
            x, img_metas, gt_bboxes, proposal_cfg=proposal_cfg, **kwargs,
        )
        roi_losses = self.roi_head.forward_train(
            x, img_metas, proposal_list, gt_bboxes, gt_labels,
            gt_bboxes_ignore, gt_masks, **kwargs,
        )
        return dict(**rpn_losses, **roi_losses, **glip_losses)


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
class GLIPMaskRCNN(GLIPMixin, MaskRCNN):
    pass
