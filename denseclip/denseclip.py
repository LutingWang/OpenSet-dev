from typing import Any, Dict, Tuple, Union

import einops

import clip
import clip.model
import todd
import torch
import torch.nn.functional as F
import torch.nn as nn
from mmcv import ConfigDict
from mmdet.datasets.coco import CocoDataset
from mmdet.models.builder import DETECTORS
from mmdet.models import RetinaHead, SingleStageDetector

from .coco import CocoGZSLDataset, CocoZSLSeenDataset
from .model import CLIPResNetWithAttention, ContextDecoder
from .utils import SimpleTokenizer


class VpeForwardPreHook(nn.Module):
    def __init__(self, vpe: torch.Tensor):
        super().__init__()
        resolution = round((vpe.shape[0] - 1) ** 0.5)
        assert resolution ** 2 + 1 == vpe.shape[0]
        vpi = torch.arange(resolution ** 2)
        vpi = einops.rearrange(vpi, '(h w) -> h w', h=resolution)
        self._vpe = vpe
        self._vpi = vpi
    
    def forward(self, module: clip.model.AttentionPool2d, input_: Tuple[torch.Tensor]):
        _, _, h, w = input_[0].shape
        vpi = einops.rearrange(self._vpi[:h, :w], 'h w -> (h w)')
        vpi = torch.cat((vpi.new_tensor([0]), vpi + 1))
        module.positional_embedding = self._vpe[vpi]


class PromptFrowardHook(nn.Module):
    def __init__(self, prompt_length: int, embedding_dim: int):
        super().__init__()
        self._prompt_length = prompt_length
        self._prompt = nn.Parameter(torch.randn(prompt_length, embedding_dim))
        nn.init.trunc_normal_(self._prompt)

    def __call__(self, module: nn.Module, input_: Any, output: Any):
        output[:, 1:1 + self._prompt_length] = self._prompt


class CLIPDistiller(todd.distillers.SingleTeacherDistiller):
    teacher: clip.model.CLIP

    def __init__(self, *args, teacher_cfg: ConfigDict, **kwargs):
        teacher: nn.Module = torch.jit.load(teacher_cfg.pretrained, map_location='cpu')
        state_dict = teacher.state_dict()
        is_vit = 'visual.proj' in state_dict
        v_name = 'visual' if is_vit else 'visual.attnpool'
        vpe_name = v_name + '.positional_embedding'

        if state_dict['input_resolution'] != teacher_cfg.input_resolution:
            # state_dict['input_resolution'] = teacher_cfg.input_resolution
            source_resolution = state_dict['input_resolution'] // 32
            target_resolution = teacher_cfg.input_resolution // 32
            assert source_resolution ** 2 + 1 == state_dict[vpe_name].shape[0]

            cls_pos = state_dict[vpe_name][[0]]
            spatial_pos = state_dict[vpe_name][1:]
            spatial_pos = einops.rearrange(spatial_pos, '(h w) dim -> 1 dim h w', h=source_resolution)
            spatial_pos = F.interpolate(spatial_pos, size=(target_resolution,) * 2, mode='bilinear')  # TODO: supress warning
            spatial_pos = einops.rearrange(spatial_pos, '1 dim h w -> (h w) dim')
            vpe = torch.cat([cls_pos, spatial_pos])
            state_dict[vpe_name] = vpe

        if state_dict['context_length'] > teacher_cfg.context_length:
            # state_dict['context_length'] = teacher_cfg.context_length
            state_dict['positional_embedding'] = state_dict['positional_embedding'][:teacher_cfg.context_length]

        teacher = clip.model.build_model(state_dict).float()
        super().__init__(*args, teacher=teacher, **kwargs)

        module: nn.Module = todd.utils.getattr_recur(self.teacher, v_name)
        self._vpe_forward_pre_hook = VpeForwardPreHook(
            vpe=module._parameters.pop('positional_embedding'), 
        )
        module.register_forward_pre_hook(self._vpe_forward_pre_hook)

        module = self.teacher.token_embedding
        self._prompt_forward_hook = PromptFrowardHook(
            prompt_length=teacher_cfg.prompt_length, 
            embedding_dim=module.embedding_dim,
        )
        module.register_forward_hook(self._prompt_forward_hook)
    
    @property
    def num_features(self) -> int:
        return self.teacher.visual.output_dim


@DETECTORS.register_module()
@CLIPDistiller.wrap()
class DenseCLIP_RetinaNet(SingleStageDetector):
    CLASSES: Tuple[str]
    distiller: CLIPDistiller
    backbone: CLIPResNetWithAttention
    bbox_head: RetinaHead

    def __init__(
        self, 
        *args,
        context_decoder: dict,
        tau=0.07,
        **kwargs,
    ):
        # super().__init__(init_cfg)
        super().__init__(*args, pretrained=None, **kwargs)
        self.context_decoder = ContextDecoder(**context_decoder)

        self.tau = tau
        self.gamma = nn.Parameter(torch.FloatTensor(data=[1e-4]))

    def _init_with_distiller(self):
        self._tokenizer = SimpleTokenizer(
            bpe_path='denseclip/bpe_simple_vocab_16e6.txt.gz', 
            context_length=self.distiller.teacher.context_length,
            prompt_length=self.distiller._prompt_forward_hook._prompt_length,
        )
        module = self.backbone.attnpool
        module._parameters.pop('positional_embedding'), 
        module.register_forward_pre_hook(self.distiller._vpe_forward_pre_hook)

        retina_cls = nn.Conv2d(
            self.bbox_head.feat_channels,
            self.distiller.num_features,  # NOTE: assumes single anchor
            kernel_size=3, padding=1,
        )
        retina_cls.register_forward_hook(self._retina_cls_forward_hook)
        self.bbox_head.retina_cls = retina_cls
        self._retina_cls_bias = nn.Parameter(torch.FloatTensor(data=[-4]))

    def extract_feat(self, img: torch.Tensor) -> torch.Tensor:
        x, visual_embeddings = self.backbone(img)

        x = self.neck(x)
        if self.training:
            return x, visual_embeddings
        else:
            return x

    def get_score_map(self, x, visual_embeddings):
        b, _, h, w = x.shape
        visual_embeddings = einops.rearrange(visual_embeddings, 'n b c -> b n c', n=h * w + 1)

        text_embeddings = einops.repeat(self.text_embeddings, 'k d -> b k d', b=b)
        text_diff = self.context_decoder(text_embeddings, visual_embeddings)
        text_embeddings = text_embeddings + self.gamma * text_diff

        text_embeddings = F.normalize(text_embeddings, dim=-1)
        spatial_embeddings = F.normalize(visual_embeddings[:, 1:, :], dim=-1)
        score_map = torch.einsum('b k c, b n c -> b k n', text_embeddings, spatial_embeddings) / self.tau
        score_map = einops.rearrange(score_map, 'b k (h w) -> b k h w', h=h, w=w)
        return score_map

    @property
    def classes(self) -> CocoDataset:
        return self.CLASSES if self.training else CocoGZSLDataset.CLASSES

    @property
    def tokens(self) -> torch.Tensor:
        return self._tokenizer.tokenize(self.classes, self.gamma.device)

    def _retina_cls_forward_hook(self, module: nn.Module, input_: Any, output: torch.Tensor):
        output = einops.rearrange(output, 'b (a c) h w -> b a c h w', c=self.distiller.num_features)
        output = torch.einsum(
            'b a c h w, k c -> b a k h w', 
            F.normalize(output, dim=2),
            F.normalize(self.text_embeddings, dim=-1),
        ) / self.tau + self._retina_cls_bias
        output = einops.rearrange(output, 'b a k h w -> b (a k) h w')
        return output

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None):
        self.text_embeddings: torch.Tensor = self.distiller.teacher.encode_text(self.tokens)
        super(SingleStageDetector, self).forward_train(img, img_metas)
        x, visual_embeddings = self.extract_feat(img)
        score_map = self.get_score_map(x[2], visual_embeddings)
        losses = self.bbox_head.forward_train(x, img_metas, gt_bboxes,
                                              gt_labels, gt_bboxes_ignore)
        # self.distiller.teacher.encode_image(img)
        losses['loss_seg'] = self.compute_seg_loss(img, score_map, img_metas, gt_bboxes, gt_labels)
        self.text_embeddings = None
        return losses

    def compute_seg_loss(self, img, score_map, img_metas, gt_bboxes, gt_labels):
        target, mask = self.build_seg_target(img, img_metas, gt_bboxes, gt_labels)
        score_map = F.interpolate(score_map, target.shape[2:], mode='bilinear')
        loss = F.binary_cross_entropy(score_map.sigmoid(), target, weight=mask, reduction='sum')
        loss = loss / mask.sum() * 0.5
        return loss

    def build_seg_target(self, img, img_metas, gt_bboxes, gt_labels):
        B, C, H, W = img.shape
        H //= 4
        W //= 4
        target = torch.zeros(B, len(self.CLASSES), H, W)
        mask = torch.zeros(B, 1, H, W)
        for i, (bboxes, gt_labels) in enumerate(zip(gt_bboxes, gt_labels)):
            bboxes = (bboxes / 4).long()
            bboxes[:, 0] = bboxes[:, 0].clamp(0, W - 1)
            bboxes[:, 1] = bboxes[:, 1].clamp(0, H - 1)
            bboxes[:, 2] = bboxes[:, 2].clamp(0, W - 1)
            bboxes[:, 3] = bboxes[:, 3].clamp(0, H - 1)
            for bbox, label in zip(bboxes, gt_labels):
                target[i, label, bbox[1]: bbox[3], bbox[0]: bbox[2]] = 1
                mask[i, :, bbox[1]: bbox[3], bbox[0]: bbox[2]] = 1
        mask = mask.expand(-1, len(self.CLASSES), -1, -1)
        target = target.to(img.device)
        mask = mask.to(img.device)
        return target, mask

    def forward_test(self, *args, **kwargs) -> Any:
        old_coc = self.bbox_head.cls_out_channels
        old_nc = self.bbox_head.num_classes

        self.bbox_head.cls_out_channels -= self.bbox_head.num_classes
        self.bbox_head.num_classes = len(self.classes)
        self.bbox_head.cls_out_channels += self.bbox_head.num_classes

        self.text_embeddings: torch.Tensor = self.distiller.teacher.encode_text(self.tokens)
        result = super().forward_test(*args, **kwargs)
        self.text_embeddings = None

        self.bbox_head.cls_out_channels = old_coc
        self.bbox_head.num_classes = old_nc
        return result
