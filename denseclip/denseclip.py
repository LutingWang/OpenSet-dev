from typing import Any, Dict, List, Optional, Tuple

import einops
import clip.clip
import clip.model
import todd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from easydict import EasyDict
from mmcv import ConfigDict
from mmdet.core import AssignResult, anchor_inside_flags
from mmdet.models import DETECTORS, RetinaHead, SingleStageDetector

from .datasets import CocoGZSLDataset
from .model import CLIPResNetWithAttention, ContextDecoder, RetinaRPNHead, Classifier
from .prior_generator import AnchorGeneratorWithPos
# from .utils import SimpleTokenizer, encode_bboxes


class VpeForwardPreHook(nn.Module):
    def __init__(self, vpe: torch.Tensor):
        super().__init__()
        resolution = round((vpe.shape[0] - 1) ** 0.5)
        assert resolution ** 2 + 1 == vpe.shape[0]
        vpi = torch.arange(resolution ** 2)
        vpi = einops.rearrange(vpi, '(h w) -> h w', h=resolution)
        self._vpe = vpe
        self._vpi = vpi
    
    def register(self, module: nn.Module):
        delattr(module, 'positional_embedding')
        # module._parameters.pop('positional_embedding')
        module.register_forward_pre_hook(self)

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
        nn.init.trunc_normal_(self._prompt, std=0.02)

    def register(self, module: nn.Module):
        module.register_forward_hook(self)

    def forward(self, module: nn.Module, input_: Any, output: Any):
        output[:, 1:1 + self._prompt_length] = self._prompt


class CLIPDistiller(todd.distillers.SingleTeacherDistiller):
    teacher: clip.model.CLIP

    def __init__(
        self, 
        *args, 
        teacher_cfg: ConfigDict, 
        **kwargs,
    ):
        self._teacher_cfg = teacher_cfg

        teacher = torch.jit.load(teacher_cfg.pretrained, map_location='cpu')
        teacher = self.customize_teacher(teacher)
        super().__init__(*args, teacher=teacher, **kwargs)

        if teacher_cfg.get('image_only', False):
            self.teacher.token_embedding = None
            self.teacher.positional_embedding = None
            self.teacher.transformer = None
            self.teacher.ln_final = None
            self.teacher.text_projection = None
        elif teacher_cfg.get('prompt_hook', True):
            self._prompt_forward_hook = PromptFrowardHook(
                prompt_length=teacher_cfg.prompt_length, 
                embedding_dim=self.teacher.token_embedding.embedding_dim,
            )
            self._prompt_forward_hook.register(self.teacher.token_embedding)

        if teacher_cfg.get('text_only', False):
            assert not teacher_cfg.get('with_preprocess')
            delattr(self.teacher, 'visual')
            self.teacher.visual = EasyDict(
                conv1=self.teacher.token_embedding,  # hack self.teacher.dtype
            )
        else:
            if teacher_cfg.get('with_preprocess', False):
                self._preprocess = clip.clip._transform(self.teacher.visual.input_resolution)
            if teacher_cfg.get('vpe_hook', True):
                self._vpe_forward_pre_hook = VpeForwardPreHook(
                    vpe=self.teacher.visual.attnpool.positional_embedding, 
                )
                self._vpe_forward_pre_hook.register(self.teacher.visual.attnpool)

    def customize_teacher(self, teacher: nn.Module) -> nn.Module:
        state_dict = teacher.state_dict()

        input_resolution = self._teacher_cfg.get('input_resolution')
        if input_resolution is not None and state_dict['input_resolution'] != input_resolution:
            # state_dict['input_resolution'] = input_resolution
            assert 'visual.proj' not in state_dict
            vpe_name = 'visual.attnpool.positional_embedding'
            source_resolution = state_dict['input_resolution'] // 32
            target_resolution = input_resolution // 32
            assert source_resolution ** 2 + 1 == state_dict[vpe_name].shape[0]

            cls_pos = state_dict[vpe_name][[0]]
            spatial_pos = state_dict[vpe_name][1:]
            spatial_pos = einops.rearrange(spatial_pos, '(h w) dim -> 1 dim h w', h=source_resolution)
            spatial_pos = F.interpolate(spatial_pos, size=(target_resolution,) * 2, mode='bilinear')  # TODO: supress warning
            spatial_pos = einops.rearrange(spatial_pos, '1 dim h w -> (h w) dim')
            vpe = torch.cat([cls_pos, spatial_pos])
            state_dict[vpe_name] = vpe

        context_length = self._teacher_cfg.get('context_length')
        if context_length is not None and state_dict['context_length'] > context_length:
            # state_dict['context_length'] = teacher_cfg.context_length
            state_dict['positional_embedding'] = state_dict['positional_embedding'][:context_length]

        teacher = clip.model.build_model(state_dict).float()
        return teacher

    @property
    def num_features(self) -> int:
        return self.teacher.visual.output_dim

    # @property
    # def device(self) -> torch.device:
    #     return self._prompt_forward_hook._prompt.device


@DETECTORS.register_module()
@CLIPDistiller.wrap()
class DenseCLIP_RetinaNet(SingleStageDetector):
    CLASSES: Tuple[str]
    distiller: CLIPDistiller
    backbone: CLIPResNetWithAttention
    bbox_head: RetinaRPNHead

    def __init__(
        self, 
        *args,
        context_decoder: dict,
        refine: bool = True,
        **kwargs,
    ):
        # super().__init__(init_cfg)
        super().__init__(*args, pretrained=None, **kwargs)
        self._context_decoder = ContextDecoder(**context_decoder) if refine else None
        self._gamma = nn.Parameter(torch.FloatTensor(data=[1e-4]))
        self._classifier = Classifier()  # TODO: bias need to be specified as -4.18
        self.bbox_head.retina_cls = nn.Conv2d(
            self.bbox_head.feat_channels, 
            self.backbone.output_dim,  # NOTE: assumes single anchor
            kernel_size=3, padding=1,
            bias=False,
        )

    def _init_with_distiller(self):
        self.distiller._vpe_forward_pre_hook.register(self.backbone.attnpool)
        self.bbox_head.retina_cls.register_forward_hook(self._classifier)

    def extract_feat(self, img: torch.Tensor, **kwargs) -> torch.Tensor:
        x, visual_embeddings = self.backbone(img)
        visual_embeddings = einops.rearrange(visual_embeddings, 'n b c -> b n c')

        classes = self.CLASSES if self.training else CocoGZSLDataset.CLASSES
        text_embeddings = self.distiller.encode_text(classes)
        text_embeddings = einops.repeat(text_embeddings, 'k c -> b k c', b=img.shape[0])
        text_embeddings = self.refine_text_embeddings(visual_embeddings, text_embeddings)
        self._classifier.set_weight(text_embeddings)

        x = self.neck(x)

        if self.training:
            return x, visual_embeddings

        return x

    def refine_text_embeddings(self, visual_embeddings: torch.Tensor, text_embeddings: torch.Tensor) -> torch.Tensor:
        if self._context_decoder is not None:
            context_embeddings = self._context_decoder(text_embeddings, visual_embeddings)
            text_embeddings = text_embeddings + self._gamma * context_embeddings
        return text_embeddings

    def forward_train(
        self,
        img: torch.Tensor,
        img_metas: List[dict],
        gt_bboxes: List[torch.Tensor],
        gt_labels: List[torch.Tensor],
        gt_bboxes_ignore: Optional[List[torch.Tensor]] = None,
    ) -> Dict[str, Any]:
        super(SingleStageDetector, self).forward_train(img, img_metas)
        x, visual_embeddings = self.extract_feat(img)
        losses, proposal_list = self.bbox_head.forward_train(
            x, img_metas, gt_bboxes, gt_labels, gt_bboxes_ignore, 
        )

        for i, proposals in enumerate(proposal_list):
            proposals[:, 2:4] -= proposals[:, :2]
            proposals[:, -1] = i  # record bs

        h, w = x[2].shape[-2:]
        cls_embedding = visual_embeddings[:, 0, :]
        spatial_embeddings = visual_embeddings[:, 1:, :]
        spatial_embeddings = einops.rearrange(spatial_embeddings, 'b (h w) c -> b c h w', h=h, w=w)

        kd_losses = self.distiller.distill(dict(
            batch_input_shape=img_metas[0]['batch_input_shape'],
            gt_bboxes=gt_bboxes,
            gt_labels=gt_labels,
            seg_map=self._classifier(None, None, spatial_embeddings, with_bias=False),
            image_features=cls_embedding,
            teacher_image_features=self.distiller.teacher.encode_image(img),
            teacher_crop_features=torch.cat([
                encode_bboxes(self.distiller.teacher, self.distiller._preprocess, img_, proposals[:, :4]) 
                for img_, proposals in zip(img, proposal_list)
            ]),
            crop_indices=torch.cat([proposals[:, [5, 8, 6, 7]] for proposals in proposal_list])
        ))
        return {**losses, **kd_losses}

    def forward_test(self, *args, **kwargs) -> Any:
        old_coc = self.bbox_head.cls_out_channels
        old_nc = self.bbox_head.num_classes

        self.bbox_head.cls_out_channels -= self.bbox_head.num_classes
        self.bbox_head.num_classes = len(CocoGZSLDataset.CLASSES)
        self.bbox_head.cls_out_channels += self.bbox_head.num_classes

        result = super().forward_test(*args, **kwargs)
        self.distiller.reset()

        self.bbox_head.cls_out_channels = old_coc
        self.bbox_head.num_classes = old_nc
        return result

    def forward(self, *args, **kwargs) -> Any:
        result = super().forward(*args, **kwargs)
        self._classifier.set_weight(None)
        return result


@DETECTORS.register_module()
@CLIPDistiller.wrap()
class DenseCLIP_RetinaNetKD(SingleStageDetector):
    CLASSES: Tuple[str]
    distiller: CLIPDistiller
    backbone: CLIPResNetWithAttention
    bbox_head: RetinaHead

    def __init__(
        self, 
        *args,
        context_decoder: dict,
        refine: bool = True,
        **kwargs,
    ):
        # super().__init__(init_cfg)
        super().__init__(*args, pretrained=None, **kwargs)
        self._context_decoder = ContextDecoder(**context_decoder) if refine else None
        self._gamma = nn.Parameter(torch.FloatTensor(data=[1e-4]))
        self._classifier = Classifier()
        self.bbox_head.retina_cls = nn.Conv2d(
            self.bbox_head.feat_channels, 
            self.backbone.output_dim,  # NOTE: assumes single anchor
            kernel_size=3, padding=1,
            bias=False,
        )

    def _init_with_distiller(self):
        self.distiller._vpe_forward_pre_hook.register(self.backbone.attnpool)
        self.bbox_head.retina_cls.register_forward_hook(self._classifier)

    def extract_feat(self, img: torch.Tensor, **kwargs) -> torch.Tensor:
        x, visual_embeddings = self.backbone(img)
        visual_embeddings = einops.rearrange(visual_embeddings, 'n b c -> b n c')

        classes = self.CLASSES if self.training else CocoGZSLDataset.CLASSES
        text_embeddings = self.distiller.encode_text(classes)
        text_embeddings = einops.repeat(text_embeddings, 'k c -> b k c', b=img.shape[0])
        text_embeddings = self.refine_text_embeddings(visual_embeddings, text_embeddings)
        self._classifier.set_weight(text_embeddings)

        x = self.neck(x)

        if self.training:
            return x, visual_embeddings

        return x

    def refine_text_embeddings(self, visual_embeddings: torch.Tensor, text_embeddings: torch.Tensor) -> torch.Tensor:
        if self._context_decoder is not None:
            context_embeddings = self._context_decoder(text_embeddings, visual_embeddings)
            text_embeddings = text_embeddings + self._gamma * context_embeddings
        return text_embeddings

    def forward_train(
        self,
        img: torch.Tensor,
        img_metas: List[dict],
        gt_bboxes: List[torch.Tensor],
        gt_labels: List[torch.Tensor],
        bboxes: List[torch.Tensor],
        bbox_embeddings: List[torch.Tensor],
        gt_bboxes_ignore: Optional[List[torch.Tensor]] = None,
    ) -> Dict[str, Any]:
        super(SingleStageDetector, self).forward_train(img, img_metas)
        x, visual_embeddings = self.extract_feat(img)
        losses = self.bbox_head.forward_train(
            x, img_metas, gt_bboxes, gt_labels, gt_bboxes_ignore, 
        )

        h, w = x[2].shape[-2:]
        cls_embedding = visual_embeddings[:, 0, :]
        spatial_embeddings = visual_embeddings[:, 1:, :]
        spatial_embeddings = einops.rearrange(spatial_embeddings, 'b (h w) c -> b c h w', h=h, w=w)

        teacher_crop_features = []
        crop_indices = []
        prior_generator: AnchorGeneratorWithPos = self.bbox_head.prior_generator
        with prior_generator.with_pos():
            anchor_list, valid_flag_list = self.bbox_head.get_anchors(
                featmap_sizes=[featmap.shape[-2:] for featmap in x], 
                img_metas=img_metas, device=img.device,
            )
        for i, (flat_anchors, valid_flags, img_meta, bbox, bbox_embedding) in enumerate(zip(
            anchor_list, valid_flag_list, img_metas, bboxes, bbox_embeddings,
        )):
            flat_anchors = torch.cat(flat_anchors)
            valid_flags = torch.cat(valid_flags)
            inside_flags = anchor_inside_flags(
                flat_anchors, valid_flags, img_meta['img_shape'][:2],
                self.bbox_head.train_cfg.allowed_border,
            )
            anchors = flat_anchors[inside_flags, :]
            assign_result: AssignResult = self.bbox_head.assigner.assign(
                anchors[:, :4], bbox, None, torch.arange(bbox.shape[0], device=img.device),
            )
            matched_flags = assign_result.labels != -1
            anchors = anchors[matched_flags]
            labels = assign_result.labels[matched_flags]
            teacher_crop_features.append(bbox_embedding[labels])
            anchors[:, -1] = i
            crop_indices.append(anchors[:, [4, 7, 5, 6]])

        kd_losses = self.distiller.distill(dict(
            batch_input_shape=img_metas[0]['batch_input_shape'],
            gt_bboxes=gt_bboxes,
            gt_labels=gt_labels,
            seg_map=self._classifier(None, None, spatial_embeddings, with_bias=False),
            image_features=cls_embedding,
            teacher_image_features=self.distiller.teacher.encode_image(img),
            teacher_crop_features=torch.cat(teacher_crop_features).float(),
            crop_indices=torch.cat(crop_indices),
        ))
        return {**losses, **kd_losses}

    def forward_test(self, *args, **kwargs) -> Any:
        old_coc = self.bbox_head.cls_out_channels
        old_nc = self.bbox_head.num_classes

        self.bbox_head.cls_out_channels -= self.bbox_head.num_classes
        self.bbox_head.num_classes = len(CocoGZSLDataset.CLASSES)
        self.bbox_head.cls_out_channels += self.bbox_head.num_classes

        result = super().forward_test(*args, **kwargs)
        self.distiller.reset()

        self.bbox_head.cls_out_channels = old_coc
        self.bbox_head.num_classes = old_nc
        return result

    def forward(self, *args, **kwargs) -> Any:
        result = super().forward(*args, **kwargs)
        self._classifier.set_weight(None)
        return result

