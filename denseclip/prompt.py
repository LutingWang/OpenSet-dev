import logging
from typing import Any, Dict, List, Optional, Tuple

import clip
import clip.model
import einops
import numpy as np
import todd
import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch.nn as nn
import torchvision.transforms as transforms
from easydict import EasyDict
from mmdet.datasets import DATASETS
from mmcv import ConfigDict
from mmcv.parallel import DataContainer as DC
from mmcv.runner import BaseModule
from mmdet.models import DETECTORS
from mmdet.datasets import CocoDataset
from todd.datasets import PthDataset
from tqdm import tqdm

from .utils import SimpleTokenizer, has_debug_flag


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
        nn.init.trunc_normal_(self._prompt)

    def register(self, module: nn.Module):
        module.register_forward_hook(self)

    def forward(self, module: nn.Module, input_: Any, output: Any):
        output[:, 1:1 + self._prompt_length] = self._prompt


class CLIPDistiller(todd.distillers.SingleTeacherDistiller):
    teacher: clip.model.CLIP

    def __init__(self, *args, teacher_cfg: ConfigDict, **kwargs):
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
        else:
            self._tokenizer = SimpleTokenizer(
                bpe_path='clip/bpe_simple_vocab_16e6.txt.gz', 
                context_length=teacher_cfg.context_length,
                prompt_length=teacher_cfg.prompt_length,
            )
            self._prompt_forward_hook = PromptFrowardHook(
                prompt_length=teacher_cfg.prompt_length, 
                embedding_dim=self.teacher.token_embedding.embedding_dim,
            )
            self._prompt_forward_hook.register(self.teacher.token_embedding)

        if teacher_cfg.get('text_only', False):
            delattr(self.teacher, 'visual')
            self.teacher.visual = EasyDict(
                conv1=self.teacher.token_embedding,  # hack self.teacher.dtype
            )
        else:
            self._preprocess = transforms.Compose([
                transforms.Resize(teacher_cfg.input_resolution, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(teacher_cfg.input_resolution),
            ])
            self._vpe_forward_pre_hook = VpeForwardPreHook(
                vpe=self.teacher.visual.attnpool.positional_embedding, 
            )
            self._vpe_forward_pre_hook.register(self.teacher.visual.attnpool)

    def customize_teacher(self, teacher: nn.Module) -> nn.Module:
        state_dict = teacher.state_dict()
        assert 'visual.proj' not in state_dict
        vpe_name = 'visual.attnpool.positional_embedding'

        input_resolution = self._teacher_cfg.get('input_resolution')
        if input_resolution is not None and state_dict['input_resolution'] != input_resolution:
            # state_dict['input_resolution'] = input_resolution
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

    @property
    def device(self) -> torch.device:
        return self._prompt_forward_hook._prompt.device

    @torch.no_grad()
    def encode_image(self, image: torch.Tensor, norm: bool = False) -> torch.Tensor:
        image = self._preprocess(image)
        image_embeddings = self.teacher.encode_image(image)
        if norm:
            image_embeddings = F.normalize(image_embeddings)
        return image_embeddings

    def encode_text(self, texts: Tuple[str], norm: bool = False) -> torch.Tensor:
        tokens = self._tokenizer.tokenize(texts, self.device)
        text_embeddings = self.teacher.encode_text(tokens)
        if norm:
            text_embeddings = F.normalize(text_embeddings)
        return text_embeddings


class Classifier(nn.Module):
    def __init__(self, tau: Tuple[float, float] = (0.07, 0.07), bias: Optional[float] = None):
        super().__init__()
        if isinstance(tau, float):
            tau = (tau, tau)
        self._tau = tau
        self._bias = (
            None if bias is None else 
            nn.Parameter(torch.FloatTensor(data=[bias]))
        )

    @property
    def tau(self) -> float:
        return self._tau[self.training]

    def set_weight(self, weight: Optional[torch.Tensor], norm: bool = True):
        if weight is None:
            assert not norm
        if norm:
            weight = F.normalize(weight)
        self._weight = weight

    def forward_hook(
        self, module: Any, input_: Any, output: torch.Tensor, 
    ) -> torch.Tensor:
        return self.forward(output)

    def forward(self, x: torch.Tensor, norm: bool = True) -> torch.Tensor:
        if norm:
            x = F.normalize(x)
        if self._weight is None:
            return x
        if x.ndim == 2:
            input_pattern = 'b c'
            output_pattern = 'b k'
        elif x.ndim == 4:
            input_pattern = 'b c h w'
            output_pattern = 'b k h w'
        if self._weight.ndim == 2:
            weight_pattern = 'k c'
        elif self._weight.ndim == 3:
            weight_pattern = 'b k c'
        x = torch.einsum(
            f'{input_pattern}, {weight_pattern} -> {output_pattern}', 
            x, self._weight,
        )
        x = x / self.tau
        if self._bias is not None:
            x = x + self._bias
        return x


@DATASETS.register_module()
class PromptDataset(PthDataset):
    CLASSES = CocoDataset.CLASSES

    def __init__(self, *args, test_mode: bool = False, **kwargs):
        super().__init__(*args, task_name='val' if test_mode else 'train', map_indices=True, **kwargs)
        self._test_mode = test_mode
        self.flag = np.zeros(len(self), dtype=np.uint8)  # for GroupSamper to read

    def __len__(self) -> int:
        if has_debug_flag(2):
            return 5
        return super().__len__()

    def __getitem__(self, *args, **kwargs) -> Dict[str, DC]:
        proposals, crops, max_overlaps, labels = super().__getitem__(*args, **kwargs)
        item = dict(crops=DC(crops.float()))
        if not self._test_mode:
            item['labels'] = DC(labels)
        return item

    def evaluate(self, results: List[torch.Tensor], logger: Optional[logging.Logger] = None, **kwargs) -> Dict[str, float]:
        if logger is None:
            logger = self._logger
        pos_acc = todd.utils.Accuracy(topks=(1, 2, 3, 5, 10))
        neg_acc = todd.utils.BinaryAccuracy(thrs=np.linspace(0.05, 0.5, 10))
        for i, result in tqdm(enumerate(results)):
            proposals, crops, max_overlaps, labels = super().__getitem__(i)
            pos_acc.evaluate(result[labels >= 0], labels[labels >= 0].to(result.device))
            neg_acc.evaluate(result.softmax(-1), labels.to(result.device))
        logger.info(f"Positive Accuracies\n{pos_acc}\n")
        logger.info(f"Negative Accuracies\n{neg_acc}\n")
        pos_acc = {f'top{k}': v for k, v in pos_acc.todict().items()}
        neg_acc = {f'thr{k:.1f}': v for k, v in neg_acc.todict()['accuracies'].items()}
        return {**pos_acc, **neg_acc}


@DETECTORS.register_module()
@CLIPDistiller.wrap()
class PromptTrainer(BaseModule):
    CLASSES: Tuple[str]
    distiller: CLIPDistiller
    
    def __init__(self, *args, **kwargs):
        super().__init__()
        self._classifier = Classifier(tau=0.01)

    def forward_single(
        self, 
        crops: torch.Tensor,
    ) -> torch.Tensor:
        # if has_debug_flag(4):
        #     crops = crops.float()
        label_embeddings = self.distiller.encode_text(self.CLASSES, norm=True)
        self._classifier.set_weight(label_embeddings, norm=False)
        logits = self._classifier(crops)
        return logits

    def forward_test(
        self, 
        crops: List[torch.Tensor],
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        return [self.forward_single(crop) for crop in crops]

    def forward_train(
        self, 
        crops: List[torch.Tensor],
        labels: List[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        b = len(crops)
        crops = torch.cat(crops)
        labels = torch.cat(labels)

        logits = self.forward_single(crops)
        pos_inds = labels >= 0
        loss_pos = F.cross_entropy(logits[pos_inds], labels[pos_inds], reduction="sum")
        neg_inds = labels == -1
        loss_neg = -F.log_softmax(logits[neg_inds], dim=1).sum() / len(self.CLASSES)
        return dict(loss_pos=loss_pos / b, loss_neg=loss_neg / b)

    def forward(self, *args, return_loss=True, **kwargs):
        if return_loss:
            return self.forward_train(*args, **kwargs)
        else:
            return self.forward_test(*args, **kwargs)

    def train_step(self, data: Dict[str, List[torch.Tensor]], optimizer: Optional[torch.optim.Optimizer] = None):
        losses = self(**data)
        loss, log_vars = self._parse_losses(losses)
        outputs = dict(loss=loss, log_vars=log_vars, num_samples=len(data['crops']))
        return outputs

    def _parse_losses(self, losses: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
        loss = sum(losses.values())
        log_vars = dict(loss=loss.item())
        for k, v in losses.items():
            if dist.is_available() and dist.is_initialized():
                v = v.data.clone()
                v.div_(dist.get_world_size())
                dist.all_reduce(v)
            log_vars[k] = v.item()
        return loss, log_vars

    def val_step(self, *args, **kwargs):
        return self.train_step(*args, **kwargs)
