from typing import Any, Dict, List, Optional, Tuple

import clip
import clip.model
import einops
import numpy as np
import todd.datasets
import todd.utils
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.models import DETECTORS, BaseDetector

from .datasets import CocoGZSLDataset
from .utils import encode_bboxes


class CLIPBaseDetector(BaseDetector):
    CLASSES: Tuple[str]

    def __init__(self, *args, pretrained: str = 'pretrained/clip/RN50.pt', **kwargs):
        super().__init__()
        model, _ = clip.load(pretrained)
        self._model: clip.model.CLIP = todd.utils.freeze_model(model)
        self._loss = nn.Parameter(torch.zeros([]), requires_grad=True)

    def train(self, mode: bool = False) -> 'CLIPBaseDetector':
        super().train(False)
        return self

    @property
    def device(self) -> torch.device:
        return self._model.visual.conv1.weight.device

    @torch.no_grad()
    def extract_feat(self, img: List[torch.Tensor]) -> List[torch.Tensor]:
        batch_sizes = [img_.shape[0] for img_ in img]
        imgs = torch.cat(img)
        if imgs.shape[0] == 0:
            return [
                imgs.new_zeros(0, self._model.visual.output_dim) 
                for _ in range(len(img))
            ]
        imgs: torch.Tensor = self._model.encode_image(imgs)
        return [F.normalize(
            einops.reduce(crops, '(n b) d -> b d', n=2, reduction='sum'), 
            dim=-1,
        ) for crops in imgs.split(batch_sizes)]

    def simple_test(self):
        pass

    def aug_test(self):
        pass

    def forward_train(self) -> Dict[str, torch.Tensor]:
        return dict(loss_zero=self._loss * 0)

    def forward_test(self, batch_size: int) -> List[List[np.ndarray]]:
        bbox_results = [[
            np.zeros((0, 5), dtype=np.float32)
        ] * len(self.CLASSES)] * batch_size
        return bbox_results


@DETECTORS.register_module()
class CLIPFeatureExtractor(CLIPBaseDetector):
    def __init__(self, *args, data_root: str, **kwargs):
        super().__init__(*args, **kwargs)
        self._train_writer = todd.datasets.PthAccessLayer(data_root=data_root, task_name='train', exist_ok=True)
        self._val_writer = todd.datasets.PthAccessLayer(data_root=data_root, task_name='val', exist_ok=True)

    def extract_feat(
        self, 
        img: torch.Tensor,
        img_metas: List[dict], 
        bboxes: List[torch.Tensor],
        training: bool = True,
        *args, **kwargs,
    ):
        writer = self._train_writer if training else self._val_writer
        crops = super().extract_feat(img)
        for img_meta, value in zip(img_metas, zip(bboxes, crops)):
            writer[img_meta['id']] = value

    def forward_train(self, *args, **kwargs) -> Dict[str, Any]:
        self.extract_feat(*args, training=True, **kwargs)
        return super().forward_train()

    def forward_test(
        self,
        img: List[torch.Tensor],
        *args, **kwargs,
    ) -> Dict[str, Any]:
        self.extract_feat(img, *args, training=False, **kwargs)
        return super().forward_test(batch_size=len(img))


@DETECTORS.register_module()
class CLIPDetector(CLIPBaseDetector):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self._text = None
        self._eval = Evaluator()

    def encode_text(self) -> torch.Tensor:
        if self._text is None:
            classes = ["a photo of a " + c for c in CocoGZSLDataset.CLASSES]
            text = clip.tokenize(classes).to(self.device)
            text = self._model.encode_text(text)
            text = F.normalize(text, dim=-1)
            self._text = text
        return self._text

    def forward_train(
        self,
        img: torch.Tensor,
        img_metas: List[dict],
        gt_bboxes: List[torch.Tensor],
        gt_labels: List[torch.Tensor],
        gt_bboxes_ignore: Optional[List[torch.Tensor]] = None,
        *args, **kwargs,
    ) -> Dict[str, Any]:
        for i, bboxes in enumerate(gt_bboxes):
            image_features = encode_bboxes(self._model, self._preprocess, img[i], bboxes)
            text_features = self.encode_text()
            logits = self._model.logit_scale.exp() * image_features @ text_features.t()
            _, inds = logits.max(-1)
            self._eval.add(inds == gt_labels[i])
        return dict(loss_zero=torch.zeros([], requires_grad=True))

    def forward_test(self, img: torch.Tensor, *args, **kwargs):
        pass
