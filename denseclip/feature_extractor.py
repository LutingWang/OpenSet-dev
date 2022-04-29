import os
from typing import Any, Dict, List, Optional, Tuple

import PIL.Image as Image
import clip
import clip.model
import cv2
import einops
import numpy as np
import todd.datasets
import todd.utils
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.parallel import DataContainer as DC
from mmdet.datasets import DATASETS, PIPELINES
from mmdet.models import DETECTORS, BaseDetector
from todd.utils import BBoxes

from .utils import has_debug_flag


@PIPELINES.register_module()
class LoadImageFromRegions:
    def __init__(self, n_px: int):
        self._n_px = n_px
        self._preprocess = clip.clip._transform(n_px)

    def __call__(self, results: dict) -> dict:
        if 'img_fields' not in results or len(results['img_fields']) == 0:
            filename = os.path.join(
                results['img_prefix'], results['img_info']['filename'],
            )
            image = Image.open(filename)
        elif len(results['img_fields']) == 1:
            assert results['img_fields'][0] == 'img', results['img_fields']
            image: np.ndarray = results['img']
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image: Image.Image = Image.fromarray(image)
        else:
            raise ValueError(f"Unexpected img_fields: {results['img_fields']}")
        results['img_fields'] = ['img']

        if len(results['bbox_fields']) == 1:
            assert results['bbox_fields'][0] == 'proposals', results['bbox_fields']
            bbox_field = 'proposals'
        elif len(results['bbox_fields']) == 2:
            assert 'gt_bboxes' in results['bbox_fields'], results['bbox_fields']
            assert 'gt_bboxes_ignore' in results['bbox_fields'], results['bbox_fields']
            bbox_field = 'gt_bboxes'
        else:
            raise ValueError(f"Unexpected bbox_fields: {results['bbox_fields']}")
        results['bbox_fields'] = ['bboxes']

        bboxes = results[bbox_field]
        if has_debug_flag(3):
            bboxes = bboxes[:5]
        bboxes = BBoxes(torch.as_tensor(bboxes))
        # bboxes = bboxes[bboxes.areas > 32 * 32]
        results['bboxes'] = DC(bboxes.to_tensor())

        if bboxes.empty:
            regions = torch.zeros(0, 3, self._n_px, self._n_px)
        else:
            bboxes15 = bboxes.expand(ratio=1.5, image_shape=(image.height, image.width))
            bboxes = bboxes + bboxes15
            regions = torch.stack([
                self._preprocess(image.crop(bbox)) 
                for bbox in bboxes.round().to_tensor().int().tolist()
            ])
        image.close()
        results['img'] = DC(regions)

        results['id'] = results['img_info']['id']
        return results


@DETECTORS.register_module()
class CLIPFeatureExtractor(BaseDetector):
    CLASSES: Tuple[str]

    def __init__(self, *args, clip_model: str = 'pretrained/clip/RN50.pt', data_root: str, **kwargs):
        super().__init__()
        model, _ = clip.load(clip_model)
        self._model: clip.model.CLIP = todd.utils.freeze_model(model)
        self._loss = nn.Parameter(torch.zeros([]), requires_grad=True)
        self._train_writer = todd.datasets.PthAccessLayer(data_root=data_root, task_name='train', readonly=False, exist_ok=True)
        self._val_writer = todd.datasets.PthAccessLayer(data_root=data_root, task_name='val', readonly=False, exist_ok=True)

    def train(self, mode: bool = False) -> 'CLIPFeatureExtractor':
        super().train(False)
        return self

    def simple_test(self):
        raise NotImplementedError

    def aug_test(self):
        raise NotImplementedError

    @torch.no_grad()
    def extract_feat(
        self, 
        img: List[torch.Tensor],
        img_metas: List[dict], 
        *args, 
        writer: todd.datasets.PthAccessLayer,
        bboxes: List[torch.Tensor], 
        **kwargs,
    ):
        batch_sizes = [img_.shape[0] for img_ in img]
        imgs = torch.cat(img)
        if imgs.shape[0] == 0:
            crops = [
                imgs.new_zeros(0, self._model.visual.output_dim) 
                for _ in range(len(img))
            ]
        else:
            imgs: torch.Tensor = self._model.encode_image(imgs)
            crops = [
                F.normalize(einops.reduce(
                    crops, '(n b) d -> b d', n=2, reduction='sum',
                ), dim=-1) 
                for crops in imgs.split(batch_sizes)
            ]

        for img_meta, value in zip(img_metas, zip(bboxes, crops)):
            writer[img_meta['id']] = value

    def forward_train(self, *args, **kwargs) -> Dict[str, Any]:
        self.extract_feat(*args, writer=self._train_writer, **kwargs)
        return dict(loss_zero=self._loss * 0)

    def forward_test(
        self,
        img: List[torch.Tensor],
        *args, 
        **kwargs,
    ) -> List[List[np.ndarray]]:
        self.extract_feat(img, *args, writer=self._val_writer, **kwargs)
        bbox_results = [[
            np.zeros((0, 5), dtype=np.float32)
        ] * len(self.CLASSES)] * len(img)
        return bbox_results
