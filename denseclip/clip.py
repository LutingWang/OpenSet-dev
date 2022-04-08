import io
import os
import shutil
from typing import Any, Dict, List, NamedTuple, Optional, Tuple, Union

import numpy as np

import clip
import clip.model
import lmdb
from mmcv.runner import BaseRunner, HOOKS, Hook, get_dist_info
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from denseclip.utils import encode_bboxes, has_debug_flag
import todd
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as tf
from mmdet.models import DETECTORS, BaseDetector

from .coco import CocoGZSLDataset


class Evaluator:
    def __init__(self) -> None:
        self._total = 0
        self._correct = 0

    def add(self, result: torch.Tensor):
        self._total += result.numel()
        self._correct += result.sum()

    def report(self):
        print("total:", self._total)
        print("correct:", self._correct)
        print("acc:", self._correct / self._total)


class CLIPBaseDetector(BaseDetector):
    CLASSES: Tuple[str]

    def __init__(self):
        super().__init__()
        model, preprocess = clip.load('pretrained/RN50.pt')
        # todd.utils.freeze_model(model)
        model.eval()
        preprocess.transforms = preprocess.transforms[:2]
        self._model: clip.model.CLIP = model
        self._preprocess = preprocess

    def train(self, mode: bool = False):
        super().train(False)
        return self

    @property
    def device(self):
        return self._model.visual.conv1.weight.device

    def extract_feat(self):
        pass

    def simple_test(self):
        pass

    def aug_test(self):
        pass

    def forward_train(
        self,
        img: torch.Tensor,
        gt_bboxes: List[torch.Tensor],
    ) -> Tuple[List[torch.Tensor], Dict[str, torch.Tensor]]:
        # if has_debug_flag(3):
        #     gt_bboxes = [gt_bbox[:2] for gt_bbox in gt_bboxes]
        crops = [
            encode_bboxes(self._model, self._preprocess, img_, gt_bbox)
            for img_, gt_bbox in zip(img, gt_bboxes)
        ]
        losses = dict(loss_zero=img.new_zeros([], requires_grad=True))
        return crops, losses

    def forward_test(self, img: torch.Tensor):
        bbox_results = [[
            np.zeros((0, 5), dtype=np.float32)
        ] * len(self.CLASSES)] * img.shape[0]
        return bbox_results


@HOOKS.register_module()
class SaveLmdbToOSS(Hook):
    def __init__(self, lmdb_filepath: str):
        super().__init__()
        self._lmdb_filepath = lmdb_filepath

    def after_train_epoch(self, runner: BaseRunner):
        super().after_val_epoch(runner)
        rank, _ = get_dist_info()
        lmdb_filepath = os.path.join(self._lmdb_filepath, f'worker{rank}')
        os.makedirs(lmdb_filepath)

        model: Union[MMDataParallel, MMDistributedDataParallel] = runner.model
        module: CLIPFeatureExtractor = model.module
        module._env.copy(lmdb_filepath)


@DETECTORS.register_module()
class CLIPFeatureExtractor(CLIPBaseDetector):
    CLASSES: Tuple[str]

    def __init__(self, *args, lmdb_filepath: str, **kwargs):
        super().__init__()
        rank, _ = get_dist_info()
        self._lmdb_filepath = lmdb_filepath
        self._env: lmdb.Environment = lmdb.open(
            os.path.join(self._lmdb_filepath, f'worker{rank}'), 
            map_size=10 * 2 ** 30,  # 10 GB
            readonly=False, 
            max_dbs=2,
        )
        self._train_db: lmdb._Database = self._env.open_db('train'.encode(), create=True)
        self._val_db: lmdb._Database = self._env.open_db('val'.encode(), create=True)

    def _write_db(
        self, 
        db: lmdb._Database, 
        img_metas: List[dict], 
        gt_bboxes: List[torch.Tensor], 
        crops: List[torch.Tensor],
    ):
        img_ids = [
            str(img_meta['img_info']['id']).encode() 
            for img_meta in img_metas
        ]
        with self._env.begin(db, write=True) as txn:
            for crop, img_id, gt_bbox in zip(crops, img_ids, gt_bboxes):
                buffer = io.BytesIO()
                torch.save((gt_bbox, crop), buffer)
                txn.put(img_id, buffer.getvalue())

    def forward_train(
        self,
        img: torch.Tensor,
        img_metas: List[dict],
        gt_bboxes: List[torch.Tensor],
        gt_labels: List[torch.Tensor],
        gt_bboxes_ignore: Optional[List[torch.Tensor]] = None,
        *args, **kwargs,
    ) -> Dict[str, Any]:
        crops, losses = super().forward_train(img, gt_bboxes)
        self._write_db(self._train_db, img_metas, gt_bboxes, crops)
        return losses

    def forward_test(
        self,
        img: torch.Tensor,
        img_metas: List[dict],
        gt_bboxes: List[torch.Tensor],
        gt_labels: List[torch.Tensor],
        gt_bboxes_ignore: Optional[List[torch.Tensor]] = None,
        *args, **kwargs,
    ) -> Dict[str, Any]:
        crops, _ = super().forward_train(img, gt_bboxes)
        self._write_db(self._val_db, img_metas, gt_bboxes, crops)
        return super().forward_test(img)


@DETECTORS.register_module()
class CLIPDetector(CLIPBaseDetector):
    CLASSES: Tuple[str]

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
