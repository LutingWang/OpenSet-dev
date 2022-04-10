import io
import os
import random
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union


import clip
import clip.model
import lmdb
import numpy as np
import todd
import torch
import torch.nn.functional as F
from denseclip.utils import encode_bboxes, has_debug_flag
from mmcv.runner import BaseRunner, HOOKS, Hook, get_dist_info
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmdet.core import AssignResult, MaxIoUAssigner
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

    def __init__(self, *args, **kwargs):
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
        bboxes: List[torch.Tensor],
    ) -> Tuple[List[torch.Tensor], Dict[str, torch.Tensor]]:
        # if has_debug_flag(3):
        #     gt_bboxes = [gt_bbox[:2] for gt_bbox in gt_bboxes]
        crops = [
            encode_bboxes(self._model, self._preprocess, img_, bbox)
            for img_, bbox in zip(img, bboxes)
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
    def __init__(self, *args, lmdb_filepath: str, **kwargs):
        super().__init__(*args, **kwargs)
        rank, _ = get_dist_info()
        self._lmdb_filepath = lmdb_filepath
        if not os.path.exists(self._lmdb_filepath):
            os.makedirs(self._lmdb_filepath)
        self._env: lmdb.Environment = lmdb.open(
            os.path.join(self._lmdb_filepath, f'worker{rank}'), 
            map_size=10 * 2 ** 30,  # 10 GB
            readonly=False, 
            max_dbs=2,
        )

    def _write_db(
        self, 
        db: lmdb._Database, 
        img_metas: List[dict], 
        data: Iterable[Any],
    ):
        img_ids = [
            str(img_meta['img_info']['id']).encode() 
            for img_meta in img_metas
        ]
        with self._env.begin(db, write=True) as txn:
            for img_id, datum in zip(img_ids, data):
                buffer = io.BytesIO()
                torch.save(datum, buffer)
                txn.put(img_id, buffer.getvalue())


@DETECTORS.register_module()
class CLIPProposalFeatureExtractor(CLIPFeatureExtractor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._assigner = MaxIoUAssigner(
            pos_iou_thr=0.5,
            neg_iou_thr=(0.1, 0.5),
            min_pos_iou=0.5,
        )
        self._train_db: lmdb._Database = self._env.open_db('train'.encode(), create=True)
        self._val_db: lmdb._Database = self._env.open_db('val'.encode(), create=True)

    def assign(
        self, 
        proposals: torch.Tensor, 
        gt_bboxes: torch.Tensor, 
        gt_labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        assign_result: AssignResult = self._assigner.assign(proposals, gt_bboxes, None, gt_labels)
        pos_inds, = torch.where(assign_result.gt_inds > 0)
        neg_inds, = torch.where(assign_result.gt_inds == 0)
        neg_inds = random.sample(neg_inds.tolist(), neg_inds.numel() // 10)
        neg_inds = pos_inds.new_tensor(neg_inds)
        inds = torch.cat([pos_inds, neg_inds])
        return proposals[inds], assign_result.max_overlaps[inds], assign_result.labels[inds]

    def forward_train(
        self,
        img: torch.Tensor,
        img_metas: List[dict],
        gt_bboxes: List[torch.Tensor],
        gt_labels: List[torch.Tensor],
        gt_bboxes_ignore: Optional[List[torch.Tensor]] = None,
        proposals: Optional[torch.Tensor] = None,
        *args, **kwargs,
    ) -> Dict[str, Any]:
        max_overlaps = labels = [None] * len(proposals)
        for i, data in enumerate(zip(proposals, gt_bboxes, gt_labels)):
            proposals[i], max_overlaps[i], labels[i] = self.assign(*data)
            
        crops, losses = super().forward_train(img, proposals)
        self._write_db(self._train_db, img_metas, zip(proposals, crops, max_overlaps, labels))
        return losses

    def forward_test(
        self,
        img: torch.Tensor,
        img_metas: List[dict],
        gt_bboxes: List[torch.Tensor],
        gt_labels: List[torch.Tensor],
        gt_bboxes_ignore: Optional[List[torch.Tensor]] = None,
        proposals: Optional[torch.Tensor] = None,
        *args, **kwargs,
    ) -> Dict[str, Any]:
        max_overlaps = labels = [None] * len(proposals)
        for i, data in enumerate(zip(proposals, gt_bboxes, gt_labels)):
            proposals[i], max_overlaps[i], labels[i] = self.assign(*data)

        crops, _ = super().forward_train(img, proposals)
        self._write_db(self._val_db, img_metas, zip(proposals, crops, max_overlaps, labels))
        return super().forward_test(img)


@DETECTORS.register_module()
class CLIPGTFeatureExtractor(CLIPFeatureExtractor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._train_db: lmdb._Database = self._env.open_db('train'.encode(), create=True)
        self._val_db: lmdb._Database = self._env.open_db('val'.encode(), create=True)

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
        self._write_db(self._train_db, img_metas, zip(gt_bboxes, crops))
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
        self._write_db(self._val_db, img_metas, zip(gt_bboxes, crops))
        return super().forward_test(img)


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
