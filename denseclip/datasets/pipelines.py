import math
from pathlib import Path
from typing import Optional, cast

import numpy as np

import clip.clip
import cv2
from PIL import Image
from mmcv.parallel import DataContainer as DC
from mmdet.datasets import PIPELINES
from mmdet.datasets.pipelines.loading import LoadImageFromFile
from todd.datasets import build_access_layer
from todd import BBoxesXYXY

from ..utils import has_debug_flag


@PIPELINES.register_module()
class LoadPthEmbeddings:
    def __init__(
        self, 
        data_root: str, 
        task_name: str = 'train', 
        min_bbox_area: Optional[int] = None,
        detpro: bool = False,
        sampling_ratio: Optional[float] = None,
        with_leading_zeros: bool = ...,
    ):
        self._pth_access_layer =  build_access_layer(dict(
            type='PthAccessLayer',
            data_root=data_root,
            task_name=task_name,
        ))
        self._min_bbox_area = min_bbox_area
        self._detpro = detpro
        self._sampling_ratio = sampling_ratio
        if with_leading_zeros is ...:
            with_leading_zeros = detpro
        self._with_leading_zeros = with_leading_zeros

    def __call__(self, results: dict) -> dict:
        if self._with_leading_zeros:
            id_ = Path(results['img_info']['filename']).stem
            if has_debug_flag(1):
                id_ = '000000000030'
        else:
            id_ = results['img_info']['id']
        if self._detpro:
            bbox_embeddings = self._pth_access_layer[id_]
            bboxes = results['proposals'][:, :4]
            bbox_embeddings = bbox_embeddings.numpy()
        else:
            bboxes, bbox_embeddings = self._pth_access_layer[id_]
            bboxes = bboxes.numpy()
            bbox_embeddings = bbox_embeddings.numpy()
        bboxes = cast(np.ndarray, bboxes)
        bbox_embeddings = cast(np.ndarray, bbox_embeddings)
        if self._min_bbox_area is not None:
            indices = BBoxesXYXY(bboxes).areas > self._min_bbox_area
            bboxes = bboxes[indices]
            bbox_embeddings = bbox_embeddings[indices]
        if self._sampling_ratio is not None:
            indices = math.ceil(len(bboxes) * self._sampling_ratio)
            bboxes = bboxes[:indices]
            bbox_embeddings = bbox_embeddings[:indices]
        if has_debug_flag(1):
            bbox_embeddings = bbox_embeddings[:bboxes.shape[0]]
        results['bbox_fields'].append('bboxes')
        results['bboxes'] = bboxes
        results['bbox_embeddings'] = bbox_embeddings
        return results


@PIPELINES.register_module()
class LoadZipEmbeddings:
    def __init__(self, data_root: str, task_name: str = 'train'):
        self._zip_access_layer =  build_access_layer(dict(
            type='ZipAccessLayer',
            data_root=data_root,
            task_name=task_name,
        ))

    def __call__(self, results: dict) -> dict:
        id_ = Path(results['img_info']['filename']).stem + '.pth'
        if has_debug_flag(1):
            id_ = '000000000030.pth'
        bboxes = results['proposals'][:, :4]
        bbox_embeddings = self._zip_access_layer[id_].numpy()
        if has_debug_flag(1):
            bbox_embeddings = bbox_embeddings[:bboxes.shape[0]]
        results['bbox_fields'].append('bboxes')
        results['bboxes'] = bboxes
        results['bbox_embeddings'] = bbox_embeddings
        return results


@PIPELINES.register_module()
class LoadUnzippedEmbeddings:
    def __init__(self, data_root: str, task_name: str = 'train'):
        self._pth_access_layer =  build_access_layer(dict(
            type='PthAccessLayer',
            data_root=data_root,
            task_name=task_name,
        ))

    def __call__(self, results: dict) -> dict:
        id_ = Path(results['img_info']['filename']).stem
        if has_debug_flag(1):
            id_ = '000000000030'
        bboxes = results['proposals'][:, :4]
        bbox_embeddings = self._pth_access_layer[id_].numpy()
        if has_debug_flag(1):
            bbox_embeddings = bbox_embeddings[:bboxes.shape[0]]
        results['bbox_fields'].append('bboxes')
        results['bboxes'] = bboxes
        results['bbox_embeddings'] = bbox_embeddings
        return results


@PIPELINES.register_module()
class LoadRawImageFromFile(LoadImageFromFile):
    def __init__(self, *args, n_px: int, **kwargs):
        super().__init__()
        self._preprocess = clip.clip._transform(n_px)

    def __call__(self, results: dict) -> dict:
        results = super().__call__(results)
        raw_image = cv2.cvtColor(results['img'], cv2.COLOR_BGR2RGB)
        raw_image = Image.fromarray(raw_image)
        raw_image = self._preprocess(raw_image)
        results['raw_image'] = raw_image
        return results


@PIPELINES.register_module()
class LoadImageEmbeddingFromFile:
    def __init__(self, data_root: str):
        self._pth_access_layer =  build_access_layer(dict(
            type='PthAccessLayer',
            data_root=data_root,
            task_name='train',
        ))

    def __call__(self, results: dict) -> dict:
        results['image_embeddings'] = self._pth_access_layer[results['img_info']['id']]
        return results
