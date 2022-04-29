from pathlib import Path

from mmdet.datasets import PIPELINES
from todd.datasets import build_access_layer

from ..utils import has_debug_flag


@PIPELINES.register_module()
class LoadPthEmbeddings:
    def __init__(self, data_root: str, task_name: str = 'train'):
        self._pth_access_layer =  build_access_layer(dict(
            type='PthAccessLayer',
            data_root=data_root,
            task_name=task_name,
        ))

    def __call__(self, results: dict) -> dict:
        id_ = results['img_info']['id']
        bboxes, bbox_embeddings = self._pth_access_layer[id_]
        results['bbox_fields'].append('bboxes')
        results['bboxes'] = bboxes.numpy()
        results['bbox_embeddings'] = bbox_embeddings.numpy()
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
