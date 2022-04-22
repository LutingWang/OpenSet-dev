import os

import PIL.Image as Image
import clip
import torch
from mmcv.parallel import DataContainer as DC
from mmdet.datasets import PIPELINES
from todd.datasets import build_access_layer
from todd.utils import BBoxes

from ..utils import has_debug_flag


@PIPELINES.register_module()
class LoadImageFromRegions:
    def __init__(self, n_px: int):
        self._n_px = n_px
        self._preprocess = clip.clip._transform(n_px)

    def __call__(self, results: dict) -> dict:
        results['id'] = results['img_info']['id']

        assert len(results['bbox_fields']) == 1
        bbox_field = results['bbox_fields'][0]
        results['bbox_fields'][0] = 'bboxes'

        bboxes = results[bbox_field]
        if has_debug_flag(3):
            bboxes = bboxes[:5]
        bboxes = BBoxes(torch.as_tensor(bboxes))
        bboxes = bboxes[bboxes.areas > 32 * 32]
        results['bboxes'] = DC(bboxes.to_tensor())

        if bboxes.empty:
            results['img'] = DC(torch.zeros(0, 3, self._n_px, self._n_px))
            return results

        assert results['img_prefix'] is not None
        filename = os.path.join(
            results['img_prefix'], results['img_info']['filename'],
        )
        with Image.open(filename) as image:
            bboxes15 = bboxes.expand(ratio=1.5, image_shape=(image.height, image.width))
            bboxes = bboxes + bboxes15
            img = torch.stack([
                self._preprocess(image.crop(bbox)) 
                for bbox in bboxes.round().to_tensor().int().tolist()
            ])
        results['img'] = DC(img)
        return results


@PIPELINES.register_module()
class LoadProposalEmbeddings:
    def __init__(self, data_root: str):
        self._pth_access_layer =  build_access_layer(dict(
            type='PthAccessLayer',
            data_root=data_root,
            task_name='train',
        ))

    def __call__(self, results: dict) -> dict:
        id_ = results['img_info']['id']
        bboxes, bbox_embeddings = self._pth_access_layer[id_]
        results['bboxes'] = DC(bboxes)
        results['bbox_embeddings'] = DC(bbox_embeddings)
        return results
