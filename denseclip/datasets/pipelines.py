from mmcv.parallel import DataContainer as DC
from mmdet.datasets import PIPELINES
from todd.datasets import build_access_layer


@PIPELINES.register_module()
class LoadEmbeddings:
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
