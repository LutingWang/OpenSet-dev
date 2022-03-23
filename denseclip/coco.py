import contextlib
import functools
import io
import logging
from typing import Literal, Optional, Tuple

from mmcv.utils import print_log
from mmdet.datasets import CocoDataset, DATASETS
from mmdet.datasets.api_wrappers import COCOeval
import numpy as np


SEEN_65_15 = (
    'person', 'bicycle', 'car', 'motorcycle', 'bus', 'truck', 'boat',
    'traffic light', 'fire hydrant',
    'stop sign', 'bench', 'bird', 'dog',
    'horse', 'sheep', 'cow', 'elephant', 'zebra', 'giraffe',
    'backpack', 'umbrella', 'handbag', 'tie',
    'skis', 'sports ball', 'kite', 'baseball bat',
    'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'wine glass', 'cup', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'orange', 'broccoli', 'carrot',
    'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'dining table', 'tv', 'laptop',
    'remote', 'keyboard', 'cell phone', 'microwave',
    'oven', 'sink', 'refrigerator', 'book', 'clock',
    'vase', 'scissors', 'teddy bear', 'toothbrush',
)
UNSEEN_65_15 = (
    'airplane', 'train', 'parking meter', 'cat', 'bear', 'suitcase', 'frisbee', 'snowboard', 'fork',
    'sandwich', 'hot dog', 'toilet', 'mouse', 'toaster', 'hair drier',
)

SEEN_48_17 = (
    'person', 'bicycle', 'car', 'motorcycle', 'truck', 'boat', 'bench', 'bird', 'horse', 'sheep',
    'zebra', 'giraffe', 'backpack', 'handbag', 'skis', 'kite', 'surfboard', 'bottle', 'spoon',
    'bowl', 'banana', 'apple', 'orange', 'broccoli', 'carrot', 'pizza', 'donut', 'chair', 'bed',
    'tv', 'laptop', 'remote', 'microwave', 'oven', 'refrigerator', 'book', 'clock', 'vase',
    'toothbrush', 'train', 'bear', 'suitcase', 'frisbee', 'fork', 'sandwich', 'toilet', 'mouse',
    'toaster',
)
UNSEEN_48_17 = (
    'bus', 'dog', 'cow', 'elephant', 'umbrella', 'tie', 'skateboard', 'cup', 'knife', 'cake',
    'couch', 'keyboard', 'sink', 'scissors', 'airplane', 'cat', 'snowboard',
)


@DATASETS.register_module()
class CocoZSLDataset(CocoDataset):
    @staticmethod
    @functools.cache
    def index(split: Literal['SEEN_65_15', 'UNSEEN_65_15', 'SEEN_48_17', 'UNSEEN_48_17']) -> Tuple[int]:
        indices = [i for i, class_name in enumerate(CocoDataset.CLASSES) if class_name in eval(split)]
        return indices

    # def __len__(self):
    #     return 50

    def summarize(self, cocoEval: COCOeval, logger=None, split: Optional[str] = None) -> dict:
        redirect_string = io.StringIO()
        with contextlib.redirect_stdout(redirect_string):
            cocoEval.summarize()
        eval_results = {
            'bbox_' + metric: round(cocoEval.stats[i], 3) 
            for i, metric in enumerate([
                'mAP', 'mAP_50', 'mAP_75', 'mAP_s', 'mAP_m', 'mAP_l',
            ])
        }
        eval_results['bbox_mAP_copypaste'] = ' '.join(str(ap) for ap in eval_results.values())
        if split is not None:
            print_log(f'Evaluate split *{split}*', logger=logger)
            eval_results = {f'{split}_{k}': v for k, v in eval_results.items()}
        print_log('\n' + redirect_string.getvalue(), logger=logger)
        return eval_results

    def evaluate(
        self, results, metric='bbox', logger=None,
        iou_thrs: Optional[Tuple[float]] = None,
        max_dets: Optional[Tuple[int]] = (100, 300, 1000),
    ):
        predictions = self._det2json(results)

        cocoGt = self.coco
        try:
            cocoDt = self.coco.loadRes(predictions)
        except IndexError:
            print_log(
                'The testing results of the whole dataset is empty.',
                logger=logger, level=logging.ERROR,
            )
            return
        cocoEval = COCOeval(cocoGt, cocoDt, 'bbox')
        # cocoEval.params.catIds = self.cat_ids
        # cocoEval.params.imgIds = self.img_ids
        # cocoEval.params.imgIds = [ann['image_id'] for ann in cocoDt.loadAnns(cocoDt.getAnnIds())]
        if iou_thrs is not None:
            cocoEval.params.iouThrs = np.array(iou_thrs)
        if max_dets is not None:
            cocoEval.params.maxDets = list(max_dets)
        cocoEval.evaluate()
        cocoEval.accumulate()

        eval_results = self.summarize(cocoEval, logger)

        precision: np.ndarray = cocoEval.eval['precision']  # Thresh x Recall x K x Area x MaxDets
        recall: np.ndarray = cocoEval.eval['recall']  # Thresh x K x Area x MaxDets
        assert len(self.cat_ids) == precision.shape[2] == recall.shape[1]

        for split in ['SEEN_65_15', 'UNSEEN_65_15', 'SEEN_48_17', 'UNSEEN_48_17']:
            cocoEval.eval['precision'] = precision[:, :, self.index(split), :, :]
            cocoEval.eval['recalls'] = recall[:, self.index(split), :, :]
            eval_results.update(self.summarize(cocoEval, logger, split=split))

        return eval_results
