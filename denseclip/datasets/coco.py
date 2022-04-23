import contextlib
import io
import logging
from typing import Optional, Tuple

import numpy as np
from mmcv.utils import print_log
from mmdet.datasets import CocoDataset, DATASETS
from mmdet.datasets.api_wrappers import COCOeval
from mmdet.datasets.lvis import LVISV1Dataset

from ..utils import has_debug_flag

from .zsl import ZSLDataset


# TODO: distinguish unseen from novel


SEEN_65_15 = [
     0,  1,  2,  3,  5,  7,  8,  9, 10, 11, 13, 14, 16, 17, 18, 
    19, 20, 22, 23, 24, 25, 26, 27, 30, 32, 33, 34, 35, 36, 37, 
    38, 39, 40, 41, 43, 44, 45, 46, 47, 49, 50, 51, 53, 54, 55, 
    56, 57, 58, 59, 60, 62, 63, 65, 66, 67, 68, 69, 71, 72, 73, 
    74, 75, 76, 77, 79,
] 
UNSEEN_65_15 = [
     4,  6, 12, 15, 21, 28, 29, 31, 42, 48, 52, 61, 64, 70, 78,
]
SEEN_48_17 = [
     0,  1,  2,  3,  6,  7,  8, 13, 14, 17, 18, 21, 22, 23, 24, 
    26, 28, 29, 30, 33, 37, 39, 42, 44, 45, 46, 47, 48, 49, 50, 
    51, 53, 54, 56, 59, 61, 62, 63, 64, 65, 68, 69, 70, 72, 73, 
    74, 75, 79,
]
UNSEEN_48_17 = [
     4,  5, 15, 16, 19, 20, 25, 27, 31, 36, 41, 43, 55, 57, 66, 
    71, 76,
]
ALL_48_17 = sorted(SEEN_48_17 + UNSEEN_48_17)
INDEX_SEEN_48_17 = [i for i, c in enumerate(ALL_48_17) if c in SEEN_48_17]
INDEX_UNSEEN_48_17 = [i for i, c in enumerate(ALL_48_17) if c in UNSEEN_48_17]


@DATASETS.register_module()
class CocoZSLSeenDataset(ZSLDataset, CocoDataset):
    CLASSES, PALETTE = zip(*[
        (CocoDataset.CLASSES[i], CocoDataset.PALETTE[i]) 
        for i in SEEN_48_17
    ])
    PALETTE = list(PALETTE)


@DATASETS.register_module()
class CocoZSLUnseenDataset(ZSLDataset, CocoDataset):
    CLASSES, PALETTE = zip(*[
        (CocoDataset.CLASSES[i], CocoDataset.PALETTE[i]) 
        for i in UNSEEN_48_17
    ])
    PALETTE = list(PALETTE)


@DATASETS.register_module()
class CocoGZSLDataset(ZSLDataset, CocoDataset):
    CLASSES, PALETTE = zip(*[
        (CocoDataset.CLASSES[i], CocoDataset.PALETTE[i]) 
        for i in ALL_48_17
    ])
    PALETTE = list(PALETTE)

    def summarize(self, cocoEval: COCOeval, logger=None, split_name: Optional[str] = None) -> dict:
        redirect_string = io.StringIO()
        with contextlib.redirect_stdout(redirect_string):
            cocoEval.summarize()
        eval_results = {
            'bbox_' + metric: round(cocoEval.stats[i], 4) 
            for i, metric in enumerate([
                'mAP', 'mAP_50', 'mAP_75', 'mAP_s', 'mAP_m', 'mAP_l',
            ])
        }
        eval_results['bbox_mAP_copypaste'] = ' '.join(str(ap) for ap in eval_results.values())
        if split_name is not None:
            print_log(f'Evaluate split *{split_name}*', logger=logger)
            eval_results = {f'{split_name}_{k}': v for k, v in eval_results.items()}
        print_log('\n' + redirect_string.getvalue(), logger=logger)
        return eval_results

    def evaluate(
        self, results, metric='bbox', gpu_collect=False, logger=None,
        iou_thrs: Optional[Tuple[float]] = None,
        max_dets: Optional[Tuple[int]] = (100, 300, 1000),
    ) -> dict:
        predictions = self._det2json(results)

        cocoGt = self.coco
        try:
            cocoDt = self.coco.loadRes(predictions)
        except IndexError:
            print_log(
                'The testing results of the whole dataset is empty.',
                logger=logger, level=logging.ERROR,
            )
            return {}
        cocoEval = COCOeval(cocoGt, cocoDt, 'bbox')
        cocoEval.params.catIds = self.cat_ids
        cocoEval.params.imgIds = self.img_ids
        if iou_thrs is not None:
            cocoEval.params.iouThrs = np.array(iou_thrs)
        if max_dets is not None:
            cocoEval.params.maxDets = list(max_dets)

        cocoEval.evaluate()
        cocoEval.accumulate()

        eval_results = self.summarize(cocoEval, logger)

        precision: np.ndarray = cocoEval.eval['precision']  # Thresh x Recall x K x Area x MaxDets
        recall: np.ndarray = cocoEval.eval['recall']  # Thresh x K x Area x MaxDets
        assert len(self.cat_ids) == precision.shape[2] == recall.shape[1], f"{len(self.cat_ids)}, {precision.shape}, {recall.shape}"

        for split in ['SEEN_48_17', 'UNSEEN_48_17']:
            cocoEval.eval['precision'] = precision[:, :, eval('INDEX_' + split), :, :]
            cocoEval.eval['recall'] = recall[:, eval('INDEX_' + split), :, :]
            eval_results.update(self.summarize(cocoEval, logger, split_name=split))

        return eval_results


class FeatureExtractionDataset(ZSLDataset):
    def prepare_test_img(self, idx):
        return self.prepare_train_img(idx)


@DATASETS.register_module()
class CocoFeatureExtractionDataset(FeatureExtractionDataset, CocoDataset):
    pass


@DATASETS.register_module()
class LVISV1FeatureExtractionDataset(FeatureExtractionDataset, LVISV1Dataset):
    pass
