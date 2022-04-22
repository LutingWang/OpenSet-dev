from typing import Any, Dict, List

import todd
import torch
from mmdet.datasets import DATASETS

from ..utils import has_debug_flag


@DATASETS.register_module()
class ZSLDataset:
    def __len__(self):
        if has_debug_flag(2):
            return 2
        return super().__len__()

    def load_proposals(self, proposal_file: str) -> List[torch.Tensor]:
        logger = todd.logger.get_logger()
        logger.info(f'Loading proposals from: {proposal_file}.')
        proposals = super().load_proposals(proposal_file)
        logger.info(
            f'Loaded {sum(len(proposal) for proposal in proposals)} proposals '
            f'for {len(proposals)} images.'
        )
        proposals = [p[:, :4] for p in proposals]
        if 'deleted_images' in self.coco.dataset:
            deleted_images = self.coco.dataset['deleted_images']
            proposals = [p for i, p in enumerate(proposals) if i not in deleted_images]
        assert len(proposals) == len(self.data_infos)  # len(self) is set manually when debugging
        return proposals

    # def pre_pipeline(self, results: Dict[str, Any]):
    #     super().pre_pipeline(results)
    #     if self.proposals is not None:
    #         results['bbox_fields'].append('proposals')

    def evaluate(self, *args, gpu_collect: bool = False, **kwargs) -> Any:
        with todd.utils.setattr_temp(self, 'img_ids', self.img_ids[:len(self)]):
            return super().evaluate(*args, **kwargs)
