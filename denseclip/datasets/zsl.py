from typing import Any, List

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
        proposals = super().load_proposals(proposal_file)
        assert len(proposals) == len(self.data_infos)  # len(self) is set manually when debugging
        return proposals

    def evaluate(self, *args, gpu_collect: bool = False, **kwargs) -> Any:
        with todd.utils.setattr_temp(self, 'img_ids', self.img_ids[:len(self)]):
            return super().evaluate(*args, **kwargs)
