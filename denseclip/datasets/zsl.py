from typing import Any, List

import torch

from ..utils import has_debug_flag


class ZSLDataset:
    def __len__(self):
        if has_debug_flag(2):
            return 4
        return super().__len__()

    def load_annotations(self, *args, **kwargs):
        data_infos = super().load_annotations(*args, **kwargs)
        if has_debug_flag(2):
            # if not self.test_mode:
            #     self.coco.dataset['images'] = [self.coco.dataset['images'][i] for i in [77076, 32658, 48276, 94712]]
            #     data_infos = [data_infos[i] for i in [77076, 32658, 48276, 94712]]

            self.coco.dataset['images'] = self.coco.dataset['images'][:len(self)]
            self.img_ids = [img['id'] for img in self.coco.dataset['images']]
            self.coco.dataset['annotations'] = [
                ann for ann in self.coco.dataset['annotations']
                if ann['image_id'] in self.img_ids
            ]
            self.coco.imgs = {img['id']: img for img in self.coco.dataset['images']}
            data_infos = data_infos[:len(self)]
        return data_infos

    def load_proposals(self, proposal_file: str) -> List[torch.Tensor]:
        proposals = super().load_proposals(proposal_file)
        if has_debug_flag(2):
            # if not self.test_mode:
            #     proposals = [proposals[i] for i in [77076, 32658, 48276, 94712]]
            proposals = proposals[:len(self)]
        assert len(proposals) == len(self.data_infos)  # len(self) is set manually when debugging
        return proposals

    # def __getitem__(self, *args, **kwargs) -> Any:
    #     return super().__getitem__(*args, **kwargs)

    def evaluate(self, *args, **kwargs) -> Any:
        kwargs.pop('gpu_collect', None)
        kwargs.pop('tmpdir', None)
        # with todd.utils.setattr_temp(self, 'img_ids', self.img_ids[:len(self)]):
        return super().evaluate(*args, **kwargs)
