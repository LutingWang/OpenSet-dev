import os
import os.path as osp
from typing import Tuple

import clip.model
import clip.simple_tokenizer
import einops
import todd
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.transforms.functional as tf
from PIL.Image import Image as PILImage
from mmcv import Config
from mmcv.runner import TextLoggerHook
from mmcv.cnn import NORM_LAYERS


# class SimpleTokenizer(clip.simple_tokenizer.SimpleTokenizer):
#     def __init__(self, *args, context_length: int, prompt_length: int, **kwargs):
#         super().__init__(*args, **kwargs)
#         self._context_length = context_length
#         self._prompt_length = prompt_length
#         self._cache = {}

#     def tokenize(self, texts: Tuple[str], device: torch.device):
#         texts_hash = len(texts)
#         if texts_hash in self._cache:
#             return self._cache[texts_hash]

#         sot_token = self.encoder["<|startoftext|>"]
#         eot_token = self.encoder["<|endoftext|>"]

#         tokens = torch.zeros(len(texts), self._context_length, dtype=torch.long, device=device)

#         for token, text in zip(tokens, texts):
#             token_ = self.encode(text)
#             token_ = [sot_token] * (1 + self._prompt_length) + token_ + [eot_token]
#             assert len(token_) <= token.shape[0]
#             token[:len(token_)] = token.new_tensor(token_)

#         self._cache[texts_hash] = tokens
#         return tokens


def odps_init():
    logger = todd.logger.get_logger()
    logger.debug("ODPS initializing.")

    def _dump_log(*args, **kwargs):
        return 

    TextLoggerHook._dump_log = _dump_log
    if not osp.lexists('data'):
        os.symlink('/data/oss_bucket_0', 'data')
    if not osp.lexists('pretrained'):
        os.symlink('/data/oss_bucket_0/ckpts', 'pretrained')
    if not osp.lexists('work_dirs'):
        os.symlink('/data/oss_bucket_0/work_dirs', 'work_dirs')

    logger.debug(f"ODPS initialization done with {os.listdir('.')}.")


# @torch.no_grad()
# def encode_bboxes(
#     model: clip.model.CLIP, 
#     preprocess: transforms.Compose,
#     image: torch.Tensor,
#     bboxes: torch.Tensor,
# ) -> torch.Tensor:
#     bboxes_ = todd.utils.BBoxes(bboxes)
#     if bboxes_.empty:
#         return bboxes_.to_tensor().new_zeros((0, 1024))
#     bboxes15_ = bboxes_.expand(ratio=1.5, image_shape=image.shape[-2:])
#     bboxes_ = bboxes_ + bboxes15_
#     bboxes = bboxes_.round().to_tensor()
#     bboxes = bboxes.int().tolist()

#     pil_image = image[[2, 1, 0]].type(torch.uint8)  # BGR to RGB
#     pil_image: PILImage = tf.to_pil_image(pil_image)
#     crops = [pil_image.crop(bbox) for bbox in bboxes]
#     crops = [preprocess(crop) for crop in crops]
#     crops = torch.stack(crops)
#     crops = model.encode_image(crops)
#     crops = einops.reduce(crops, '(n b) d -> b d', n=2, reduction='sum')
#     crops = F.normalize(crops, dim=-1)
#     return crops


def debug_init(debug: bool, cfg: Config):
    if torch.cuda.is_available() and not debug:
        return
    if 'DEBUG' not in os.environ:
        os.environ['DEBUG'] = '001' if torch.cuda.is_available() else '011011'
    os.environ['DEBUG'] += '0' * 10
    if has_debug_flag(1) and 'data' in cfg and 'train' in cfg.data:
        data_train = cfg.data.train
        if 'dataset' in data_train:
            data_train = data_train.dataset
        if 'ann_file' in cfg.data.val:
            data_train.ann_file = cfg.data.val.ann_file
        if 'img_prefix' in cfg.data.val:
            data_train.img_prefix = cfg.data.val.img_prefix
        if 'proposal_file' in cfg.data.val:
            data_train.proposal_file = cfg.data.val.proposal_file
    if has_debug_flag(4):
        NORM_LAYERS.register_module('SyncBN', force=True, module=NORM_LAYERS.get('BN'))
    if has_debug_flag(5):
        cfg.data.samples_per_gpu = 2


def has_debug_flag(level: int) -> bool:
    """Parse debug flags.

    Levels:
        0: custom flag
        1: use val dataset as train dataset
        2: use smaller datasets
        3: use fewer gt bboxes
        4: cpu
        5: use batch size 2

    Note:
        Flag 1/2/3 are set by default when cuda is unavailable.
    """
    if 'DEBUG' not in os.environ:
        return False
    return bool(int(os.environ['DEBUG'][level]))
