import os
import os.path as osp
from typing import Tuple

import clip.model
import clip.simple_tokenizer
import einops
import lmdb
import todd
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.transforms.functional as tf
from mmcv.runner import TextLoggerHook
from mmdet.datasets import PIPELINES


class SimpleTokenizer(clip.simple_tokenizer.SimpleTokenizer):
    def __init__(self, *args, context_length: int, prompt_length: int, **kwargs):
        super().__init__(*args, **kwargs)
        self._context_length = context_length
        self._prompt_length = prompt_length
        self._cache = {}

    def tokenize(self, texts: Tuple[str], device: torch.device):
        texts_hash = len(texts)
        if texts_hash in self._cache:
            return self._cache[texts_hash]

        sot_token = self.encoder["<|startoftext|>"]
        eot_token = self.encoder["<|endoftext|>"]

        tokens = torch.zeros(len(texts), self._context_length, dtype=torch.long, device=device)

        for token, text in zip(tokens, texts):
            token_ = self.encode(text)
            token_ = [sot_token] * (1 + self._prompt_length) + token_ + [eot_token]
            assert len(token_) <= token.shape[0]
            token[:len(token_)] = token.new_tensor(token_)

        self._cache[texts_hash] = tokens
        return tokens


def odps_init():
    logger = todd.logger.get_logger()
    logger.debug("ODPS initializing.")

    def _dump_log(*args, **kwargs):
        return 

    TextLoggerHook._dump_log = _dump_log
    if not osp.exists('data'):
        os.symlink('/data/oss_bucket_0', 'data')
    # if not osp.exists('local_data'):
    #     os.mkdir('local_data')
    #     shutil.copytree('data/coco/embeddings6.lmdb', 'local_data/embeddings6.lmdb')
    if not osp.exists('pretrained'):
        os.symlink('/data/oss_bucket_0/ckpts', 'pretrained')
    if not osp.exists('work_dirs'):
        os.symlink('/data/oss_bucket_0/work_dirs', 'work_dirs')

    logger.debug(f"ODPS initialization done with {os.listdir('.')}.")


@torch.no_grad()
def encode_bboxes(
    model: clip.model.CLIP, 
    preprocess: transforms.Compose,
    image: torch.Tensor,
    bboxes: torch.Tensor,
) -> torch.Tensor:
    if bboxes.shape[0] == 0:
        return bboxes.new_zeros((0, 1024))
    bboxes15 = todd.utils.expand_bboxes(bboxes, ratio=1.5, image_shape=image.shape[-2:])
    bboxes = torch.cat([bboxes, bboxes15])
    bboxes = bboxes[:, [1, 0, 3, 2]]  # tf.crop requires y, x, h, w
    bboxes = bboxes.int().tolist()
    crops = [tf.crop(image, *bbox) for bbox in bboxes]
    crops = [preprocess(crop) for crop in crops]
    crops = torch.stack(crops)
    crops = model.encode_image(crops)
    crops = einops.reduce(crops, '(n b) d -> b d', n=2, reduction='sum')
    crops = F.normalize(crops, dim=-1)
    return crops


def has_debug_flag(level: int) -> bool:
    """Parse debug flags.

    Levels:
        0: custom flag
        1: use val dataset as train dataset
        2: use smaller datasets
        3: use fewer gt bboxes
        4: cpu

    Note:
        Flag 1/2/3 are set by default when cuda is unavailable.
    """
    if not torch.cuda.is_available() and level in [1, 2, 3, 4]:
        return True
    flags = os.environ.get('DEBUG', '')
    flags += '0' * 10
    return bool(int(flags[level]))


@PIPELINES.register_module()
class LoadEmbeddings:
    def __init__(self, lmdb_filepath: str):
        self._lmdb_filepath = lmdb_filepath
        self._env: lmdb.Environment = lmdb.open
