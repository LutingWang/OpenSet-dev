import os
import os.path as osp
from typing import Tuple

import clip.simple_tokenizer
import torch
from mmcv.runner import TextLoggerHook


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

    def _dump_log(*args, **kwargs):
        return 

    TextLoggerHook._dump_log = _dump_log
    if not osp.exists('data'):
        os.symlink('/data/oss_bucket_0', 'data')
    if not osp.exists('pretrained'):
        os.symlink('/data/oss_bucket_0/ckpts', 'pretrained')
    if not osp.exists('work_dirs'):
        os.symlink('/data/oss_bucket_0/work_dirs', 'work_dirs')
