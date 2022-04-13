import argparse
import os
import sys; sys.path.insert(0, '')
from typing import List

import clip
import clip.model
import todd
import torch
import torch.nn as nn
import torch.nn.functional as F
from easydict import EasyDict
from mmdet.datasets import DATASETS, CustomDataset
from tqdm import tqdm

from denseclip.utils import odps_init


def vild(args):
    model: nn.Module = torch.jit.load(args.pretrained, map_location='cpu')
    state_dict = model.state_dict()
    state_dict['positional_embedding'] = state_dict['positional_embedding'][:13]
    model = clip.model.build_model(state_dict).float()
    delattr(model, 'visual')
    model.visual = EasyDict(
        conv1=model.token_embedding,  # hack self.teacher.dtype
    )
    todd.utils.freeze_model(model)

    tmp = {}
    with open(args.templates) as f:
        exec(f.read(), globals(), tmp)
    templates: List[str] = tmp['templates']

    dataset: CustomDataset = DATASETS.get(args.dataset)
    classes = dataset.CLASSES
    save_path = os.path.join(os.path.dirname(args.templates), f'vild_{args.dataset}.pth')

    class_embeddings = []
    for template in tqdm(templates):
        texts = clip.tokenize([template.format(c) for c in classes], context_length=13)
        with torch.no_grad():
            embeddings = model.encode_text(texts)
            embeddings = F.normalize(embeddings)
        class_embeddings.append(embeddings)
    class_embeddings = torch.stack(class_embeddings).mean(dim=0)
    torch.save(class_embeddings, save_path)


def parse_args():
    parser = argparse.ArgumentParser(description="ViLD class embeddings")
    subparsers = parser.add_subparsers(help="Method")
    vild_parser = subparsers.add_parser('vild')
    vild_parser.add_argument('--pretrained', default='pretrained/RN50.pt')
    vild_parser.add_argument('--templates', default='data/coco/prompt/vild.py')
    vild_parser.add_argument('--dataset', default='CocoDataset', help="Used to get classes.")
    vild_parser.set_defaults(func=vild)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    args.func(args)


if __name__ == '__main__':
    odps_init()
    main()
