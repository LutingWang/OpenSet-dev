import argparse
import os
import sys; sys.path.insert(0, '')
from typing import List, Tuple

import clip
import clip.clip
import clip.model
import todd
import torch
import torch.nn as nn
import torch.nn.functional as F
from easydict import EasyDict
from mmdet.datasets import DATASETS, CustomDataset
from tqdm import tqdm

from denseclip.utils import odps_init


vild_templates = [
    "This is a {}",
    "There is a {}",
    "a photo of a {} in the scene",
    "a photo of a small {} in the scene",
    "a photo of a medium {} in the scene",
    "a photo of a large {} in the scene",
    "a photo of a {}",
    "a photo of a small {}",
    "a photo of a medium {}",
    "a photo of a large {}",
    "This is a photo of a {}",
    "This is a photo of a small {}",
    "This is a photo of a medium {}",
    "This is a photo of a large {}",
    "There is a {} in the scene",
    "There is the {} in the scene",
    "There is one {} in the scene",
    "This is a {} in the scene",
    "This is the {} in the scene",
    "This is one {} in the scene",
    "This is one small {} in the scene",
    "This is one medium {} in the scene",
    "This is one large {} in the scene",
    "There is a small {} in the scene",
    "There is a medium {} in the scene",
    "There is a large {} in the scene",
    "There is a {} in the photo",
    "There is the {} in the photo",
    "There is one {} in the photo",
    "There is a small {} in the photo",
    "There is the small {} in the photo",
    "There is one small {} in the photo",
    "There is a medium {} in the photo",
    "There is the medium {} in the photo",
    "There is one medium {} in the photo",
    "There is a large {} in the photo",
    "There is the large {} in the photo",
    "There is one large {} in the photo",
    "There is a {} in the picture",
    "There is the {} in the picture",
    "There is one {} in the picture",
    "There is a small {} in the picture",
    "There is the small {} in the picture",
    "There is one small {} in the picture",
    "There is a medium {} in the picture",
    "There is the medium {} in the picture",
    "There is one medium {} in the picture",
    "There is a large {} in the picture",
    "There is the large {} in the picture",
    "There is one large {} in the picture",
    "This is a {} in the photo",
    "This is the {} in the photo",
    "This is one {} in the photo",
    "This is a small {} in the photo",
    "This is the small {} in the photo",
    "This is one small {} in the photo",
    "This is a medium {} in the photo",
    "This is the medium {} in the photo",
    "This is one medium {} in the photo",
    "This is a large {} in the photo",
    "This is the large {} in the photo",
    "This is one large {} in the photo",
    "This is a {} in the picture",
    "This is the {} in the picture",
    "This is one {} in the picture",
    "This is a small {} in the picture",
    "This is the small {} in the picture",
    "This is one small {} in the picture",
    "This is a medium {} in the picture",
    "This is the medium {} in the picture",
    "This is one medium {} in the picture",
    "This is a large {} in the picture",
    "This is the large {} in the picture",
    "This is one large {} in the picture",
]


def get_dataset_class(dataset: str) -> CustomDataset:
    dataset_class_mapping = {
        'coco': 'CocoDataset',
        'lvis_v1': 'LVISV1Dataset',
    }
    if dataset not in dataset_class_mapping:
        raise ValueError(f'Unknown dataset: {dataset}')
    dataset_class = dataset_class_mapping[dataset]
    return DATASETS.get(dataset_class)


def get_pretrained_filepath(pretrained: str) -> Tuple[str, str]:
    pretrained = os.path.basename(clip.clip._MODELS[pretrained])
    pretrained_filepath = os.path.join('pretrained/clip', pretrained)
    return pretrained, pretrained_filepath


def get_output_filepath(dataset: str, pretrained: str) -> str:
    output_dir = os.path.join('data', dataset, 'prompt')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_filepath = os.path.join(output_dir, f'vild_{pretrained}')
    return output_filepath


def vild(args):
    dataset = get_dataset_class(args.dataset)
    pretrained, pretrained_filepath = get_pretrained_filepath(args.pretrained)
    output_filepath = get_output_filepath(args.dataset, pretrained)

    context_length = clip.tokenize(dataset.CLASSES).argmax(-1).max() + 9

    model: nn.Module = torch.jit.load(pretrained_filepath, map_location='cpu')
    state_dict = model.state_dict()
    state_dict['positional_embedding'] = state_dict['positional_embedding'][:context_length]
    model = clip.model.build_model(state_dict).float()
    delattr(model, 'visual')
    model.visual = EasyDict(
        conv1=model.token_embedding,  # hack self.teacher.dtype
    )
    todd.utils.freeze_model(model)

    class_embeddings = []
    for template in tqdm(vild_templates):
        texts = clip.tokenize([template.format(c) for c in dataset.CLASSES], context_length=context_length)
        with torch.no_grad():
            embeddings = model.encode_text(texts)
            embeddings = F.normalize(embeddings)
        class_embeddings.append(embeddings)
    class_embeddings = torch.stack(class_embeddings).mean(dim=0)
    class_embeddings = F.normalize(class_embeddings)
    torch.save(class_embeddings, output_filepath)


def parse_args():
    parser = argparse.ArgumentParser(description="ViLD class embeddings")
    subparsers = parser.add_subparsers(help="Method")
    vild_parser = subparsers.add_parser('vild')
    vild_parser.add_argument('--dataset', default='coco')
    vild_parser.add_argument('--pretrained', default='RN50')
    vild_parser.set_defaults(func=vild)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    args.func(args)


if __name__ == '__main__':
    odps_init()
    main()
