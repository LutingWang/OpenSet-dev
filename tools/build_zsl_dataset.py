import argparse
import json
import os.path as osp
import sys; sys.path.insert(0, '')
from collections import namedtuple
from pathlib import Path
from typing import Dict, Set, Tuple, Type

from mmcv import Config, ConfigDict
from mmdet.datasets import CustomDataset
from todd.logger import get_logger

from denseclip import datasets
from denseclip.utils import odps_init


Datasets: Type[Tuple[CustomDataset]] = namedtuple('Datasets', ['train', 'val'])
metas: Dict[str, Dict[str, Datasets]] = dict(
    CocoDataset={
        '48_17': Datasets(
            train=datasets.CocoZSLSeenDataset, 
            val=datasets.CocoGZSLDataset,
        ),
    },
    LVISV1Dataset={
        '866_337': Datasets(
            train=datasets.LVISV1ZSLSeenDataset, 
            val=datasets.LVISV1GZSLDataset,
        ),
    },
)


def parse_args():
    parser = argparse.ArgumentParser(description='Build COCO ZSL dataset')
    parser.add_argument('config', help='data config file path')
    parser.add_argument('--split')
    args = parser.parse_args()
    return args


def split_dataset(cfg: ConfigDict, split: str, train: bool):
    logger = get_logger()

    while 'dataset' in cfg:
        cfg = cfg.dataset
    if cfg.type not in metas:
        raise ValueError(f"Unknown dataset type: {cfg.type}")
    logger.info(f"Dataset type: {cfg.type}")

    if split not in metas[cfg.type]:
        raise ValueError(f"Unknown split: {split}.")
    logger.info(f"Split: {split}")

    save_file = Path(f'_{split}_4'.join(osp.splitext(cfg.ann_file)))
    if save_file.exists():
        raise ValueError(f"{save_file} already exists.")
    logger.info(f"Saving to {save_file}.")

    dataset = metas[cfg.type][split][not train]
    expected_cats: Set[str] = set(dataset.CLASSES)
    logger.info(f"Expected categories: {len(expected_cats)}.")

    with open(cfg.ann_file) as f:
        data = json.load(f)
    logger.info(f"Loaded {len(data['categories'])} categories.")

    invalid_cats: Set[str] = expected_cats - {cat['name'] for cat in data['categories']}
    if invalid_cats:
        raise ValueError(f"Invalid categories: {invalid_cats}.")
    if len(expected_cats) == len(data['categories']):
        logger.info("No need to split.")
        return

    cat_ids: Set[int] = set()
    for cat in data['categories']:
        if cat['name'] in expected_cats:
            cat_ids.add(cat['id'])
        else:
            cat['name'] = None
    data['annotations'] = [ann for ann in data['annotations'] if ann['category_id'] in cat_ids]
    logger.info(f"Split {len(data['annotations'])} annotations.")

    with save_file.open('w') as f:
        json.dump(data, f)


def main():
    args = parse_args()
    odps_init()

    logger = get_logger()
    cfg = Config.fromfile(args.config)
    logger.info("Split training dataset.")
    split_dataset(cfg.data.train, args.split, True)
    logger.info("Split validation dataset.")
    split_dataset(cfg.data.val, args.split, False)


if __name__ == '__main__':
    main()
