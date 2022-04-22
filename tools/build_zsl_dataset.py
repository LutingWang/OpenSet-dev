import argparse
import json
import os.path as osp
import sys; sys.path.insert(0, '')
from typing import Tuple

from mmcv import Config

from denseclip.datasets import CocoZSLSeenDataset, CocoGZSLDataset, LVISV1GZSLDataset, LVISV1ZSLSeenDataset
from denseclip.utils import odps_init


def parse_args():
    parser = argparse.ArgumentParser(description='Build COCO ZSL dataset')
    parser.add_argument('config', help='data config file path')
    # parser.add_argument('--split', type=str, nargs='+', choices=['65_15', '48_17'])
    args = parser.parse_args()
    return args


def split_dataset(ann_file: str, split: Tuple[str], name: str):
    with open(ann_file) as f:
        data = json.load(f)

    data['categories'] = [cat for cat in data['categories'] if cat['name'] in split]
    print("#categories:", len(data['categories']))

    class_name2id = {cat['name']: cat['id'] for cat in data['categories']}
    split: Tuple[int] = tuple(class_name2id[class_name] for class_name in split)
    data['annotations'] = [anno for anno in data['annotations'] if anno['category_id'] in split]
    print("#annotations:", len(data['annotations']))

    seen_image_ids = set([anno['image_id'] for anno in data['annotations']])
    data['deleted_images'] = [i for i, image in enumerate(data['images']) if image['id'] not in seen_image_ids]
    print("#deleted_images:", len(data['deleted_images']))

    data['images'] = [image for image in data['images'] if image['id'] in seen_image_ids]
    print("#images:", len(data['images']))

    ann_file_name, ext = osp.splitext(ann_file)
    seen_ann_file = f'{ann_file_name}_{name}_1{ext}'
    print("Saving to", seen_ann_file)
    with open(seen_ann_file, 'w') as f:
        json.dump(data, f)


def get_dataset_cfg(dataset_cfg: Config) -> Config:
    dataset_type = dataset_cfg.type
    if dataset_type == 'ClassBalancedDataset':
        dataset_cfg = dataset_cfg.dataset
    return dataset_cfg


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)

    print("Splitting training dataset")
    train_cfg = get_dataset_cfg(cfg.data.train)
    dataset_type = train_cfg.type
    if dataset_type == 'CocoDataset':
        split_dataset(train_cfg.ann_file, CocoZSLSeenDataset.CLASSES, '48_17')
    elif dataset_type == 'LVISV1Dataset':
        split_dataset(train_cfg.ann_file, LVISV1ZSLSeenDataset.CLASSES, 'seen')
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")
    print()
    print("Splitting validation dataset")
    val_cfg = get_dataset_cfg(cfg.data.val)
    dataset_type = val_cfg.type
    if dataset_type == 'CocoDataset':
        split_dataset(val_cfg.ann_file, CocoGZSLDataset.CLASSES, '48_17')
    elif dataset_type == 'LVISV1Dataset':
        # split_dataset(val_cfg.ann_file, LVISV1GZSLDataset.CLASSES, 'unseen')
        print("Passed for lvis v1 dataset.")
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")


if __name__ == '__main__':
    odps_init()
    main()