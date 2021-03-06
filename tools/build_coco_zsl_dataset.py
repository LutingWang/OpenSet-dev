import argparse
import json
import os.path as osp
import sys; sys.path.insert(0, '')
from typing import Tuple

from mmcv import Config

from denseclip.utils import odps_init
from denseclip.coco import CocoZSLSeenDataset, CocoZSLUnseenDataset, CocoGZSLDataset


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
    data['images'] = [image for image in data['images'] if image['id'] in seen_image_ids]
    print("#images:", len(data['images']))

    ann_file_name, ext = osp.splitext(ann_file)
    seen_ann_file = f'{ann_file_name}_{name}{ext}'
    print("Saving to", seen_ann_file)
    with open(seen_ann_file, 'w') as f:
        json.dump(data, f)


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)

    print("Splitting training dataset")
    split_dataset(cfg.data.train.ann_file, CocoZSLSeenDataset.CLASSES, '48_17')
    print()
    print("Splitting validation dataset")
    split_dataset(cfg.data.val.ann_file, CocoGZSLDataset.CLASSES, '48_17')


if __name__ == '__main__':
    odps_init()
    main()