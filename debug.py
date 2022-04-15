from typing import List
from PIL import Image
import clip
import torch
import torch.nn.functional as F


def clip_image_forward(clip_model, preprocess, img_meta,bboxes):
    scale_factor = img_meta['scale_factor']
    img = Image.open(img_meta['filename'])
    if img_meta['flip']:
        w = img_meta['img_shape'][1]
        flipped = bboxes.clone()
        flipped[..., 1::4] = w - bboxes[..., 3::4]
        flipped[..., 3::4] = w - bboxes[..., 1::4]
        bboxes = flipped
    bbox_raw = bboxes[:,1:]
    bbox_raw /= bbox_raw.new_tensor(scale_factor)
    img_shape = img.size
    bboxes = torch.dstack([torch.floor(bbox_raw[:,0]-0.001),torch.floor(bbox_raw[:,1]-0.001),torch.ceil(bbox_raw[:,2]+0.001),torch.ceil(bbox_raw[:,3]+0.001)]).squeeze(0)
    bboxes[:,[0,2]].clamp_(min=0,max=img_shape[0])
    bboxes[:,[1,3]].clamp_(min=0,max=img_shape[1])
    bboxes = bboxes.detach().cpu().numpy()

    cropped_images = []
    for box in bboxes:
        cropped_image = img.crop(box)
        cropped_image = preprocess(cropped_image)
        cropped_images.append(cropped_image)
    cropped_images = torch.stack(cropped_images)
    image_features = clip_model.encode_image(cropped_images)
    return image_features

def boxto15(bboxes):
    bboxes15 = torch.dstack([
                bboxes[:,0],
                1.25 * bboxes[:, 1] - 0.25 * bboxes[:, 3], 
                1.25 * bboxes[:, 2] - 0.25 * bboxes[:, 4],
                1.25 * bboxes[:, 3] - 0.25 * bboxes[:, 1], 
                1.25 * bboxes[:, 4] - 0.25 * bboxes[:, 2]
                ]).squeeze(0)
    return bboxes15


def main(bboxes_all: List[torch.Tensor], img_metas: List[dict]):
    clip_model, preprocess = clip.load('pretrained/clip/RN50.pt')
    clip_image_features_ensemble = []
    for i in range(len(img_metas)):
        bboxes_single_image = bboxes_all[i]
        bboxes15 = boxto15(bboxes_single_image)

        clip_image_features = clip_image_forward(clip_model, preprocess, img_metas[i], bboxes_single_image)
        clip_image_features15 = clip_image_forward(clip_model, preprocess, img_metas[i], bboxes15)

        clip_image_features_single = clip_image_features + clip_image_features15
        clip_image_features_single = clip_image_features_single.float()
        clip_image_features_single = F.normalize(clip_image_features_single, p=2, dim=1)

        clip_image_features_ensemble.append(clip_image_features_single)
    clip_image_features_ensemble = torch.cat(clip_image_features_ensemble, dim=0)
    return clip_image_features_ensemble