# import os
# import shutil
# import zipfile
# print('extract')
# os.symlink('/data/oss_bucket_0', 'data')
# filename = 'data/lvis_v1/proposal_embeddings.zip'
# if not zipfile.is_zipfile(filename):
#     raise RuntimeError(f"{filename} is not a zip file.")
# f = zipfile.ZipFile(filename)
# f.extractall('data/lvis_v1')

# import os
# import shutil
# os.symlink('/data/oss_bucket_0', 'data')
# print(len(os.listdir('data/lvis_v1/data/lvis_clip_image_embedding/train2017/')))
# print(len(os.listdir('data/lvis_v1/proposal_embeddings10/train')))
# shutil.copytree('data/lvis_v1/data/lvis_clip_image_embedding/train2017/', 'data/lvis_v1/proposal_embeddings10/train')

# import pickle
# import torch
# import plotly.graph_objects as go
# from todd.utils import BBoxes
# areas = []
# with open('data/lvis_v1/proposals/rpn_r101_fpn_lvis_v1_val.pkl', 'rb') as f:
#     for proposal in pickle.load(f)[:200]:
#         proposal = torch.Tensor(proposal[:, :4])
#         proposal = BBoxes(proposal)
#         areas.append(proposal.areas)
# areas = torch.cat(areas)
# areas = areas[areas > 1]
# areas = areas[areas < 64 * 64]
# # area_hist = torch.histc(areas, bins=100)
# # area_range = torch.linspace(areas.min(), areas.max(), 100)
# # print(area_hist, area_range)
# fig = go.Figure(data=[go.Histogram(x=areas, nbinsx=1000)])
# fig.show()

import torch
# from mmdet.datasets import CocoDataset
from mmdet.datasets import LVISV1Dataset as CocoDataset
# from denseclip.datasets import COCO_SEEN_48_17, COCO_UNSEEN_48_17
from denseclip.datasets import LVIS_V1_SEEN_866_337 as COCO_SEEN_48_17
from denseclip.datasets import LVIS_V1_UNSEEN_866_337 as COCO_UNSEEN_48_17
seen_classes = [CocoDataset.CLASSES[i] for i in COCO_SEEN_48_17]
unseen_classes = [CocoDataset.CLASSES[i] for i in COCO_UNSEEN_48_17]
# class_embeddings = torch.load('data/coco/prompt/vild_RN50.pt')
class_embeddings = torch.load('data/lvis_v1/prompt/detpro_vild_ViT-B-32.pt', 'cpu')
seen_class_embeddings = class_embeddings[COCO_SEEN_48_17]
unseen_class_embeddings = class_embeddings[COCO_UNSEEN_48_17]
similarity = torch.einsum('u c, s c -> u s', unseen_class_embeddings, seen_class_embeddings).float()
print(similarity.min(), similarity.mean(), similarity.max())
# values, indices = torch.topk(1 - similarity.abs(), k=5, dim=1)
# values = 1 - values
# for i, (value, index) in enumerate(zip(values, indices)):
#     print(unseen_classes[i], {seen_classes[i]: v for i, v in zip(index, value)})
