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

import pickle
import torch
import plotly.graph_objects as go
from todd.utils import BBoxes
areas = []
with open('data/lvis_v1/proposals/rpn_r101_fpn_lvis_v1_val.pkl', 'rb') as f:
    for proposal in pickle.load(f)[:200]:
        proposal = torch.Tensor(proposal[:, :4])
        proposal = BBoxes(proposal)
        areas.append(proposal.areas)
areas = torch.cat(areas)
areas = areas[areas > 1]
areas = areas[areas < 64 * 64]
# area_hist = torch.histc(areas, bins=100)
# area_range = torch.linspace(areas.min(), areas.max(), 100)
# print(area_hist, area_range)
fig = go.Figure(data=[go.Histogram(x=areas, nbinsx=1000)])
fig.show()