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

import os
os.symlink('/data/oss_bucket_0', 'data')
print(len(os.listdir('data/lvis_v1/data/lvis_clip_image_embedding/train2017/')))
