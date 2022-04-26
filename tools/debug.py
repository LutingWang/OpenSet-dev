import os
import shutil
import zipfile
os.symlink('/data/oss_bucket_0', 'data')
filename = 'lvis_clip_image_embedding_1.zip'
shutil.copyfile('data/lvis_clip_image_embedding.zip', filename)
if not zipfile.is_zipfile(filename):
    raise RuntimeError(f"{filename} is not a zip file.")
with zipfile.ZipFile(filename, 'r') as f:
    f.extractall('.')
