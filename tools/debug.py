import io
import os
import shutil
import torch
import lmdb


def tmp(filepath):
    a = lmdb.open(filepath, readonly=True, max_dbs=2)
    b = a.open_db('train'.encode())
    c = a.begin(b)
    d = c.cursor()
    e = dict(d)
    f = e[b'55244']
    g = io.BytesIO(f)
    h = torch.load(g, map_location='cpu')
    return e.keys()

os.symlink('/data/oss_bucket_0', 'data')
os.mkdir('local_data')
shutil.copytree('data/coco/embeddings6.lmdb', 'local_data/embeddings6.lmdb')
print(os.listdir('local_data'))
print(os.listdir('local_data/embeddings6.lmdb'))
k1 = tmp('local_data/embeddings6.lmdb')
# k2 = tmp('data/coco/embeddings5.lmdb/worker0')
# k3 = tmp('data/coco/embeddings5.lmdb/worker1')
print(k1)
...