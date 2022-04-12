import os
from pathlib import Path
os.symlink('/data/oss_bucket_0', 'data')
print(len(os.listdir('data/coco/proposal_embeddings7.pth/train')))
print([path.stem for path in Path('data/coco/proposal_embeddings7.pth/train').glob('*.pth')][:200])