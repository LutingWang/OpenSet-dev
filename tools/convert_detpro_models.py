from typing import Dict
import einops

import torch


ckpt: Dict[str, torch.Tensor] = torch.load('epoch_20.pth', map_location='cpu')['state_dict']
ckpt = {k: v.float() for k, v in ckpt.items() if 'clip_model' not in k and 'mask' not in k}
ckpt['roi_head.bbox_head._bg_class_embedding'] = (
    einops.rearrange(ckpt.pop('roi_head.bg_embedding.weight'), 'b 1 -> 1 b') + 
    einops.rearrange(ckpt.pop('roi_head.bg_embedding.bias'), 'b -> 1 b')
)
change = {
    'roi_head.bbox_head.shared_convs_for_image.': 'roi_head._ensemble_head.shared_convs.',
    'roi_head.bbox_head.shared_fcs_for_image.': 'roi_head._ensemble_head.shared_fcs.',
    'roi_head.projection.': 'roi_head.bbox_head.fc_cls.0.',
    'roi_head.projection_for_image.': 'roi_head._ensemble_head.fc_cls.0.',
}
for old_key, new_key in change.items():
    ckpt = {
        k.replace(old_key, new_key) if k.startswith(old_key) else k: v 
        for k, v in ckpt.items()
    }
torch.save(ckpt, 'detpro.pth')