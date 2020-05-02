import torch
import numpy as np
from PIL import Image
from os import path
from ROIDataset import ROI_DATASET

from Pytorch.utils import Normalize_bbox_to_0_1, Convert_bbox_from_TLWH_to_TLBR, clamp

class PATCH_DATASET(ROI_DATASET):

  def __getitem__(self, index):

    # sample a random annotation from a random class
    item_class_idx = np.random.choice(self.L_not_empty_class_idx)
    D_item_annotation = np.random.choice(self.D_class_to_annotations[item_class_idx])

    item_image_id = D_item_annotation[self.image_id_key]

    if(self.F_image_id_to_relative_path):
        image_relative_path = self.F_image_id_to_relative_path(item_image_id)
    else:
        image_relative_path = item_image_id
    item_PIL_image = Image.open(path.join(self.images_dirpath, image_relative_path)).convert('RGB')
    image_width, image_height = item_PIL_image.size

    item_bbox = D_item_annotation['bbox']
    item_bbox[0] = clamp(item_bbox[0], 0, image_width)
    item_bbox[1] = clamp(item_bbox[1], 0, image_height)
    item_bbox[2] = clamp(item_bbox[2], 0, image_width-item_bbox[0])
    item_bbox[3] = clamp(item_bbox[3], 0, image_height-item_bbox[1])

    item_PIL_image_patch = item_PIL_image.crop((item_bbox[0], item_bbox[1], item_bbox[0]+item_bbox[2], item_bbox[1]+item_bbox[3]))

    patch_image_width, patch_image_height = item_PIL_image_patch.size
    item_bbox[0] = 0
    item_bbox[1] = 0
    item_bbox[2] = patch_image_width
    item_bbox[3] = patch_image_height

    D_item_albumentation = {'image': np.array(item_PIL_image_patch), 'bboxes': [item_bbox], self.category_id_key: [item_class_idx]}

    D_item_transformed = self.transform(**D_item_albumentation)
    if(len(D_item_transformed['bboxes'])==0):
        return self[0]

    # D_item_transformed['bboxes'] has got the only bbox we are passing in D_item_albumentation
    L_bbox_TLBR = Convert_bbox_from_TLWH_to_TLBR(list(D_item_transformed['bboxes'][0]))
    L_bbox_normalized = Normalize_bbox_to_0_1(L_bbox_TLBR, (self.desired_size, self.desired_size))
    T_bbox_normalized = torch.Tensor(L_bbox_normalized)

    return (D_item_transformed["image"], T_bbox_normalized), item_class_idx