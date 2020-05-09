
import torch
from torch.utils.data import Dataset
from PIL import Image
from os import path
import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

import cv2
from albumentations import Compose, Normalize, VerticalFlip, HorizontalFlip, Rotate, Resize, LongestMaxSize, PadIfNeeded, BboxParams
from albumentations.pytorch import ToTensor

from Pytorch.utils import Normalize_bbox_to_0_1, Convert_bbox_from_TLWH_to_TLBR, clamp

'''
CODING CONVENTIONS

- All names use underscore casing
- L_* for list variable names
- D_* for dictionary variable names
- T_* for tensor variable names
- M_* for Pytorch models
- DL_* for Data Loader
- DSET_* for Dataset
- Class names are in upper case
- Function names start with a capital letter
'''

class ROI_DATASET(Dataset):
    def __init__(self,
                root_dirpath,
                images_dirpath,
                annotations_path,
                image_id_key="image_id",
                category_id_key="category_id",
                F_image_id_to_relative_path=None,
                augmentation=True,
                desired_size=280,
                batch_size=12,
                num_batch=1):

        self.root_dirpath = root_dirpath
        self.images_dirpath = images_dirpath
        self.annotations_path = annotations_path
        self.image_id_key=image_id_key
        self.category_id_key=category_id_key
        self.F_image_id_to_relative_path = F_image_id_to_relative_path
        self.augmentation = augmentation
        self.desired_size = desired_size
        self.batch_size = batch_size
        self.num_batch = num_batch

        with open(self.annotations_path) as f:
            self.D_instances = json.load(f)
            self.L_annotations = self.D_instances['annotations']
            self.L_categories = self.D_instances['categories']

        # list of real class indices in COCO dataset
        self.L_real_class_idx = [x['id'] for x in self.L_categories]
        # zero-based class indices of COCO dataset (0-79)
        self.L_class_idx = range(len(self.L_real_class_idx))
        # dict to map real class indices to zero-based indices
        self.D_real_class_idx_to_class_idx = { real_class_idx : class_idx for class_idx, real_class_idx in enumerate(self.L_real_class_idx)}

        # dict to map zero based indices to the relative annotations
        self.D_class_to_annotations = self.Create_class_to_annotations_dict()

        # filter zero-based class indices which have no annotations
        self.L_not_empty_class_idx = [class_idx for class_idx in self.L_class_idx if len(self.D_class_to_annotations[class_idx])>0]

        bbox_params = BboxParams(format='coco', label_fields=[self.category_id_key])
        augmentation_transform = Compose(
            [
                VerticalFlip(p=0.5),
                HorizontalFlip(p=0.5),
                LongestMaxSize(max_size=600),
                PadIfNeeded(min_height=710, min_width=710, border_mode=cv2.BORDER_CONSTANT),
                Rotate(p=1.0, limit=12, interpolation=cv2.INTER_LINEAR, border_mode=cv2.BORDER_CONSTANT),
                Resize(height=desired_size, width=desired_size),
                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensor()
            ],
            bbox_params=bbox_params)

        loading_transform = Compose(
            [
                LongestMaxSize(max_size=desired_size),
                PadIfNeeded(min_height=desired_size, min_width=desired_size, border_mode=cv2.BORDER_CONSTANT),
                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensor()
            ],
            bbox_params=bbox_params)

        self.transform = augmentation_transform if self.augmentation else loading_transform

    def Create_class_to_annotations_dict(self):
        '''
        Create a dictionary of type (int : list) which map to each zero-based class index a list of
        all the COCO annotations of a specific class using self.D_real_class_idx_to_class_idx as map
        '''
        # init empty dictionary with zero-based class indices
        D_class_to_annotations = {idx : [] for idx in self.L_class_idx}

        for annotation in self.L_annotations:
            real_class_idx = annotation[self.category_id_key]
            class_idx = self.D_real_class_idx_to_class_idx[real_class_idx]
            D_class_to_annotations[class_idx].append(annotation)

        return D_class_to_annotations

    def Get_num_classes(self):
        return len(self.L_categories)

    def Get_class_idx_description(self, class_idx):
        return self.L_categories[class_idx]

    def __getitem__(self, index):
        try:
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

            D_item_albumentation = {'image': np.array(item_PIL_image), 'bboxes': [item_bbox], self.category_id_key: [item_class_idx]}

            D_item_transformed = self.transform(**D_item_albumentation)
            if(len(D_item_transformed['bboxes'])==0):
                return self[0]

            # D_item_transformed['bboxes'] has got the only bbox we are passing in D_item_albumentation
            L_bbox_TLBR = Convert_bbox_from_TLWH_to_TLBR(list(D_item_transformed['bboxes'][0]))
            L_bbox_normalized = Normalize_bbox_to_0_1(L_bbox_TLBR, (self.desired_size, self.desired_size))
            T_bbox_normalized = torch.Tensor(L_bbox_normalized)

            return (D_item_transformed["image"], T_bbox_normalized), item_class_idx
        except:
            return self[0]

    def __len__(self):
        return self.num_batch*self.batch_size


'''
USAGE EXAMPLE:
DSET_benedettini_training = ROI_DATASET(
    root_dirpath = 'Datasets/Points Of Interest Recognition/Monastero dei Benedettini/',
    images_dirpath = 'Datasets/Points Of Interest Recognition/Monastero dei Benedettini/',
    annotations_path = 'Datasets/Points Of Interest Recognition/Monastero dei Benedettini/train_dataset.json',
    image_id_key="path",
    category_id_key="class_id",
    augmentation=True,
    batch_size=batch_size,
    num_batch=num_batch_train
)
'''
