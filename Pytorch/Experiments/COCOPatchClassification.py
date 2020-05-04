
import os
from os import path
import torch
from torch.utils.data import DataLoader
from Pytorch.Datasets.PatchDataset import PATCH_DATASET
from Pytorch.Models.ROIClassifier import M_ROI_CLASSIFIER

from torchtrainer.trainer import Trainer, Mode
from torchtrainer.callbacks.calculateaccuracycallback import CalculateTopNAccuracyCallback
from torchtrainer.callbacks.calculatelosscallback import CalculateLossCallback
from torchtrainer.callbacks.plotcallback import PlotCallback
from torchtrainer.callbacks.saveparameterscallback import SaveParametersCallback
from torchtrainer.callbacks.settqdmbardescription import SetTQDMBarDescription
from torchtrainer.callbacks.lrbatchschedulercallback import LRBatchSchedulerCallBack

def main():
    exp_name = "RN50-SGD-CyclicLR-Patch-COCO-DA"
    epochs = 60
    batch_size = 10
    num_batch_train = 200
    num_batch_val = 80
    parameters_path = path.join("parameters", exp_name)
    plots_path = path.join("plots", exp_name)

    DSET_coco_training = PATCH_DATASET(
        root_dirpath = 'Datasets/COCO',
        images_dirpath = 'Datasets/COCO/train2017',
        annotations_path = 'Datasets/COCO/annotations/instances_train2017.json',
        F_image_id_to_relative_path = lambda image_id: "{:012d}.jpg".format(image_id),
        augmentation = True,
        batch_size = batch_size,
        num_batch = num_batch_train
    )

    DSET_coco_validation = PATCH_DATASET(
        root_dirpath = 'Datasets/COCO',
        images_dirpath = 'Datasets/COCO/val2017',
        annotations_path = 'Datasets/COCO/annotations/instances_val2017.json',
        F_image_id_to_relative_path = lambda image_id: "{:012d}.jpg".format(image_id),
        augmentation = False,
        batch_size = batch_size,
        num_batch = num_batch_val
    )

    M_roi_classifier = M_ROI_CLASSIFIER(num_classes=DSET_coco_training.Get_num_classes()).to('cuda')

    DL_coco_training = DataLoader(DSET_coco_training, batch_size=batch_size, shuffle=False)
    DL_coco_validation = DataLoader(DSET_coco_validation, batch_size=batch_size, shuffle=False)

    def prepare_batch_fn(batch, gt):
        gt = gt.long()
        return batch, gt


    optimizer = torch.optim.SGD(M_roi_classifier.parameters(), lr = 0.0002, momentum = 0.1)
    criterion = torch.nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.CyclicLR(
        optimizer,
        step_size_up = 100,
        base_lr = 0.0002, max_lr = 0.002,
        cycle_momentum=True,
        base_momentum = 0.1, max_momentum = 0.3,
    )

    trainer = Trainer(
        device = 'cuda',
        modes = [Mode.TRAIN, Mode.EVALUATE],
        model = M_roi_classifier,
        data_loaders = {Mode.TRAIN : DL_coco_training, Mode.EVALUATE : DL_coco_validation},
        epochs = epochs,
        starting_epoch = 0,
        optimizer = optimizer,
        criterion = criterion,
        prepare_batch_fn = prepare_batch_fn,
        callbacks = [
            LRBatchSchedulerCallBack(scheduler),
            CalculateLossCallback(key='Loss'),
            CalculateTopNAccuracyCallback(keys=('Top-1 accuracy',), topk=(1,)),
            PlotCallback(plots_path, labels_map={Mode.TRAIN:"Train", Mode.EVALUATE:"Val"}, columns=['Loss', 'Top-1 accuracy']),
            SetTQDMBarDescription(keys=['Loss', 'Top-1 accuracy']),
            SaveParametersCallback(parameters_path),
        ]
    )

    trainer.start()

if __name__ == "__main__":
    main()
