import os
from os import path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image


def clamp(value, min_value, max_value):
    return max(min(value, max_value), min_value)

def Bbox_clamp(bbox, image_width, image_height):
    item_bbox = bbox.copy()
    item_bbox[0] = clamp(item_bbox[0], 0, image_width)
    item_bbox[1] = clamp(item_bbox[1], 0, image_height)
    item_bbox[2] = clamp(item_bbox[2], 0, image_width-item_bbox[0])
    item_bbox[3] = clamp(item_bbox[3], 0, image_height-item_bbox[1])
    return item_bbox

def Resize_image_with_padding(PIL_image, desired_size=240):
    image_size = PIL_image.size  # PIL_image_size is in (width, height) format

    scale_ratio = float(desired_size) / max(image_size)
    new_size = tuple([int(size * scale_ratio) for size in image_size])

    PIL_image = PIL_image.resize(new_size, Image.ANTIALIAS)

    # create a new image and paste the resized on it
    new_PIL_image = Image.new("RGB", (desired_size, desired_size))
    top_left_corner = (
        (desired_size - new_size[0]) // 2,
        (desired_size - new_size[1]) // 2,
    )
    new_PIL_image.paste(PIL_image, top_left_corner)

    return new_PIL_image, top_left_corner


def Convert_bbox_from_TLWH_to_TLBR(bbox):
    """
    Convert boundingbox cordinates from Top-Left and Width-Height to Top-Left Bottom-Right
    """
    new_bbox = bbox.copy()
    new_bbox[2] += new_bbox[0]
    new_bbox[3] += new_bbox[1]

    return new_bbox


def Normalize_bbox_to_0_1(bbox, image_size):
    normalized_bbox = bbox.copy()
    normalized_bbox[0] = clamp(normalized_bbox[0] / image_size[0], 0, 1)
    normalized_bbox[1] = clamp(normalized_bbox[1] / image_size[1], 0, 1)
    normalized_bbox[2] = clamp(normalized_bbox[2] / image_size[0], 0, 1)
    normalized_bbox[3] = clamp(normalized_bbox[3] / image_size[1], 0, 1)
    return normalized_bbox


def Resize_bbox_for_resized_image_with_padding(bbox, original_size, desired_size):

    normalized_bbox = Normalize_bbox_to_0_1(bbox, original_size)
    resized_bbox = [None] * 4

    scale_ratio = float(desired_size) / max(original_size)
    new_size = tuple([int(size * scale_ratio) for size in original_size])
    top_left_corner = (
        (desired_size - new_size[0]) // 2,
        (desired_size - new_size[1]) // 2,
    )

    resized_bbox[0] = (
        normalized_bbox[0] * new_size[0] + top_left_corner[0]
    ) / desired_size
    resized_bbox[1] = (
        normalized_bbox[1] * new_size[1] + top_left_corner[1]
    ) / desired_size
    resized_bbox[2] = (
        normalized_bbox[2] * new_size[0] + top_left_corner[0]
    ) / desired_size
    resized_bbox[3] = (
        normalized_bbox[3] * new_size[1] + top_left_corner[1]
    ) / desired_size

    return resized_bbox


def Show_figure_with_bbox(image, img_size, bbox):
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(image)
    bbox_TL_x = bbox[0] * img_size[0]
    bbox_TL_y = bbox[1] * img_size[1]
    bbox_width = (bbox[2] - bbox[0]) * img_size[0]
    bbox_height = (bbox[3] - bbox[1]) * img_size[1]
    rect = patches.Rectangle(
        (bbox_TL_x, bbox_TL_y),
        bbox_width,
        bbox_height,
        linewidth=1,
        edgecolor="r",
        facecolor="none",
    )
    # Add the patch to the Axes
    # rect_patch = ax.add_patch(rect)
    ax.add_patch(rect)
    fig.show()


class Denormalize_tensor(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor
