import os
import numpy as np
import cv2


def load_ground_masks_all(seg_folder_path, original_shape, erode=False, viz_flag=False):
    if not os.path.exists(seg_folder_path):
        raise ValueError("segmentation masks does not exist!")
    gnd_masks = []
    for file_name in sorted(os.listdir(seg_folder_path)):
        dir_name = os.path.join(seg_folder_path, file_name)
        mask = load_ground_mask_from_file(dir_name, original_shape, erode, viz_flag)
        gnd_masks.append(mask)
    gnd_masks = np.array(gnd_masks)
    return gnd_masks


def load_ground_mask_from_file(file_name, original_shape, erode=False, viz_flag=False):
    """
    load a ground mask from deeplab segmentation result for one frame
    :param file_name: mask file name (*.npy)
    :param original_shape: original shape of image frame
    :param erode: erode mask or not
    :return: loaded ground mask
    """
    seg = np.load(file_name)
    # select mask from deeplab segmentation result, ground==0
    mask0 = np.array(seg == 0)
    mask1 = np.array(seg == 1)
    mask = np.logical_or(mask0, mask1)

    binary_img = np.array(mask, dtype=np.uint8) * 255

    # reshape ground mask
    if mask.shape != original_shape:
        binary_img = cv2.resize(binary_img, original_shape)
        mask = np.array(binary_img == 255)
        if viz_flag:
            viz_name = file_name.replace('/masks_seg/', '/masks_seg_viz/')
            viz_name = viz_name.replace('.npy', '.jpg')
            # cv2.imshow('binary', binary_img)
            # cv2.imshow('resized', resized_img)
            # cv2.waitKey(0)
            cv2.imwrite(viz_name, binary_img)

    # add erosion to filter out noise from segmentation
    if erode:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 1))
        eroded_img = cv2.erode(binary_img, kernel)
        mask = np.array(eroded_img == 255)

    return mask
