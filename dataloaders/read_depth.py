import os
import numpy as np
import cv2
import tqdm

from utils.depthmap import disp_to_depth


def load_depth_from_file(file_name, original_shape):
    disps = np.load(file_name)
    depthmaps = disp_to_depth(disps, original_width=original_shape[0], cap=80)
    depthmaps_reshape = []
    for depthmap in depthmaps:
        # rescale depthmap to the original image size
        depthmap_reshape = cv2.resize(depthmap, dsize=original_shape, interpolation=cv2.INTER_CUBIC)
        depthmaps_reshape.append(depthmap_reshape)
    depthmaps_reshape = np.array(depthmaps_reshape)
    return depthmaps_reshape


def load_depth_from_files(folder_name, original_shape):
    filenames = sorted(os.listdir(folder_name))
    depthmaps_reshape = []
    for filename in tqdm(filenames):
        # rescale depthmap to the original image size
        depthmap = np.load(os.path.join(folder_name, filename))
        depthmap_reshape = cv2.resize(depthmap, dsize=original_shape, interpolation=cv2.INTER_CUBIC)
        depthmaps_reshape.append(depthmap_reshape)
    depthmaps_reshape = np.array(depthmaps_reshape)
    return depthmaps_reshape
