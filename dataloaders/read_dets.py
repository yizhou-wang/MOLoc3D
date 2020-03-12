import os
import pickle
import numpy as np

from Object import Object

CLASS_AREA_LIST = {
    'pedestrian': 1600,
    'cyclist': 2500,
    'car': 3600,
    'truck': 3600,
    'train': 3600
}


def read_mrcnn_dets(data_root, operate_folder, image_folder, dets_file, mask_folder, class_name_list,
                    filter_dets_by_area=False, encode_mask=False):
    """
    Load mask r-cnn detection results from a txt file (bbox) and pkl files (masks).
    Format of the txt file:
        frame_id, obj_id, x, y, w, h, score, class_name
    Name of the pkl file:
        %010d_%02d.pkl" % (frame_id, obj_id)
    :param data_root: data sequence folder path
    :param operate_folder: sub directory with all data and to write results
    :param image_folder: folder stored images
    :param dets_file: name of the mrcnn detection txt
    :param mask_folder: folder for storing the object masks
    :param class_name_list: interested classes
    :param filter_dets_by_area: if filter small objects out by their areas
    :return:
    """
    frame_dir = os.path.join(data_root, operate_folder, image_folder)
    dets_file_name = os.path.join(data_root, operate_folder, dets_file)
    mask_dir = os.path.join(data_root, operate_folder, mask_folder)

    frame_num = len([f for f in os.listdir(frame_dir)])

    dets_mrcnn = [None] * frame_num
    with open(dets_file_name, 'r') as f_dets:
        data = f_dets.readlines()

    for line in data:
        # read mrcnn detections from txt file
        frame_id, obj_id, x, y, w, h, score, class_name = line.rstrip().split(',')

        # select interested classes
        if class_name not in class_name_list:
            continue
        # if class_name == 'bus' or class_name == 'truck':
        #     class_name = 'car'

        frame_id = int(float(frame_id))
        obj_id = int(float(obj_id))
        x = int(float(x))
        y = int(float(y))
        w = int(float(w))
        h = int(float(h))
        score = float(score)

        if filter_dets_by_area:
            area = w * h
            if area < CLASS_AREA_LIST[class_name]:
                continue

        # read mrcnn masks from npy file
        if not encode_mask:
            mask_file = mask_dir + "/%06d_%02d.npy" % (frame_id, obj_id)
            mask = np.load(mask_file)
            obj = Object(category=class_name, left=x, top=y, width=w, height=h, score=score, mask=mask,
                         matched=False, encode_mask=encode_mask)
        else:
            mask_file = mask_dir + "/%06d_%02d.pkl" % (frame_id, obj_id)
            with open(mask_file, 'rb') as f:
                mask_encode = pickle.load(f)
            obj = Object(category=class_name, left=x, top=y, width=w, height=h, score=score, mask=mask_encode,
                         matched=False, encode_mask=encode_mask)
        try:
            dets_mrcnn[frame_id].append(obj)
        except:
            dets_mrcnn[frame_id] = [obj]

    return dets_mrcnn
