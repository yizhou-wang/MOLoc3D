import os
import numpy as np
import pykitti

from Object import Object
import utils.parseTrackletXML as xmlParser

# DATE = '2011_09_26'
# DRIVE = '0091'
# FOLDER_NAME = DATE + "_drive_" + DRIVE + "_sync"
# BASEDIR = "/mnt/disk2/kitti-dataset/raw_data/"
# DATA_ROOT = BASEDIR + DATE + "/" + FOLDER_NAME
#
# # interested classes
# CLASS_NAME_LIST = ['person', 'car', 'bus', 'truck']
#
# data = pykitti.raw(BASEDIR, DATE, DRIVE)
# K_cam2 = data.calib.K_cam2
# T_cam2_velo = data.calib.T_cam2_velo


def velo_to_cam2(x3d, y3d, z3d, T):
    velo_3d = np.array([x3d, y3d, z3d, 0])
    cam2_3d = T.dot(velo_3d)
    return cam2_3d


def read_gt_dets_from_txt(data_root, class_name_list): # load gt detections

    frame_dir = data_root + "/image_02/data"
    dets_file_name = data_root + "/dets_gt.txt"
    frame_num = len([f for f in os.listdir(frame_dir) if f.endswith('.png')])

    dets_gt = [None] * frame_num
    if not os.path.exists(dets_file_name):
        print('Warning: No ground truth detection file found!')
        return dets_gt
    with open(dets_file_name, 'r') as f:
        lines = f.readlines()
        for line in lines:
            box = line.rstrip().split()
            frame_id = int(float(box[0]))
            x1 = int(float(box[1]))
            x2 = int(float(box[2]))
            y1 = int(float(box[3]))
            y2 = int(float(box[4]))
            x1, x2, y1, y2 = adjust_bbox_within_image(x1, x2, y1, y2)
            left = x1
            top = y1
            width = x2 - x1
            height = y2 - y1
            x3d = float(box[5])
            y3d = float(box[6])
            z3d = float(box[7])
            cam2_3d = velo_to_cam2(x3d, y3d, z3d)
            # print(cam2_3d)
            objectType = box[8].lower()
            if objectType == 'pedestrian':
                objectType = 'person'
            if objectType == 'van' or objectType == 'truck': # ignore tram
                objectType = 'car'
            if objectType in class_name_list:
                # obj = Object2D(objectType, x1, x2, y1, y2, x3d, y3d, z3d)
                # obj = Object(objectType, left, top, width, height, x3d=x3d, y3d=y3d, z3d=z3d)
                obj = Object(category=objectType, left=left, top=top, width=width, height=height,
                             x_3d=cam2_3d[0], y_3d=cam2_3d[1], z_3d=cam2_3d[2], matched=True)
                try:
                    dets_gt[frame_id].append(obj)
                except:
                    dets_gt[frame_id] = [obj]
    return dets_gt


def project_cam3d_to_cam2d(points3d, K):

    points2d = K.dot(points3d)

    # scale projected points
    points2d[0, :] = points2d[0,:] / points2d[2,:]
    points2d[1, :] = points2d[1,:] / points2d[2,:]
    points2d[2, :] = 1.0

    return points2d[:2, :]


def project_velo_to_cam3d(points3d, T):

    # project 3d points from vole to cam2
    point_num = points3d.shape[1]
    points3d = np.vstack((points3d, np.ones((1,point_num))))
    points3d = T.dot(points3d)

    return points3d[:3, :]


def project_velo_to_cam(points3d, K, T):

    # project 3d points from vole to cam2
    point_num = points3d.shape[1]
    points3d = np.vstack((points3d, np.ones((1,point_num))))
    points3d = T.dot(points3d)
    points2d = K.dot(points3d[:3, :])

    # scale projected points
    points2d[0, :] = points2d[0,:] / points2d[2,:]
    points2d[1, :] = points2d[1,:] / points2d[2,:]
    points2d[2, :] = 1.0

    return points2d[:2, :]


def adjust_bbox_within_image(x1, x2, y1, y2, im_size=(1242, 375)):

    # adjust bbox within image
    if x1 < 0:
        x1 = 0
        if x2 <= x1:
            x2 = x1 + 1
    if x2 > im_size[0] - 1:
        x2 = im_size[0] - 1
        if x1 >= x2:
            x1 = x2 - 1
    if y1 < 0:
        y1 = 0
        if y2 <= y1:
            y2 = y1 + 1
    if y2 > im_size[1] - 1:
        y2 = im_size[1] - 1
        if y1 >= y2:
            y1 = y2 - 1
    assert x1 < x2, "x1 = %s, x2 = %s" % (x1, x2)
    assert y1 < y2, "y1 = %s, y2 = %s" % (y1, y2)

    return x1, x2, y1, y2


def corner_to_bbox(corner):

    # find original 2d bbox
    x1 = int(np.min(corner[0, :]))
    x2 = int(np.max(corner[0, :]))
    y1 = int(np.min(corner[1, :]))
    y2 = int(np.max(corner[1, :]))

    x1, x2, y1, y2 = adjust_bbox_within_image(x1, x2, y1, y2)

    return x1, x2, y1, y2


def read_gt_dets_from_xml(data_root, class_name_list, dataset):

    dataset_data = dataset.data

    frame_dir = data_root + "/image_02/data"
    frame_names = [f for f in os.listdir(frame_dir) if f.endswith('.png')]
    frame_num = len(frame_names)

    dets_gt = [None] * frame_num

    xml_file_name = data_root + "/tracklet_labels.xml"
    if not os.path.exists(xml_file_name):
        print('Warning: No ground truth detection file found!')
        return dets_gt

    # load tracklets from xml and parse into frame level
    frame_tracklets, frame_tracklets_types, frame_tracklets_locations, frame_tracklets_states = \
        load_tracklets_for_frames(frame_num, xml_file_name)

    for frame_id in range(frame_num):
        corners3d = frame_tracklets[frame_id]
        types = frame_tracklets_types[frame_id]
        locations3d = frame_tracklets_locations[frame_id]
        states = frame_tracklets_states[frame_id]
        for corner3d, type, location3d, (trun, oclu) in zip(corners3d, types, locations3d, states):
            # corner2d = project_velo_to_cam(corner3d)
            corner3d_cam = project_velo_to_cam3d(corner3d, T=dataset_data.calib.T_cam2_velo)
            corner2d = project_cam3d_to_cam2d(corner3d_cam, K=dataset_data.calib.K_cam2)
            x1, x2, y1, y2 = corner_to_bbox(corner2d)
            left = x1
            top = y1
            width = x2 - x1
            height = y2 - y1
            x3d = location3d[0]
            y3d = location3d[1]
            z3d = location3d[2]
            cam2_3d = velo_to_cam2(x3d, y3d, z3d, T=dataset_data.calib.T_cam2_velo)
            # print(cam2_3d)
            objectType = type.lower()
            if objectType == 'pedestrian':
                objectType = 'person'
            if objectType == 'van' or objectType == 'truck': # ignore tram
                objectType = 'car'
            if objectType in class_name_list:
                obj = Object(category=objectType, left=left, top=top, width=width, height=height, corner_3d=corner3d_cam,
                             x_3d=cam2_3d[0], y_3d=cam2_3d[1], z_3d=cam2_3d[2], trun=trun, oclu=oclu)
                try:
                    dets_gt[frame_id].append(obj)
                except:
                    dets_gt[frame_id] = [obj]

    return dets_gt


def load_tracklets_for_frames(n_frames, xml_path):
    """
    Loads dataset labels also referred to as tracklets, saving them individually for each frame.

    Parameters
    ----------
    n_frames    : Number of frames in the dataset.
    xml_path    : Path to the tracklets XML.

    Returns
    -------
    Tuple of dictionaries with integer keys corresponding to absolute frame numbers and arrays as values. First array
    contains coordinates of bounding box vertices for each object in the frame, and the second array contains objects
    types as strings.
    """
    tracklets = xmlParser.parseXML(xml_path)

    frame_tracklets = {}            # tracklets: 8 corners of 3d bbox
    frame_tracklets_types = {}      # types: object type
    frame_tracklets_locations = {}  # locations: object 3d translation in velo
    frame_tracklets_states = {}     # states: (truncation, occlusion)
    for i in range(n_frames):
        frame_tracklets[i] = []
        frame_tracklets_types[i] = []
        frame_tracklets_locations[i] = []
        frame_tracklets_states[i] = []

    # loop over tracklets
    for i, tracklet in enumerate(tracklets):
        # this part is inspired by kitti object development kit matlab code: computeBox3D
        h, w, l = tracklet.size
        # in velodyne coordinates around zero point and without orientation yet
        trackletBox = np.array([
            [-l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2],
            [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2],
            [0.0, 0.0, 0.0, 0.0, h, h, h, h]
        ])
        # loop over all data in tracklet
        for translation, rotation, state, occlusion, truncation, amtOcclusion, amtBorders, absoluteFrameNumber in tracklet:
            # determine if object is in the image; otherwise continue
            if truncation not in (xmlParser.TRUNC_IN_IMAGE, xmlParser.TRUNC_TRUNCATED):
                continue
            # re-create 3D bounding box in velodyne coordinate system
            yaw = rotation[2]  # other rotations are supposedly 0
            assert np.abs(rotation[:2]).sum() == 0, 'object rotations other than yaw given!'
            rotMat = np.array([
                [np.cos(yaw), -np.sin(yaw), 0.0],
                [np.sin(yaw), np.cos(yaw), 0.0],
                [0.0, 0.0, 1.0]
            ])
            cornerPosInVelo = np.dot(rotMat, trackletBox) + np.tile(translation, (8, 1)).T
            frame_tracklets[absoluteFrameNumber] = frame_tracklets[absoluteFrameNumber] + [cornerPosInVelo]
            frame_tracklets_types[absoluteFrameNumber] = frame_tracklets_types[absoluteFrameNumber] + [
                tracklet.objectType]
            frame_tracklets_locations[absoluteFrameNumber] = frame_tracklets_locations[absoluteFrameNumber] + [
                translation]
            frame_tracklets_states[absoluteFrameNumber] = frame_tracklets_states[absoluteFrameNumber] + [
                (truncation, occlusion)]

    return frame_tracklets, frame_tracklets_types, frame_tracklets_locations, frame_tracklets_states


if __name__ == '__main__':
    read_gt_dets_from_xml()
