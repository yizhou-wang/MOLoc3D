import numpy as np


def project_points_3d_to_2d(points_3d, K):
    """
    project 3D points to 2d image
    :param points_3d: 3d points [3 x n_points]
    :param K: camera intrinsics
    :return: 2d points [2 x n_points]
    """
    points_2d = np.dot(K, points_3d)
    p3 = np.tile(points_2d[2], (3, 1))
    points_2d = points_2d / p3

    return points_2d[:2]


def project_all_obj_by_depth(dets, K, dconf_thre, apply_final=True):
    """
    project all objects in all frames to cam3d by depthmap
    :param dets: all detections
    :param K: intrinsic matrix
    :return: a list of cam3d locations for all objects
    """
    n_frames = len(dets)
    points_3d = [None] * n_frames
    points_3d_conf = [None] * n_frames
    for frame_id, dets_in_frame in enumerate(dets):
        if dets_in_frame is None:
            continue
        for obj in dets_in_frame:
            # if obj.matched:  # only project by depth when obj is matched with gt --removed
            pc_3d, conf = obj.project_by_depth(K, apply_final=apply_final)
            # only add objects with high depth confidence to the list
            if conf > dconf_thre:
                try:
                    points_3d[frame_id].append(pc_3d)
                    points_3d_conf[frame_id].append(conf)
                except:
                    points_3d[frame_id] = [pc_3d]
                    points_3d_conf[frame_id] = [conf]
        if points_3d[frame_id] is not None:
            points_3d[frame_id] = np.array(points_3d[frame_id]).T
            points_3d_conf[frame_id] = np.array(points_3d_conf[frame_id])
    return points_3d, points_3d_conf


def project_all_obj_by_ground_plane(dets, K, ground_planes, apply_final=True):
    assert len(dets) == ground_planes.shape[0]
    n_frames = len(dets)
    points_3d = [None] * n_frames
    for frame_id, (dets_in_frame, ground_plane) in enumerate(zip(dets, ground_planes)):
        if dets_in_frame is None:
            continue
        for obj in dets_in_frame:
            pc_3d = obj.project_by_ground_plane(K, ground_plane, apply_final)
            try:
                points_3d[frame_id].append(pc_3d)
            except:
                points_3d[frame_id] = [pc_3d]
        if points_3d[frame_id] is not None:
            points_3d[frame_id] = np.array(points_3d[frame_id])
    return points_3d


def project_by_depth(points_2d, K, depths):
    """
    project 2d points to 3d using camera intrinsics and depth
    :param points_2d: 2d points [2 x n_points] or [3 x n_points]
    :param K: intrinsic matrix [3 x 3]
    :param depths: depth for each point [n_points]
    :return: projected 3d points [3 x n_points]
    """
    if points_2d.shape[1] == 1:
        # only one point as input
        depths = np.array([depths])
    assert points_2d.shape[1] == depths.shape[0]

    # convert to homogeneous coordinates
    if points_2d.shape[0] == 2:
        points_2d = np.vstack((points_2d, np.ones((1,points_2d.shape[1]))))

    # tile depths to matrix for element-wise multiply
    depths_tile = np.tile(depths, (3,1))
    assert depths_tile.shape == points_2d.shape

    K_inv = np.linalg.inv(K)
    points_3d = depths_tile * np.dot(K_inv, points_2d)

    return points_3d


def project_by_ground_plane(points_2d, K, ground_plane):
    """
    project 2d points to 3d using camera intrinsics and ground plane
    :param points_2d: 2d points [2 x n_points] or [3 x n_points]
    :param K: intrinsic matrix [3 x 3]
    :param ground_plane: ground plane for this frame [4]
    :return: projected 3d points [3 x n_points]
    """
    gnd_n = ground_plane[:3]
    gnd_n_T = gnd_n.reshape((1,3))
    gnd_h = ground_plane[3]
    K_inv = np.linalg.inv(K)

    # convert to homogeneous coordinates
    if points_2d.shape[0] == 2:
        points_2d = np.vstack((points_2d, np.ones((1,points_2d.shape[1]))))

    K_inv_p2d = np.dot(K_inv, points_2d)    # [3 x n_points]
    up = - gnd_h * K_inv_p2d                # [3 x n_points]
    down = np.dot(gnd_n_T, K_inv_p2d)       # [1 x n_points]
    down_tile = np.tile(down, (3,1))        # [3 x n_points]

    points_3d = up / down_tile

    return points_3d


def project_gnd_points_by_depth_all(gnd_masks, depthmaps, K, sample=10):
    """
    project ground points in all frames by ground depth
    :param gnd_masks: bool mask where True means ground <numpy>[n_frames x 375 x 1242]
    :param depthmaps: depth maps for all frames         <numpy>[n_frames x 375 x 1242]
    :param K: intrinsic matrix
    :param sample: sample rate for ground points
    :return: list of 3d points [n_frames] (each element <numpy>[3 x n_points]), number of 3d points in total
    """
    assert gnd_masks.shape == depthmaps.shape  # ground masks and depth maps must have the same shape
    n_frames = gnd_masks.shape[0]
    n_points = 0
    points_3d_list = []
    for frame_id in range(n_frames):
        points_3d = project_ground_points_by_depth(gnd_masks[frame_id], depthmaps[frame_id], K, sample)
        n_points += points_3d.shape[1]
        points_3d_list.append(points_3d)
    return points_3d_list, n_points


def project_ground_points_by_depth(ground_mask, depthmap, K, sample):
    """
    project ground points for one frame by ground depth
    :param ground_mask: bool mask   <numpy>[375 x 1242]
    :param depthmap: depth map      <numpy>[375 x 1242]
    :param K: intrinsic matrix
    :param sample: sample rate for ground points
    :return: projected points       <numpy>[3 x n_points]
    """
    assert ground_mask.shape == depthmap.shape

    # indices = np.where(ground_mask)
    indices = np.where((ground_mask == True) & (depthmap != -1.0)) # peter revised
    indices = np.asarray(indices).T.tolist()

    points_2d = []
    depths = []
    for x, y in indices[::sample]:
        points_2d.append(np.array([y, x]))
        depths.append(depthmap[x, y])

    points_2d = np.array(points_2d).T
    depths = np.array(depths)

    points_3d = project_by_depth(points_2d, K, depths)

    return points_3d


def project_obj_3d_point_cloud(obj, depthmap, d_ave, d_win, K, sample=1):
    """
    project 2d object points by depthmap
    :param obj: object              <class Object>
    :param depthmap: depth map      <numpy>[375 x 1242]
    :param K: intrinsic matrix
    :param sample: sample rate for object points
    :return: projected points       <numpy>[3 x n_points]
    """
    mask = obj.mask
    assert mask.shape == depthmap.shape

    indices = np.where(mask)
    indices = np.asarray(indices).T.tolist()

    points_2d = []
    depths = []
    for x, y in indices[::sample]:
        d = depthmap[x, y]
        if d_ave - d_win < d < d_ave + d_win:
            points_2d.append(np.array([y, x]))
            depths.append(depthmap[x, y])

    points_2d = np.array(points_2d).T
    depths = np.array(depths)

    points_3d = project_by_depth(points_2d, K, depths)

    return points_3d


if __name__ == '__main__':

    DATE = '2011_09_26'
    DRIVE = '0091'
    FOLDER_NAME = DATE + "_drive_" + DRIVE + "_sync"
    BASEDIR = "/mnt/disk1/kitti-dataset/raw_data/"
    DATA_ROOT = BASEDIR + DATE + "/" + FOLDER_NAME

    import pykitti
    data = pykitti.raw(BASEDIR, DATE, DRIVE)
    K_cam2 = data.calib.K_cam2

    point_2d = np.array([500, 200])
    point_2d = point_2d.reshape((2,1))
    point_3d = project_by_depth(point_2d, K_cam2, 10)
    # gp = np.array([0.0,-1.0,0.0,1.65])
    # point_3d = project_by_ground_plane(point_2d, K_cam2, gp)

    print(point_3d)
