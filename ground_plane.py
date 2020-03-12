import numpy as np
import time

from utils.utils import normalize, running_mean_pad
from utils.ransac import run_ransac, is_inlier, estimate
from utils.visualization import draw_ransac_plane, draw_ransac_plane_two_sets

GP_DEFAULT = np.array([0.0, -1.0, 0.0, 1.65])
CAM_GND_DEFAULT = np.array([0.0, 1.65, 0.0])


def fit_ground_plane_from_obj_points(points_3d, ground_plane_default=GP_DEFAULT, log=False):
    """
    fit ground planes for all frames from object 3d points
    :param points_3d: 3d points from all frames <list>[n_frames]
    :param ground_plane_default: default ground plane, useful when no object points exists
    :param log: print log information to terminal
    :return: ground planes for all frames <list>[n_frames]
    """
    # TODO: calculate CAM_GND_POINT from given ground plane (instead of using default)
    # cam_gnd_default = calculate_cam_gnd(ground_plane_default)
    ground_planes = []
    for frame_id, points_3d_in_frame in enumerate(points_3d):
        # iterate for 3d points for different frames
        if points_3d_in_frame is not None:
            points_3d_in_frame = points_3d_in_frame.T
            # at least one 3d point in current frame
            n_points = points_3d_in_frame.shape[0]
            if n_points == 1:
                ground_plane = calculate_ground_plane_from_one_point(points_3d_in_frame[0])
            elif n_points == 2:
                ground_plane = calculate_ground_plane_from_two_points(points_3d_in_frame[0], points_3d_in_frame[1])
            else:
                ground_plane = calculate_ground_plane_from_mul_points(points_3d_in_frame)
        else:
            # no 3d point in current frame
            if ground_plane_default is None:
                ground_plane = normalize(GP_DEFAULT)
            else:
                ground_plane = normalize(ground_plane_default)

        if log:
            print("Ground plane fitting for frame %s:" % frame_id, ground_plane)

        ground_planes.append(ground_plane)
    ground_planes = np.array(ground_planes)

    return ground_planes


def calculate_ground_plane_from_one_point(point_3d, cam_gnd=CAM_GND_DEFAULT):

    p1 = point_3d
    p2 = cam_gnd
    p3 = generate_point(cam_gnd)    # assume no roll angle and add a point at the right of the camera
    points_3d = np.vstack((p1, p2, p3))

    ground_plane = calculate_ground_plane_from_mul_points(points_3d, cam_gnd=cam_gnd, use_cam_gnd=False)

    return ground_plane


def calculate_ground_plane_from_two_points(point_3d_1, point_3d_2, cam_gnd=CAM_GND_DEFAULT):

    points_3d = np.vstack((point_3d_1, point_3d_2, cam_gnd))
    ground_plane = calculate_ground_plane_from_mul_points(points_3d, cam_gnd=cam_gnd, use_cam_gnd=False)

    return ground_plane


def calculate_ground_plane_from_mul_points(points_3d, cam_gnd=CAM_GND_DEFAULT, use_cam_gnd=True):

    if use_cam_gnd:
        points_3d = np.vstack((points_3d, cam_gnd))

    n_points = points_3d.shape[0]
    params, residuals, _, _ = np.linalg.lstsq(points_3d, - np.ones((n_points, 1)), rcond=-1)
    params = np.squeeze(params)
    params = np.append(params, 1.0)
    params = normalize(params)
    # print(params, residuals)
    # TODO: how to use residuals?

    return params


def generate_point(cam_gnd):
    new_point = cam_gnd + np.array([1.0, 0, 0])
    return new_point


def fit_ground_plane_from_ground_points(points_3d, inlier_pct=0.5, max_iterations=20, viz=False, log=False):

    n_frames = len(points_3d)
    ground_planes = np.empty((n_frames, 4))

    for frame_id in range(n_frames):
        points_3d_in_frame = points_3d[frame_id]
        ground_plane = fit_plane_by_ransac(points_3d_in_frame, inlier_pct=inlier_pct, max_iterations=max_iterations, viz=viz)
        ground_planes[frame_id, :] = ground_plane

        if log:
            print("Ground plane fitting for frame %s:" % frame_id, ground_plane)

    return ground_planes


def fit_plane_by_ransac(points_3d, inlier_pct=0.5, max_iterations=20, viz=False):

    _, n_points = points_3d.shape
    goal_inliers = n_points * inlier_pct

    plane, ic = run_ransac(points_3d.T, estimate, lambda x, y: is_inlier(x, y, 0.01), 3, goal_inliers, max_iterations)

    if viz:
        draw_ransac_plane(points_3d, plane)

    return plane


def adaptive_ground_plane_estimation(dense_points, sparse_points, sparse_points_conf,
                                     inlier_pct=0.5, max_iterations=20, viz=False, log=False):

    n_frames = len(dense_points)
    ground_planes = np.empty((n_frames, 4))

    for frame_id in range(n_frames):
        # start_time = time.time()
        dense = dense_points[frame_id]
        sparse = sparse_points[frame_id]
        if sparse is not None:
            sparse_conf = sparse_points_conf[frame_id]
            md = dense.shape[1]
            ms = sparse.shape[1]
            mdplus = (0.5 * md / ms * sparse_conf).astype(int)
            sparse_aug = sparse_augmentation(sparse, mdplus, mode='repeat')
            points_3d_in_frame = np.hstack((dense, sparse_aug))
        else:
            points_3d_in_frame = dense
        # print("sparse aug time: %s", time.time() - start_time)

        # start_time = time.time()
        ground_plane = fit_plane_by_ransac(points_3d_in_frame, inlier_pct=inlier_pct, max_iterations=max_iterations)
        # print("ransac time: %s", time.time() - start_time)

        if viz:
            if sparse is not None:
                draw_ransac_plane_two_sets(dense, sparse_aug, ground_plane)
            else:
                draw_ransac_plane(points_3d_in_frame, ground_plane)

        ground_planes[frame_id, :] = ground_plane

        if log:
            print("Ground plane fitting for frame %s:" % frame_id, ground_plane)

    return ground_planes


def sparse_augmentation(sparse, mplus, mode='repeat'):
    if mode == 'repeat':
        sparse_aug = np.repeat(sparse, mplus, axis=1)
    if mode == 'gaussian':
        for point_3d, m in zip(sparse.T, mplus):
            gauss_x = np.random.normal(point_3d[0], 0.3, m)
            gauss_y = np.random.normal(point_3d[1], 0.01, m)
            gauss_z = np.random.normal(point_3d[2], 0.1, m)
            gauss = [gauss_x, gauss_y, gauss_z]
            gauss = np.array(gauss)
            # print("***", gauss.shape)
            try:
                sparse_aug = np.hstack((sparse_aug, gauss))
            except:
                sparse_aug = gauss
        # print("****", sparse_aug.shape)

    return sparse_aug


def smoothing_gp(ground_planes):
    win = 2
    n_gp, _ = ground_planes.shape
    # n_gp_new = n_gp - win + 1
    n_gp_new = n_gp
    gp_new = np.zeros((n_gp_new, 4))

    gp_new[:, 0] = running_mean_pad(ground_planes[:, 0], win)
    gp_new[:, 1] = running_mean_pad(ground_planes[:, 1], win)
    gp_new[:, 2] = running_mean_pad(ground_planes[:, 2], win)
    gp_new[:, 3] = running_mean_pad(ground_planes[:, 3], win)

    return gp_new
