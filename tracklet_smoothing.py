import numpy as np
from sklearn.linear_model import LinearRegression, HuberRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel, Matern, ExpSineSquared

from utils.utils import running_mean
from utils.visualization import draw_tracklet, draw_tracklet_compare


def tracklet_smoothing_org(tracklets, dets, apply_final=True, viz_flag=False):
    """
    refine object 3d localization results by tracking
    :param tracklets: <list> (frame_id, obj_id)
    :param dets: detections from detectors for all frames
    :param dets_gt: ground truth detections for all frames
    :return:
    """
    for tracklet in tracklets:
        points_3d = []
        if tracklet is not None:
            if len(tracklet) <= 2:
                # only have one or two points in the tracklet, keep old value
                for point in tracklet:
                    frame_id, obj_id = point
                    obj = dets[frame_id][obj_id]
                    obj.set_tracklet_refine_loc(obj.x_3d_final, obj.y_3d_final, obj.z_3d_final)
            else:
                for point in tracklet:
                    frame_id, obj_id = point
                    obj = dets[frame_id][obj_id]
                    points_3d.append([obj.x_3d_final, obj.y_3d_final, obj.z_3d_final])

                points_3d = np.array(points_3d).T  # points_3d: 3 x n_points
                points_3d_smooth = tracklet_smoothing_regr_huber_seg(points_3d)

                if viz_flag:
                    # if points_3d_gt.shape[0] == 0:
                    #     draw_tracklet(points_3d, points_3d_smooth)
                    # else:
                    #     draw_tracklet(points_3d, points_3d_smooth, points_3d_gt)
                    points_3d_lr = tracklet_smoothing_linear_regr(points_3d)
                    points_3d_hr = tracklet_smoothing_regr_huber(points_3d, alpha=0)
                    points_3d_ts = tracklet_smoothing_regr_huber_seg(points_3d, alpha=0.2)
                    draw_tracklet_compare(points_3d, points_3d_lr, points_3d_hr, points_3d_ts)

                for p_id, point in enumerate(tracklet):
                    frame_id, obj_id = point
                    obj = dets[frame_id][obj_id]
                    point_smooth = points_3d_smooth[:, p_id]
                    obj.set_tracklet_refine_loc(point_smooth[0], point_smooth[1], point_smooth[2],
                                                apply_final=apply_final)

    return


def tracklet_smoothing_after_gpe(tracklets, dets, dets_gt, viz_flag=False):
    """
    refine object 3d localization results by tracking
    :param tracklets: <list> (frame_id, obj_id)
    :param dets: detections from detectors for all frames
    :param dets_gt: ground truth detections for all frames
    :return:
    """
    for tracklet in tracklets:
        points_3d = []
        points_3d_gt = []
        if tracklet is not None:
            if len(tracklet) <= 2:
                # only have one or two points in the tracklet, keep old value
                for p_id, point in enumerate(tracklet):
                    frame_id, obj_id = point
                    obj = dets[frame_id][obj_id]
                    if obj.matched:
                        obj_gt = dets_gt[frame_id][obj.matched_id]
                        if obj.depth_conf < 0.4 and obj_gt.z_3d > 25:  # use gp
                            obj.set_tracklet_refine_loc(obj.x_3d_proj_gp, obj.y_3d_proj_gp, obj.z_3d_proj_gp)
                        else:
                            obj.set_tracklet_refine_loc(obj.x_3d_proj_d, obj.y_3d_proj_d, obj.z_3d_proj_d)
                    else:  # no gt matched, use depth initial results
                        obj.set_tracklet_refine_loc(obj.x_3d_proj_d, obj.y_3d_proj_d, obj.z_3d_proj_d)
            else:
                for point in tracklet:
                    frame_id, obj_id = point
                    obj = dets[frame_id][obj_id]
                    if obj.matched:
                        obj_gt = dets_gt[frame_id][obj.matched_id]
                        if obj.depth_conf < 0.4 and obj_gt.z_3d > 25:  # use gp
                            points_3d.append([obj.x_3d_proj_gp, obj.y_3d_proj_gp, obj.z_3d_proj_gp])
                        else:
                            points_3d.append([obj.x_3d_proj_d, obj.y_3d_proj_d, obj.z_3d_proj_d])
                        points_3d_gt.append([obj_gt.x_3d, obj_gt.y_3d, obj_gt.z_3d])
                    else:  # no gt matched, use depth initial results
                        points_3d.append([obj.x_3d_proj_d, obj.y_3d_proj_d, obj.z_3d_proj_d])

                points_3d = np.array(points_3d).T  # points_3d: 3 x n_points
                points_3d_gt = np.array(points_3d_gt).T  # points_3d_gt: 3 x n_points

                # points_3d_smooth = tracklet_smoothing_regr_huber(points_3d)
                points_3d_smooth = tracklet_smoothing_regr_huber_seg(points_3d)

                if viz_flag:
                    # if points_3d_gt.shape[0] == 0:
                    #     draw_tracklet(points_3d, points_3d_smooth)
                    # else:
                    #     draw_tracklet(points_3d, points_3d_smooth, points_3d_gt)
                    points_3d_lr = tracklet_smoothing_linear_regr(points_3d)
                    points_3d_hr = tracklet_smoothing_regr_huber(points_3d, alpha=0)
                    points_3d_ts = tracklet_smoothing_regr_huber_seg(points_3d, alpha=0.2)
                    if points_3d_gt.shape[0] == 0:
                        draw_tracklet_compare(points_3d, points_3d_lr, points_3d_hr, points_3d_ts)
                    else:
                        draw_tracklet_compare(points_3d, points_3d_lr, points_3d_hr, points_3d_ts, points_3d_gt)

                for p_id, point in enumerate(tracklet):
                    frame_id, obj_id = point
                    obj = dets[frame_id][obj_id]
                    point_smooth = points_3d_smooth[:, p_id]
                    obj.set_tracklet_refine_loc(point_smooth[0], point_smooth[1], point_smooth[2])

    return


def tracklet_smoothing_pred(points_3d):
    """
    tracklet smoothing using prediction
    :param points_3d: 3 x n_points
    :return: smoothed points with the same shape with points_3d
    """
    vectors = np.diff(points_3d)
    n_vectors = vectors.shape[1]
    vectors_new = []

    alpha = 0.6
    beta = 0.4
    vec_prev = vectors[:, -1]
    vectors_new.append(vec_prev)

    for vec_id in reversed(range(n_vectors - 1)):
        vec_real = vectors[:, vec_id]
        vec_new = alpha * vec_real + (1 - alpha) * vec_prev
        vectors_new.append(vec_new)
        vec_prev = beta * vec_prev + (1 - beta) * vec_new

    vectors_new = np.array(vectors_new).T
    # vectors_new_flip = np.fliplr(vectors_new)
    point0 = np.reshape(points_3d[:, -1], (3, 1))
    points_3d_new = np.cumsum(np.hstack((point0, - vectors_new)), axis=1)

    return points_3d_new


def tracklet_smoothing_ravg(points_3d):
    """
    tracklet smoothing using running average
    :param points_3d: 3 x n_points
    :return:
    """
    win = 2
    _, n_points = points_3d.shape

    n_points_new = n_points - win + 1
    points_3d_new = np.zeros((3, n_points_new))

    # running average (more efficient)
    points_3d_new[0] = running_mean(points_3d[0], win)
    points_3d_new[1] = running_mean(points_3d[1], win)
    points_3d_new[2] = running_mean(points_3d[2], win)

    # running average using convolution
    # points_3d_new[0] = np.convolve(points_3d[0], np.ones((win,))/win, mode='valid')
    # points_3d_new[1] = np.convolve(points_3d[1], np.ones((win,))/win, mode='valid')
    # points_3d_new[2] = np.convolve(points_3d[2], np.ones((win,))/win, mode='valid')

    # TODO: pad points_3d_new as the same shape with the input

    return points_3d_new


def tracklet_smoothing_linear_regr(points_3d):
    """
    tracklet smoothing using regression (gaussian process regression)
    :param points_3d: 3 x n_points
    :return:
    """
    # TODO: add depth confidence into consideration
    # confidence can be used to adjust alpha
    _, n_points = points_3d.shape
    # print(points_3d.shape)
    t = np.reshape(np.arange(n_points), (n_points, 1))
    points_3d_new = []
    for coor in points_3d:
        lr = LinearRegression().fit(t, coor)
        # print(huber.score(t, coor))
        coor_new = lr.predict(t)
        coor_final = coor_new
        points_3d_new.append(coor_final)

    points_3d_new = np.array(points_3d_new)
    return points_3d_new


def tracklet_smoothing_regr_huber(points_3d, alpha=0.5):
    """
    tracklet smoothing using regression (gaussian process regression)
    :param points_3d: 3 x n_points
    :return:
    """
    # TODO: add depth confidence into consideration
    # confidence can be used to adjust alpha
    _, n_points = points_3d.shape
    # print('===')
    # print(points_3d.shape)
    # print(points_3d)
    t = np.reshape(np.arange(n_points), (n_points, 1))
    points_3d_new = []
    for coor in points_3d:
        huber = HuberRegressor(epsilon=1.35).fit(t, coor)
        # print(huber.score(t, coor))
        coor_new = huber.predict(t)
        coor_final = alpha * coor + (1 - alpha) * coor_new
        points_3d_new.append(coor_final)

    points_3d_new = np.array(points_3d_new)
    return points_3d_new


def tracklet_smoothing_regr_huber_seg(points_3d, alpha=0.4):
    win = 20
    n_points = points_3d.shape[1]

    if n_points <= win:
        points_3d_new = tracklet_smoothing_regr_huber(points_3d)
    else:
        n_regr = n_points - win + 1
        regr_results = [[[np.nan] * n_points] * 3] * n_regr
        regr_results = np.array(regr_results)
        for i in range(n_regr):
            regr_results[i, :, i:i + win] = tracklet_smoothing_regr_huber(points_3d[:, i:i + win], alpha=alpha)
        # print(regr_results)
        points_3d_new = np.nanmean(regr_results, axis=0)
        # print(points_3d_new)

    return points_3d_new


def tracklet_smoothing_regr_gpr(points_3d):
    """
    tracklet smoothing using regression (gaussian process regression)
    :param points_3d: 3 x n_points
    :return:
    """
    # TODO: add depth confidence into consideration
    # confidence can be used to adjust alpha
    alpha = 0
    _, n_points = points_3d.shape
    # print(points_3d.shape)
    t = np.reshape(np.arange(n_points), (n_points, 1))
    # t_dense = np.reshape(np.arange(0, n_points, 0.1), (10 * n_points, 1))
    points_3d_new = []
    for coor in points_3d:
        # kernel = Matern()
        gpr = GaussianProcessRegressor(alpha=0.1, normalize_y=True).fit(t, coor)
        # print(gpr.score(t, coor))
        coor_new = gpr.predict(t)
        # coor_final = alpha * coor + (1 - alpha) * coor_new
        coor_final = coor_new
        points_3d_new.append(coor_final)

    points_3d_new = np.array(points_3d_new)
    return points_3d_new
