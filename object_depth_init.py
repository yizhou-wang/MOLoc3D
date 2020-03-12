import math
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from sklearn.decomposition import PCA
from pycocotools import mask

from utils.projections import project_obj_3d_point_cloud
from utils.visualization import Arrow3D
from utils.visualization import draw_hist, draw_depthmap, draw_3d_point_cloud, draw_3d_point_cloud_with_eigen
from utils.visualization import display_mrcnn


def init_object_depth(dets, depthmaps, K, encode_mask, viz_flag=False):
    for frame_id, (depthmap, dets_in_frame) in enumerate(zip(depthmaps, dets)):
        if dets_in_frame is None:
            continue
        # # rescale depthmap to the original image size
        # depthmap_reshape = cv2.resize(depthmap, dsize=original_shape, interpolation=cv2.INTER_CUBIC)
        # print('frame id: %s' % frame_id)
        for obj in dets_in_frame:
            # # for visualize depth histogram and depthmap
            # viz_flag = False
            # if frame_id == 12:
            #     viz_flag = True
            obj_depth, depth_conf = obj_depth_from_depthmap(depthmap, obj, K, encode_mask, viz_flag)
            obj.set_depth(obj_depth, depth_conf)

    return dets


def obj_depth_from_depthmap(depthmap, obj, K, encode_mask, viz_flag=False):
    # confidence weights
    l1 = 0.5
    l2 = 1 - l1

    obj_mask = obj.mask
    if encode_mask:
        obj_mask = mask.decode(obj_mask)
        obj_mask = np.array(obj_mask, dtype=bool)
    assert depthmap.shape == obj_mask.shape

    # select depth from mask
    depths = depthmap.ravel()[obj_mask.ravel()]

    # calculate histogram
    if obj.category == 'person':
        hist, bin_edges = np.histogram(depths, bins=np.arange(0, 100, 2), density=True)
        bin_low_idx = np.argmax(hist)
        bin_high_idx = np.argmax(hist) + 1
    elif obj.category == 'car':
        hist, bin_edges = np.histogram(depths, bins=np.arange(0, 100, 2), density=True)
        # solve '-2' makes index negative
        bin_low_idx = np.argmax(hist) - 3 if np.argmax(hist) - 3 >= 0 else 0
        bin_high_idx = np.argmax(hist) + 4
    else:
        hist, bin_edges = np.histogram(depths, bins=np.arange(0, 100, 2), density=True)
        bin_low_idx = np.argmax(hist) - 2 if np.argmax(hist) - 2 >= 0 else 0
        bin_high_idx = np.argmax(hist) + 3

    bin_low = bin_edges[bin_low_idx]
    bin_high = bin_edges[bin_high_idx]

    # use average depth in the select bins to be the output object depth
    depth_ave = np.average(depths)
    depth_var = np.average((depths - depth_ave) ** 2)
    depths_select = depths[np.logical_and(depths > bin_low, depths < bin_high)]
    depth_std = np.sqrt(depth_var)
    depth_select_ave = np.average(depths_select)
    # depth_select_var = np.average((depths_select - depth_select_ave) ** 2)

    # calculate depth confidence scores
    # print(bin_low_idx, bin_high_idx)
    # print(hist)
    # print(hist[bin_low_idx:bin_high_idx])
    # print(hist[bin_low_idx:bin_high_idx] * np.diff(bin_edges[bin_low_idx:bin_high_idx+1]))

    if obj.category == 'car':
        points_3d = project_obj_3d_point_cloud(obj, depthmap, d_ave=depth_select_ave, d_win=4, K=K)
        # points_3d = filter_3d_point_cloud(points_3d)
        mean_3d, eigenvalues, eigenvectors = get_principle_vectors(points_3d, project_dim=3)

        if viz_flag:
            draw_3d_point_cloud(points_3d)
            draw_3d_point_cloud_with_eigen(points_3d, mean_3d, eigenvalues, eigenvectors)

    # first term
    conf1 = 1 - np.std(depths) / np.mean(depths)
    # second term
    conf2 = np.sum(hist[bin_low_idx:bin_high_idx] * np.diff(bin_edges[bin_low_idx:bin_high_idx + 1]))
    # conf = conf1
    # conf = conf2
    conf = conf1 * conf2
    # print("all depth   \tmean: %s, var: %s, conf:%s" % (depth_ave, depth_var, conf))
    # print("select depth\tmean: %s, var: %s, conf:%s" % (depth_select_ave, depth_select_var, conf))

    if viz_flag:
        draw_hist(depths)
        # draw_depthmap(depthmap, obj)
        display_mrcnn(depthmap, (obj.y1_2d, obj.x1_2d, obj.y2_2d, obj.x2_2d), obj.mask)
        plt.show()
        plt.close('all')

    if math.isnan(depth_select_ave):
        print("Warning: depth_select_ave is nan!")
        print("depth_select_ave =", depths)
        print("bin_low = %s, bin_high = %s" % (bin_low, bin_high))

    return depth_select_ave, conf


def get_principle_vectors(p3d_obj, project_dim=3):
    # start pca
    p3d_obj = p3d_obj.T
    pca = PCA(n_components=project_dim)
    pca.fit(p3d_obj)

    # for eigenvalue, eigenvector in zip(pca.explained_variance_, pca.components_):
    #   print(eigenvector)
    #   print(eigenvalue)
    # print('---------')
    # print(pca.components_)
    # print(eigenvalues)

    mean_x = np.mean(p3d_obj[:, 0])
    mean_y = np.mean(p3d_obj[:, 1])
    mean_z = np.mean(p3d_obj[:, 2])
    # pca end

    return [mean_x, mean_y, mean_z], pca.explained_variance_, pca.components_
