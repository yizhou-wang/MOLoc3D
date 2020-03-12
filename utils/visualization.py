import os
import numpy as np
import cv2
import skimage.io
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import proj3d
from matplotlib import patches
from matplotlib.patches import FancyArrowPatch
from matplotlib.patches import Polygon
from skimage.measure import find_contours

from utils.ransac import plot_plane
from utils.utils import random_colors, random_colors_cv2

FONT = cv2.FONT_HERSHEY_SIMPLEX
colors = random_colors(200)
colors_cv2 = random_colors_cv2(200)


class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        FancyArrowPatch.draw(self, renderer)


def visualize_dets_3d(data_root, result_dir, dataset, dets):
    for fid, dets_in_frame in enumerate(dets):
        im_name = os.path.join(data_root, dataset.operate_folder_name, dataset.image_folder_name,
                               '{0:010}.'.format(fid) + dataset.image_ext)
        save_fig_name = os.path.join(result_dir, dataset.operate_folder_name, dataset.dets_viz_save_name,
                                     '{0:010}.jpg'.format(fid))
        visualize_dets_3d_in_frame(im_name, dets_in_frame, colors_cv2, save_fig_name=save_fig_name)


def visualize_dets_3d_eval(data_root, result_dir, dataset, dets, dets_gt, eval_metrics):
    for fid, (dets_in_frame, dets_gt_in_frame) in enumerate(zip(dets, dets_gt)):
        im_name = os.path.join(data_root, dataset.operate_folder_name, dataset.image_folder_name,
                               '{0:010}.'.format(fid) + dataset.image_ext)
        save_fig_name = os.path.join(result_dir, dataset.operate_folder_name, dataset.dets_viz_save_name,
                                     '{0:010}.jpg'.format(fid))
        visualize_dets_3d_eval_in_frame(im_name, dets_in_frame, dets_gt_in_frame, eval_metrics, colors_cv2,
                                        save_fig_name=save_fig_name)


def visualize_dets_3d_eval_demo(data_root, result_dir, dataset, dets, dets_gt, gnd_masks, eval_metrics):
    for fid, (dets_in_frame, dets_gt_in_frame, gnd_mask) in enumerate(zip(dets, dets_gt, gnd_masks)):
        im_name = os.path.join(data_root, dataset.operate_folder_name, dataset.image_folder_name,
                               '{0:010}.'.format(fid) + dataset.image_ext)
        save_fig_name = os.path.join(result_dir, dataset.operate_folder_name, dataset.dets_viz_save_name,
                                     '{0:010}.jpg'.format(fid))
        visualize_dets_3d_eval_in_frame_demo(im_name, dets_in_frame, dets_gt_in_frame, gnd_mask, eval_metrics, colors,
                                             save_fig_name=save_fig_name)


def visualize_dets_3d_in_frame(im_name, dets_in_frame, colors, save_fig_name=None):
    im = cv2.imread(im_name)
    im = draw_dets(im, dets_in_frame, colors)

    if save_fig_name is None:
        cv2.imshow("Visualization", im)
        cv2.waitKey(10)
    else:
        cv2.imwrite(save_fig_name, im)


def visualize_dets_3d_eval_in_frame(im_name, dets_in_frame, obj_gt_match_in_frame, eval_metrics, colors,
                                    save_fig_name=None):
    im = cv2.imread(im_name)
    im = draw_dets(im, obj_gt_match_in_frame, gt=True)
    im = draw_dets(im, dets_in_frame, colors)
    if eval_metrics['flag']:
        tmp_text = "Depth AE: %s m (%s" % (
            round(eval_metrics['depth_ae'], 2), round(eval_metrics['depth_pct'] * 100, 2)) + " %)."
        im = cv2.putText(im, tmp_text, (10, 15), FONT, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
        if eval_metrics['depth_ae_near'] is not None:
            tmp_text = "Depth AE (near): %s m (%s" % (
                round(eval_metrics['depth_ae_near'], 2), round(eval_metrics['depth_pct_near'] * 100, 2)) + " %)."
            im = cv2.putText(im, tmp_text, (10, 30), FONT, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

    if save_fig_name is None:
        cv2.imshow("Visualization", im)
        cv2.waitKey(10)
    else:
        cv2.imwrite(save_fig_name, im)


def visualize_dets_3d_eval_in_frame_demo(im_name, dets_in_frame, obj_gt_match_in_frame, gnd_mask, eval_metrics, colors,
                                         save_fig_name=None):
    im = skimage.io.imread(im_name)
    f, (ax, ax2) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [1, 1]}, figsize=(30, 20))

    im = draw_ground_mask(im, gnd_mask, figsize=(30, 10), ax=ax, color=(0.8, 0.8, 0.8))
    im = draw_dets_mask(im, dets_in_frame, ax, colors)

    if eval_metrics['flag']:
        tmp_text = "Error: %s m (%s" % (
            round(eval_metrics['depth_ae'], 2), round(eval_metrics['depth_pct'] * 100, 2)) + " %)"
        im = cv2.putText(im, tmp_text, (10, 20), FONT, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        if eval_metrics['depth_ae_near'] is not None:
            tmp_text = "Error (near): %s m (%s" % (
                round(eval_metrics['depth_ae_near'], 2), round(eval_metrics['depth_pct_near'] * 100, 2)) + " %)"
            im = cv2.putText(im, tmp_text, (10, 40), FONT, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

    # Show area outside image boundaries.
    height, width = im.shape[:2]
    ax.set_ylim(height, 0)
    ax.set_xlim(0, width)
    ax.axis('off')
    ax.imshow(im)

    obj_xz, obj_gt_xz = get_obj_bev_xz(dets_in_frame, obj_gt_match_in_frame)
    ax2.scatter(obj_gt_xz[0], obj_gt_xz[1], c='lawngreen', s=1000, marker='*', label='Ground Truth')
    ax2.scatter(obj_xz[0], obj_xz[1], c='orangered', s=500, marker='o', label='Our Results')
    ax2.tick_params(axis='both', which='major', labelsize=20)
    ax2.tick_params(axis='both', which='minor', labelsize=15)
    ax2.set_xlim(-20, 20)
    ax2.set_ylim(0, 50)
    ax2.set_xlabel('X (m)', fontsize=20)
    ax2.set_ylabel('Z (m)', fontsize=20)
    ax2.grid(linestyle='--', linewidth='3')
    ax2.legend(prop={'size': 20})

    if save_fig_name is None:
        plt.show()
    else:
        plt.savefig(save_fig_name)
    plt.close()


def draw_dets(im, det, colors=None, gt=False):
    if det is None:
        return im
    for obj in det:
        if obj is None:
            # for gt=True: no gt matched
            continue
        if colors is not None and obj.tracklet_id is not None:
            color = colors[obj.tracklet_id]
        else:
            if gt:
                color = (113, 204, 46)
            else:
                color = (255, 185, 116)
        im = cv2.rectangle(im, (obj.left, obj.top), (obj.left + obj.width, obj.top + obj.height), color, 2)
        if obj.depth_conf is not None:
            result_str = obj.category + ' ' + str(round(obj.depth_conf, 4))
        else:
            result_str = obj.category
        im = cv2.putText(im, result_str, (obj.left, obj.top - 5), FONT, 0.3, color, 1, cv2.LINE_AA)
        if gt:
            result_str = "(%s, %s, %s)" % (round(obj.x_3d, 1), round(obj.y_3d, 1), round(obj.z_3d, 1))
            im = cv2.putText(im, result_str, (obj.left, obj.top + obj.height - 5), FONT, 0.3, (255, 255, 255), 1,
                             cv2.LINE_AA)
        else:
            result_str = "(%s, %s, %s)" % (round(obj.x_3d_final, 1), round(obj.y_3d_final, 1), round(obj.z_3d_final, 1))
            im = cv2.putText(im, result_str, (obj.left, obj.top + obj.height - 5), FONT, 0.3, (255, 255, 255), 1,
                             cv2.LINE_AA)
    return im


def draw_dets_mask(im, det, ax, colors=None):
    if det is None:
        return im
    for obj in det:
        if obj is None:
            # for gt=True: no gt matched
            continue
        if obj.tracklet_id is not None:
            color = colors[obj.tracklet_id]
        else:
            color = None
        result_str = obj.category
        im = cv2.putText(im, result_str, (obj.left, obj.top - 5), FONT, 0.3, color, 1, cv2.LINE_AA)
        result_str = "(%s, %s, %s)" % (round(obj.x_3d_final, 1), round(obj.y_3d_final, 1), round(obj.z_3d_final, 1))
        im = cv2.putText(im, result_str, (obj.left, obj.top + obj.height - 5), FONT, 0.3, (0, 0, 255), 1,
                         cv2.LINE_AA)
        box = [obj.y1_2d, obj.x1_2d, obj.y2_2d, obj.x2_2d]
        im = display_mrcnn(im, box, obj.mask, result_str, ax=ax, color=color)
    return im


def display_mrcnn(image, box, mask, obj_text, figsize=(30, 10), ax=None, color=None):
    """
    boxes: [num_instance, (y1, x1, y2, x2, class_id)] in image coordinates.
    masks: [height, width, num_instances]
    class_ids: [num_instances]
    class_names: list of class names of the dataset
    scores: (optional) confidence scores for each box
    title: (optional) Figure title
    show_mask, show_bbox: To show masks and bounding boxes or not
    figsize: (optional) the size of the image
    colors: (optional) An array or colors to use with each object
    captions: (optional) A list of strings to use as captions for each object
    """
    # If no axis is passed, create one and automatically call show()
    if not ax:
        _, ax = plt.subplots(1, figsize=figsize)

    if color is None:
        # Generate random colors
        colors = random_colors(1)
        color = colors[0]

    masked_image = image.astype(np.uint32).copy()

    # Bounding box
    y1, x1, y2, x2 = box
    p = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2,
                          alpha=0.5, linestyle="dashed",
                          edgecolor=color, facecolor='none')
    ax.add_patch(p)

    # Label
    t = ax.text(x1, y2 + 5, obj_text, color='w', size=12, weight='bold', family='sans-serif', backgroundcolor=color)
    t.set_bbox(dict(facecolor=color, alpha=0.8, edgecolor=color))

    # Mask
    masked_image = apply_mask(masked_image, mask, color, alpha=0.5)
    masked_image = masked_image.astype(np.uint8)

    # Mask Polygon
    # Pad to ensure proper polygons for masks that touch image edges.
    padded_mask = np.zeros(
        (mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
    padded_mask[1:-1, 1:-1] = mask
    contours = find_contours(padded_mask, 0.5)
    for verts in contours:
        # Subtract the padding and flip (y, x) to (x, y)
        verts = np.fliplr(verts) - 1
        p = Polygon(verts, facecolor="none", edgecolor=color)
        ax.add_patch(p)

    return masked_image


def draw_ground_mask(im, mask, figsize=(30, 10), ax=None, color=None):
    # If no axis is passed, create one and automatically call show()
    if not ax:
        _, ax = plt.subplots(1, figsize=figsize)

    if color is None:
        color = (0.5, 0.5, 0.5)

    im = apply_mask(im, mask, color=color, alpha=0.3)
    im = im.astype(np.uint8)

    # Mask Polygon
    # Pad to ensure proper polygons for masks that touch image edges.
    padded_mask = np.zeros(
        (mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
    padded_mask[1:-1, 1:-1] = mask
    contours = find_contours(padded_mask, 0.5)
    for verts in contours:
        # Subtract the padding and flip (y, x) to (x, y)
        verts = np.fliplr(verts) - 1
        p = Polygon(verts, facecolor="none", edgecolor=color)
        ax.add_patch(p)

    return im


def apply_mask(image, mask, color, alpha=0.5):
    """Apply the given mask to the image.
    """
    if len(image.shape) == 3:
        for c in range(3):
            image[:, :, c] = np.where(mask == 1,
                                      image[:, :, c] *
                                      (1 - alpha) + alpha * color[c] * 255,
                                      image[:, :, c])
    elif len(image.shape) == 2:
        image[:, :] = np.where(mask == 1,
                               image[:, :] *
                               (1 - alpha) + alpha * 0.8 * 255,
                               image[:, :])

    return image


def get_obj_bev_xz(dets_in_frame, dets_gt_in_frame):
    if dets_in_frame is None:
        return None, None
    x_list = []
    z_list = []
    x_gt_list = []
    z_gt_list = []
    for obj in dets_in_frame:
        if obj is None:
            continue
        obj_gt = dets_gt_in_frame[obj.matched_id]
        x_gt = obj_gt.x_3d
        x_gt_list.append(x_gt)
        z_gt = obj_gt.z_3d
        z_gt_list.append(z_gt)
        if obj.tracklet_id is not None:
            x = obj.x_3d_trk
            z = obj.z_3d_trk
        else:
            if obj.depth_conf < 0.4 and obj_gt.z_3d > 25:
                z = obj.z_3d_proj_gp
                x = obj.x_3d_proj_gp
            else:
                z = obj.depth
                x = obj.x_3d_proj_d
        x_list.append(x)
        z_list.append(z)
    return [x_list, z_list], [x_gt_list, z_gt_list]


def draw_ransac_plane(points_3d, plane):
    fig = plt.figure()
    ax = mplot3d.Axes3D(fig)

    points_3d = points_3d[:, ::20]
    ax.scatter3D(points_3d[0], points_3d[1], points_3d[2])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_zlim([0, 80])
    ax.set_xlim([-20, 20])
    ax.set_ylim([-10, 20])

    a, b, c, d = plane
    xx, yy, zz = plot_plane(a, b, c, d)
    ax.plot_surface(xx, yy, zz, color=(0, 1, 0, 0.5))

    plt.show()


def draw_ransac_plane_two_sets(points_3d_1, points_3d_2, plane):
    fig = plt.figure()
    ax = mplot3d.Axes3D(fig)

    points_3d_1 = points_3d_1[:, ::20]
    points_3d_2 = points_3d_2[:, ::20]
    ax.scatter3D(points_3d_1[0], points_3d_1[1], points_3d_1[2])
    ax.scatter3D(points_3d_2[0], points_3d_2[1], points_3d_2[2], c='r')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_zlim([0, 80])
    ax.set_xlim([-20, 20])
    ax.set_ylim([-10, 20])

    a, b, c, d = plane
    xx, yy, zz = plot_plane(a, b, c, d)
    ax.plot_surface(xx, yy, zz, color=(0, 1, 0, 0.5))

    plt.show()


def draw_hist(depths):
    fig = plt.figure()
    plt.hist(depths, bins=np.arange(0, 100, 2))
    plt.xlabel('Depth (m)')
    plt.ylabel('Frequency')


def draw_depthmap(depthmap, obj):
    plt.figure()
    plt.imshow(depthmap)
    plt.gca().add_patch(
        plt.Rectangle((obj.left, obj.top),
                      obj.width, obj.height, fill=False,
                      edgecolor='r', linewidth=1)
    )


def draw_tracklet(points_3d, points_3d_new, points_3d_gt=None):
    fig = plt.figure()
    ax = mplot3d.Axes3D(fig)

    ax.scatter3D(points_3d[0], points_3d[1], points_3d[2], color='gray', s=3)
    ax.plot(points_3d[0], points_3d[1], points_3d[2], color='skyblue')
    if points_3d_new is not None:
        ax.scatter(points_3d_new[0], points_3d_new[1], points_3d_new[2], color='gray', s=3)
        ax.plot(points_3d_new[0], points_3d_new[1], points_3d_new[2], color='violet')
    if points_3d_gt is not None:
        ax.scatter(points_3d_gt[0], points_3d_gt[1], points_3d_gt[2], color='gray', s=3)
        ax.plot(points_3d_gt[0], points_3d_gt[1], points_3d_gt[2], color='greenyellow')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_zlim([0, 80])
    ax.set_xlim([-20, 20])
    ax.set_ylim([-10, 20])

    plt.show()


def draw_tracklet_compare(points_3d, points_3d_lr, points_3d_hr, points_3d_ts, points_3d_gt=None):
    fig = plt.figure()
    ax = mplot3d.Axes3D(fig)

    if points_3d_gt is not None:
        ax.scatter(points_3d_gt[0], points_3d_gt[1], points_3d_gt[2], color='gray', s=3)
        ax.plot(points_3d_gt[0], points_3d_gt[1], points_3d_gt[2], color='limegreen', linewidth=3.0,
                label='Ground Truth')

    # ax.scatter3D(points_3d[0], points_3d[1], points_3d[2], color='gray', s=3)
    ax.plot(points_3d[0], points_3d[1], points_3d[2], color='deepskyblue', alpha=0.5, label='Initial Track')

    if points_3d_ts is not None:
        # ax.scatter(points_3d_lr[0], points_3d_lr[1], points_3d_lr[2], color='gray', s=3)
        ax.plot(points_3d_lr[0], points_3d_lr[1], points_3d_lr[2], color='darkorange', alpha=0.5,
                label='Linear Regression')
        # ax.scatter(points_3d_hr[0], points_3d_hr[1], points_3d_hr[2], color='gray', s=3)
        ax.plot(points_3d_hr[0], points_3d_hr[1], points_3d_hr[2], color='darkviolet', alpha=0.5,
                label='Huber Regression')
        # ax.scatter(points_3d_ts[0], points_3d_ts[1], points_3d_ts[2], color='gray', s=3)
        ax.plot(points_3d_ts[0], points_3d_ts[1], points_3d_ts[2], color='red', label='Tracklet Smoothing')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_zlim([0, 80])
    ax.set_xlim([-20, 20])
    ax.set_ylim([-10, 20])
    ax.legend()

    # Hide grid lines
    ax.grid(False)

    # Hide axes ticks
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])

    plt.show()


def draw_all_tracklets(p_3d_in_tracklets, p_3d_in_tracklets_gt):
    fig = plt.figure()
    ax = mplot3d.Axes3D(fig)

    for points_3d in p_3d_in_tracklets:
        ax.plot(points_3d[0], points_3d[1], points_3d[2], color='b')

    for points_3d_gt in p_3d_in_tracklets_gt:
        ax.plot(points_3d_gt[0], points_3d_gt[1], points_3d_gt[2], color='g')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_zlim([0, 80])
    ax.set_xlim([-20, 20])
    ax.set_ylim([-10, 20])

    plt.show()


def draw_3d_point_cloud(points_3d, sample=50, plane=None, ax=None):
    if ax is None:
        fig = plt.figure()
        ax = mplot3d.Axes3D(fig)

    points_3d = points_3d[:, ::sample]
    # ax.scatter3D(points_3d[0], points_3d[1], points_3d[2])
    ax.scatter3D(points_3d[0], points_3d[1], points_3d[2], s=1, alpha=0.5)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_zlim([10, 20])
    ax.set_xlim([-10, 0])
    ax.set_ylim([-5, 5])

    # plane = [0, -1, 0, 1.65]
    if plane is not None:
        a, b, c, d = plane
        xx, yy, zz = plot_plane(a, b, c, d)
        ax.plot_surface(xx, yy, zz, color=(0, 1, 0, 0.5))

    # Hide grid lines
    ax.grid(False)

    # Hide axes ticks
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])


def draw_3d_point_cloud_with_eigen(points_3d, mean_3d, eigenvalues, eigenvectors):
    mean_x, mean_y, mean_z = mean_3d

    fig = plt.figure()
    ax = mplot3d.Axes3D(fig)
    # ax = fig.add_subplot(211, projection='3d')

    points_3d = points_3d[:, ::50]
    ax.scatter(points_3d[0], points_3d[1], points_3d[2], c='r', marker='o')  # yizhou implementation
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    ax.set_zlim([0, 40])
    ax.set_xlim([-20, 20])
    ax.set_ylim([-5, 5])

    ax.plot([mean_x], [mean_y], [mean_z], 'o', markersize=10, color='g', alpha=0.5)

    for eigenvalue, eigenvector in zip(eigenvalues, eigenvectors):
        arrow = Arrow3D([mean_x, mean_x + 10 * eigenvector[0]], [mean_y, mean_y + 10 * eigenvector[1]], \
                        [mean_z, mean_z + 10 * eigenvector[2]], lw=1, arrowstyle="-|>", color="b")
        ax.add_artist(arrow)


def visualize_ae_conf(dets_mrcnn, show=True, save=False):
    x_conf = []
    y_depth_error = []

    for frame_id, dets_in_frame in enumerate(dets_mrcnn):
        if dets_in_frame is None:
            continue
        for obj in dets_in_frame:
            if obj.depth_conf == None or obj.eval_metrics['flag'] == False:
                continue

            x_conf.append(obj.depth_conf)
            y_depth_error.append(obj.eval_metrics['depth_ae'])

    x_conf = np.array(x_conf)
    y_depth_error = np.array(y_depth_error)
    fig = plt.figure()
    plt.plot(x_conf, y_depth_error, 'o', markersize=3)  # marker small circle, size
    plt.xlabel('Depth confidence')
    plt.ylabel('Depth error (m)')

    x_conf_inds = x_conf.argsort()
    x_conf_sorted = x_conf[x_conf_inds]
    y_depth_error_sorted = y_depth_error[x_conf_inds]
    x_conf_hist, bin_edges = np.histogram(x_conf, bins=np.arange(0, 1.001, 0.05))
    # print(x_conf_hist, bin_edges)
    # print(x_conf_sorted)
    avg_depth_error_list = []
    i = 0
    sum = 0
    for conf, error in zip(x_conf_sorted, y_depth_error_sorted):
        if conf <= bin_edges[i + 1]:
            sum += error
        else:
            avg_depth_error_list.append(sum / x_conf_hist[i])
            # print(i, avg_depth_error_list)
            i += 1
            sum = 0
    avg_depth_error_list.append(sum / x_conf_hist[i])
    # print(i, avg_depth_error_list)
    # print(avg_depth_error_list)
    avg_depth_error_list = np.array(avg_depth_error_list)

    import scipy.interpolate
    y_interp = scipy.interpolate.interp1d(np.arange(0.025, 1, 0.05), avg_depth_error_list * 5, kind='cubic')

    x_interp = np.arange(0.025, 0.95, 0.01)
    plt.plot(x_interp, y_interp(x_interp), '--')

    if save:
        plt.savefig('conf_error.png')

    if show:
        plt.show()

    plt.close(fig)
