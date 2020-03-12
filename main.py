import os
import numpy as np
import time
import argparse

from dataloaders.datasets import KITTI_raw
from dataloaders.read_dets import read_mrcnn_dets
from dataloaders.read_depth import load_depth_from_file
from dataloaders.read_seg import load_ground_masks_all
from dataloaders.read_tracking import read_traj
from utils.projections import project_all_obj_by_depth, project_gnd_points_by_depth_all, \
    project_all_obj_by_ground_plane
from utils.depthmap import add_gaussian_noise

from object_depth_init import init_object_depth
from ground_plane import fit_ground_plane_from_ground_points, fit_ground_plane_from_obj_points, \
    adaptive_ground_plane_estimation, smoothing_gp
from tracklet_smoothing import tracklet_smoothing_org, tracklet_smoothing_after_gpe

from dataloaders.write_dets_3d import save_dets_3d
from utils.visualization import visualize_dets_3d, visualize_dets_3d_eval, visualize_dets_3d_eval_demo

from dataloaders.read_gt_dets import read_gt_dets_from_xml
from evaluation import match_dets_with_det_gt
from evaluation import evaluate_3d_loc, evaluate_3d_loc_demo
from evaluation import print_eval_results


def parse_args():
    parser = argparse.ArgumentParser(description='run moloc3d project')
    parser.add_argument('--data_root', type=str, help='directory for the data sequence')
    parser.add_argument('--classes', type=str, default='pedestrian,car', help='interested classes')
    parser.add_argument('--modules', type=str, default='depth,agpe,ots', help='include what kind(s) of module')
    parser.add_argument('--dataset', type=str, default='kitti', help='use kitti settings')
    parser.add_argument('--encode_mask', action='store_true', help='load encoded masks or not')
    parser.add_argument('--result_dir', type=str, help='directory for the results')
    args = parser.parse_args()
    return args


def run_moloc3d(data_root, class_name_list, modules, dataset, encode_mask=True):
    """
    main function for running moloc3d
    :param data_root: KITTI_raw dataset sequence folder path
    :param class_name_list: interested classes
    :param modules: modules included (depth, agpe, ots, vo)
    :param dataset_config_dict: dataset configurations
    :return: None
    """
    dataset_data = dataset.data
    original_shape = dataset.original_shape
    operate_folder = dataset.operate_folder_name
    image_folder = dataset.image_folder_name
    dets_folder = dataset.dets_name
    mask_folder = dataset.mask_folder_name
    segment_folder = dataset.segment_folder_name
    tracking_folder = dataset.tracking_name

    ##################################################
    # Load detection results
    ##################################################
    # read mrcnn detection results
    print("Loading Mask R-CNN detection results ...")
    dets_all = read_mrcnn_dets(data_root, operate_folder, image_folder, dets_folder, mask_folder, class_name_list,
                               encode_mask=encode_mask)
    print(">> Mask R-CNN detection results for %d frames loaded." % len(dets_all))

    if 'depth' in modules or 'dgpe' in modules or 'agpe' in modules:
        ##################################################
        # Load depth maps
        ##################################################
        print("Loading depth maps ...")
        start_time = time.time()
        depth_file_name = os.path.join(data_root, operate_folder, dataset.depthmaps_name)
        if not os.path.exists(depth_file_name):
            raise ValueError("depthmap does not exist!")
        depthmaps = load_depth_from_file(depth_file_name, original_shape)
        print(">> depthmap shape %s loaded with eclipse time %s seconds." %
              (depthmaps.shape, time.time() - start_time))

        # depthmaps = add_gaussian_noise(depthmaps, sigma=0.1)
        # print(">> Gaussian noise added to depthmaps: simga = %s\n" % sigma)

    if 'depth' in modules:
        ##################################################
        # Object histogram analysis
        ##################################################
        # initialize object depth
        print("Initializing object depth from depthmap ...")
        start_time = time.time()
        init_object_depth(dets_all, depthmaps, K=dataset_data.calib.K_cam2, encode_mask=encode_mask, viz_flag=False)
        print(">> Object depths calculated from depthmap with eclipse time %s seconds." % (time.time() - start_time))

        # project objects to cam3d using depth
        print("Projecting all objects to Cam3D ...")
        start_time = time.time()
        points_3d_obj, points_3d_obj_conf = project_all_obj_by_depth(dets_all, K=dataset_data.calib.K_cam2,
                                                                     dconf_thre=0.4, apply_final=True)
        print(">> projection finished with eclipse time %s second." % (time.time() - start_time))

    if 'dgpe' in modules or 'agpe' in modules:
        ##################################################
        # Dense ground features
        ##################################################
        # load ground masks from segmentation results
        print("Loading segmentation masks ...")
        start_time = time.time()
        seg_folder_path = os.path.join(data_root, operate_folder, segment_folder)
        gnd_masks = load_ground_masks_all(seg_folder_path, original_shape, erode=False, viz_flag=False)
        print(">> segmentation masks shape %s loaded with eclipse time %s seconds." %
              (gnd_masks.shape, time.time() - start_time))

        # project points on ground plane to cam3d
        print("Projecting all ground points to Cam3D ...")
        start_time = time.time()
        points_3d_gnd, n_points_all = project_gnd_points_by_depth_all(gnd_masks, depthmaps, dataset_data.calib.K_cam2,
                                                                      sample=100)
        print(
            ">> total %s points are projected with eclipse time %s seconds." % (n_points_all, time.time() - start_time))

    if 'dgpe' in modules:
        ##################################################
        # DGPE
        ##################################################
        # fit ground planes for all frames from depth maps
        print("Fitting ground planes using ground depth ...")
        start_time = time.time()
        ground_planes_dense = fit_ground_plane_from_ground_points(points_3d_gnd, inlier_pct=0.5, max_iterations=20,
                                                                  viz=False, log=False)
        ground_planes = smoothing_gp(ground_planes_dense)
        print(">> Ground plane fitting with eclipse time %s seconds." % (time.time() - start_time))

    if 'sgpe' in modules:
        ##################################################
        # SGPE
        ##################################################
        # fit ground planes using object cam3d points
        print("Fitting ground plane using object points ...")
        start_time = time.time()
        try:
            gp_default = np.average(ground_planes_dense, axis=0)
        except:
            gp_default = None
        ground_planes_sparse = fit_ground_plane_from_obj_points(points_3d_obj, ground_plane_default=gp_default,
                                                                log=False)
        ground_planes = smoothing_gp(ground_planes_sparse)
        print(">> Ground plane fitted with eclipse time %s second." % (time.time() - start_time))

    if 'agpe' in modules:
        ##################################################
        # AGPE
        ##################################################
        # fit ground planes for all frames from depth maps
        print("Fitting ground planes using adaptive GPE ...")
        start_time = time.time()
        ground_planes = adaptive_ground_plane_estimation(points_3d_gnd, points_3d_obj, points_3d_obj_conf,
                                                         inlier_pct=0.5, max_iterations=20, viz=False, log=False)
        ground_planes = smoothing_gp(ground_planes)
        print(">> Ground plane fitting from depthmap with eclipse time %s seconds." % (time.time() - start_time))

    # ground_planes_agpe = np.load("/home/yzwang/Research/3D-Loc/final-yenting/v5_yizhou/adaptive_gp_smoothing.npy")

    ##################################################
    # Project unreliable objects by ground planes
    ##################################################
    # project objects to cam3d using fitted ground planes
    print("Reprojecting object points using ground planes ...")
    start_time = time.time()
    points_3d_ground_plane = project_all_obj_by_ground_plane(dets_all, dataset_data.calib.K_cam2, ground_planes,
                                                             apply_final=True)
    print(">> reprojection finished with eclipse time %s second." % (time.time() - start_time))

    if 'ots' in modules:
        ##################################################
        # Tracklet smoothing
        ##################################################
        tracking_file_name = os.path.join(data_root, operate_folder, tracking_folder)
        tracklets = read_traj(tracking_file_name, dets_all)
        tracklet_smoothing_org(tracklets, dets_all, viz_flag=False, apply_final=True)
        print(">> tracklet smoothing finished.")

    return dets_all


def parse_kitti_drive_id(folder_name):
    try:
        year, month, day, str1, drive, str2 = folder_name.split('_')
        assert str1 == 'drive'
        assert str2 == 'sync'
        date = '_'.join([year, month, day])
        return date, drive
    except TypeError as e:
        print(e)
        print("Error: folder name needs to be consistent with KITTI_raw")


def split_folder_name(path):
    if path.endswith('/'):
        folder_name = path[:-1].split('/')[-1]
        base_dir = path[:-(len(folder_name) + 12)]  # remove "date" from base_dir
    else:
        folder_name = path.split('/')[-1]
        base_dir = path[:-(len(folder_name) + 11)]  # remove "date" from base_dir
    return base_dir, folder_name


if __name__ == '__main__':
    """
    Example:
        python main.py --data_root /mnt/disk2/kitti-dataset/raw_data/2011_09_26/2011_09_26_drive_0091_sync/ \
            --classes pedestrian,car --modules depth,agpe,ots,vo
    """
    args = parse_args()
    data_root = args.data_root
    base_dir, folder_name = split_folder_name(data_root)
    date, drive = parse_kitti_drive_id(folder_name)
    classes = args.classes.split(',')
    modules = args.modules.split(',')
    dataset_flag = args.dataset
    encode_mask = args.encode_mask
    result_dir = args.result_dir
    # if 'depth' not in modules:
    #     raise ValueError("depth module is required for MOLoc3D")
    print("data_root: %s" % data_root)
    print("folder_name: %s" % folder_name)
    print("classes: %s" % classes)
    print("modules: %s" % modules)
    print("dataset: %s" % dataset_flag)

    if dataset_flag == 'kitti_raw':
        dataset = KITTI_raw(base_dir, date, drive)
        print("===== KITTI_raw data loaded =====")
    else:
        # TODO: add other dataset settings
        raise ValueError

    dets_all = run_moloc3d(data_root, classes, modules, dataset, encode_mask)
    dets_gt = read_gt_dets_from_xml(data_root, classes, dataset)
    match_dets_with_det_gt(dets_all, dets_gt)
    eval_metrics_all_frames = evaluate_3d_loc(data_root, dataset, dets_all, dets_gt, log=False)
    print_eval_results(eval_metrics_all_frames)
    save_dets_3d(data_root, result_dir, dets_all, dataset, viz_flag=True)

    # if dataset_flag == 'kitti':
    #     dets_gt = read_gt_dets_from_xml(data_root, classes, dataset)
    #     visualize_dets_3d()
