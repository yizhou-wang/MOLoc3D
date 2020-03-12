import os
import numpy as np

import utils.parseTrackletXML as xmlParser

from utils.utils import calculate_iou_btw_obj
from utils.utils import dist_btw_two_points
from utils.utils import random_colors
# from utils.visualization import visualize_3d_loc_in_frame_demo
# from utils.visualization import visualize_gt_dets

from dataloaders.read_gt_dets import read_gt_dets_from_xml


def evaluate(data_root, class_name_list):
    # read ground truth detection results
    dets_gt = read_gt_dets_from_xml(data_root=data_root, class_name_list=class_name_list)
    # for frame_id, frame_name in enumerate(frame_names):
    #     print(os.path.join(frame_dir, frame_name))
    #     visualize_gt_dets(os.path.join(frame_dir, frame_name), dets_gt[frame_id])
    print(">> Ground truth detection results for %d frames loaded." % len(dets_gt))

    # match detections with ground truth detections
    # match_dets_with_det_gt(dets, dets_gt)




def match_dets_with_det_gt(dets, dets_gt):
    # detections and gt detections must have the same length of frame number
    assert len(dets) == len(dets_gt)
    for frame_id, (dets_in_frame, dets_gt_in_frame) in enumerate(zip(dets, dets_gt)):
        # iterate for each frame
        # print("frame id: %s" % frame_id)
        if dets_in_frame is not None:
            for obj in dets_in_frame:
                # iterate over each detection result
                # find the gt with the max iou
                max_iou = 0.5
                obj_gt_match = None
                if dets_gt_in_frame is not None:
                    for gt_id, obj_gt in enumerate(dets_gt_in_frame):
                        # skip ground truth with truncation
                        if obj_gt.trun == xmlParser.TRUNC_TRUNCATED:
                            continue
                        if obj.category == obj_gt.category:
                            iou = calculate_iou_btw_obj(obj, obj_gt)
                            if iou > max_iou:
                                max_iou = iou
                                obj_gt_match = gt_id
                # only check the detections with matched gt
                if obj_gt_match is not None:
                    obj.set_matched(obj_gt_match)

    return


def calculate_eval_metrics_in_frame(dets, dets_gt, metrics):
    # TODO: add more metrics
    metrics_in_frame = {}
    flag_list = []
    depth_ae_list = []
    depth_pct_list = []
    x_ae_list = []
    dist_ae_list = []
    depth_ae_near_list = []
    depth_pct_near_list = []
    x_ae_near_list = []
    dist_ae_near_list = []
    dist_ae_near_30_list = []
    dist_ae_far_list = []
    for obj, obj_gt in zip(dets, dets_gt):
        if obj_gt is not None:
            # there is a matched gt
            flag_list.append(True)

            # calculate metrics
            if metrics == 'tracklet_smooth':
                if obj.tracklet_id is not None:
                    depth_e = obj.z_3d_trk - obj_gt.z_3d
                    x_e = obj.x_3d_trk - obj_gt.x_3d
                    dist_ae = dist_btw_two_points([obj.x_3d_trk, obj.y_3d_trk, obj.z_3d_trk],
                                                  [obj_gt.x_3d, obj_gt.y_3d, obj_gt.z_3d])
                else:
                    if obj.depth_conf < 0.4 and obj_gt.z_3d > 25:
                        depth_e = obj.z_3d_proj_gp - obj_gt.z_3d
                        x_e = obj.x_3d_proj_gp - obj_gt.x_3d
                        dist_ae = dist_btw_two_points([obj.x_3d_proj_gp, obj.y_3d_proj_gp, obj.z_3d_proj_gp],
                                                      [obj_gt.x_3d, obj_gt.y_3d, obj_gt.z_3d])
                    else:
                        depth_e = obj.depth - obj_gt.z_3d
                        x_e = obj.x_3d_proj_d - obj_gt.x_3d
                        dist_ae = dist_btw_two_points([obj.x_3d_proj_d, obj.y_3d_proj_d, obj.z_3d_proj_d],
                                                      [obj_gt.x_3d, obj_gt.y_3d, obj_gt.z_3d])

            elif metrics == 'proj_gp' and obj.depth_conf < 0.4 and obj_gt.z_3d > 25:
                depth_e = obj.z_3d_proj_gp - obj_gt.z_3d
                x_e = obj.x_3d_proj_gp - obj_gt.x_3d
                dist_ae = dist_btw_two_points([obj.x_3d_proj_gp, obj.y_3d_proj_gp, obj.z_3d_proj_gp],
                                              [obj_gt.x_3d, obj_gt.y_3d, obj_gt.z_3d])
            elif metrics == 'depth':
                depth_e = obj.depth - obj_gt.z_3d
                x_e = obj.x_3d_proj_d - obj_gt.x_3d
                dist_ae = dist_btw_two_points([obj.x_3d_proj_d, obj.y_3d_proj_d, obj.z_3d_proj_d],
                                              [obj_gt.x_3d, obj_gt.y_3d, obj_gt.z_3d])
            else:
                depth_e = obj.z_3d_final - obj_gt.z_3d
                x_e = obj.x_3d_final - obj_gt.x_3d
                dist_ae = dist_btw_two_points([obj.x_3d_final, obj.y_3d_final, obj.z_3d_final],
                                              [obj_gt.x_3d, obj_gt.y_3d, obj_gt.z_3d])

            # # for cars, test if use near center point as ground truth works
            # if obj.category == 'car':
            #     depth_e = obj.depth - obj_gt.z_near_3d

            depth_ae = abs(depth_e)
            depth_pct = depth_ae / obj_gt.z_3d
            x_ae = abs(x_e)

            # append metrics to list
            depth_ae_list.append(depth_ae)
            depth_pct_list.append(depth_pct)
            x_ae_list.append(x_ae)
            dist_ae_list.append(dist_ae)

            # for the near (<15m) objects
            if obj_gt.z_3d <= 15:
                depth_ae_near_list.append(depth_ae)
                depth_pct_near_list.append(depth_pct)
                x_ae_near_list.append(x_ae)
                dist_ae_near_list.append(dist_ae)
            if obj_gt.z_3d <= 30:
                dist_ae_near_30_list.append(dist_ae)
            else:
                dist_ae_far_list.append(dist_ae)


            # set metrics for obj
            obj.set_eval_metrics('flag', True)
            obj.set_eval_metrics('depth_e', depth_e)
            obj.set_eval_metrics('depth_ae', depth_ae)
            obj.set_eval_metrics('depth_pct', depth_pct)
            obj.set_eval_metrics('x_ae', x_ae)
            obj.set_eval_metrics('dist_ae', dist_ae)
        else:
            # no matched gt
            obj.set_eval_metrics('flag', False)
            # obj.set_eval_metrics('depth_e', None)
            # obj.set_eval_metrics('depth_ae', None)
            # obj.set_eval_metrics('depth_pct', None)

    if len(flag_list) == 0:
        metrics_in_frame['flag'] = False
        metrics_in_frame['n_obj'] = 0
        # metrics_in_frame['depth_ae'] = None
        # metrics_in_frame['depth_pct'] = None
        # metrics_in_frame['depth_ae_near'] = None
        # metrics_in_frame['depth_pct_near'] = None

    else:
        metrics_in_frame['flag'] = True
        metrics_in_frame['n_obj'] = len(flag_list)
        metrics_in_frame['depth_ae'] = np.average(np.array(depth_ae_list))
        metrics_in_frame['depth_pct'] = np.average(np.array(depth_pct_list))
        metrics_in_frame['x_ae'] = np.average(np.array(x_ae_list))
        metrics_in_frame['dist_ae'] = np.average(np.array(dist_ae_list))
        if len(depth_ae_near_list) == 0:
            metrics_in_frame['depth_ae_near'] = None
            metrics_in_frame['depth_pct_near'] = None
            metrics_in_frame['x_ae_near'] = None
            metrics_in_frame['dist_ae_near'] = None
        else:
            metrics_in_frame['depth_ae_near'] = np.average(np.array(depth_ae_near_list))
            metrics_in_frame['depth_pct_near'] = np.average(np.array(depth_pct_near_list))
            metrics_in_frame['x_ae_near'] = np.average(np.array(x_ae_near_list))
            metrics_in_frame['dist_ae_near'] = np.average(np.array(dist_ae_near_list))

        if len(dist_ae_near_30_list) == 0:
            metrics_in_frame['dist_ae_near_30'] = None
        else:
            metrics_in_frame['dist_ae_near_30'] = np.average(np.array(dist_ae_near_30_list))
        if len(dist_ae_far_list) == 0:
            metrics_in_frame['dist_ae_far'] = None
        else:
            metrics_in_frame['dist_ae_far'] = np.average(np.array(dist_ae_far_list))

    return metrics_in_frame


def evaluate_3d_loc(data_root, dataset, dets, dets_gt, metrics=None, viz=False, log=True):

    operate_folder = dataset.operate_folder_name
    image_folder = dataset.image_folder_name

    frame_dir = os.path.join(data_root, operate_folder, image_folder)
    frame_names = [os.path.join(frame_dir, f) for f in sorted(os.listdir(frame_dir))]

    # detections and gt detections must have the same length of frame number
    assert len(dets) == len(dets_gt)

    eval_metrics_all_frames = []
    colors = random_colors(200)

    for im_name, dets_in_frame, dets_gt_in_frame in zip(frame_names, dets, dets_gt):
        if log:
            print("Evaluate frame %s" % im_name)
        eval_metrics = {'flag': False}
        # iterate for each frame
        obj_gt_match_in_frame = []
        if dets_in_frame is not None:
            for obj in dets_in_frame:
                if dets_gt_in_frame is not None:
                    if obj.matched:
                        obj_gt_match_in_frame.append(dets_gt_in_frame[obj.matched_id])
                    else:
                        obj_gt_match_in_frame.append(None)
                else:
                    obj_gt_match_in_frame.append(None)
            eval_metrics = calculate_eval_metrics_in_frame(dets_in_frame, obj_gt_match_in_frame, metrics=metrics)
        if viz:
            # visualize_3d_loc_in_frame(im_name, dets_in_frame, obj_gt_match_in_frame, eval_metrics)
            visualize_3d_loc_in_frame_demo(im_name, dets_in_frame, eval_metrics, colors)
            # visualize_gt_dets(im_name, dets_gt_in_frame)
        eval_metrics_all_frames.append(eval_metrics)

    return eval_metrics_all_frames


def evaluate_3d_loc_demo(frame_names, dets, dets_gt, gnd_masks, metrics=None, viz=False, log=True):

    # detections and gt detections must have the same length of frame number
    assert len(dets) == len(dets_gt)

    eval_metrics_all_frames = []
    colors = random_colors(200)

    for im_name, dets_in_frame, dets_gt_in_frame, gnd_mask in zip(frame_names, dets, dets_gt, gnd_masks):
        if log:
            print("Evaluate frame %s" % im_name)
        eval_metrics = {'flag': False}
        # iterate for each frame
        obj_gt_match_in_frame = []
        if dets_in_frame is not None:
            for obj in dets_in_frame:
                if dets_gt_in_frame is not None:
                    if obj.matched:
                        obj_gt_match_in_frame.append(dets_gt_in_frame[obj.matched_id])
                    else:
                        obj_gt_match_in_frame.append(None)
                else:
                    obj_gt_match_in_frame.append(None)
            eval_metrics = calculate_eval_metrics_in_frame(dets_in_frame, obj_gt_match_in_frame, metrics=metrics)
        if viz:
            # visualize_3d_loc_in_frame(im_name, dets_in_frame, obj_gt_match_in_frame, eval_metrics)
            visualize_3d_loc_in_frame_demo(im_name, dets_in_frame, dets_gt_in_frame, gnd_mask, eval_metrics, colors)
            # visualize_gt_dets(im_name, dets_gt_in_frame)
        eval_metrics_all_frames.append(eval_metrics)

    return eval_metrics_all_frames


# def evaluate_3d_loc_old(frame_names, dets, dets_gt, viz=True, log=True):
#     # detections and gt detections must have the same length of frame number
#     assert len(dets) == len(dets_gt)
#
#     eval_metrics_all_frames = []
#
#     for im_name, dets_in_frame, dets_gt_in_frame in zip(frame_names, dets, dets_gt):
#         if log:
#             print("Evaluate frame %s" % im_name)
#         eval_metrics = {'flag': False}
#         # iterate for each frame
#         obj_gt_match_in_frame = []
#         if dets_in_frame is not None:
#             for obj in dets_in_frame:
#                 # iterate over each detection result
#                 # find the gt with the max iou
#                 max_iou = 0.5
#                 obj_gt_match = None
#                 if dets_gt_in_frame is not None:
#                     for obj_gt in dets_gt_in_frame:
#                         # skip ground truth with truncation
#                         if obj_gt.trun == xmlParser.TRUNC_TRUNCATED:
#                             continue
#                         if obj.category == obj_gt.category:
#                             iou = calculate_iou_btw_obj(obj, obj_gt)
#                             if iou > max_iou:
#                                 max_iou = iou
#                                 obj_gt_match = obj_gt
#                 # only check the detections with matched gt
#                 if obj_gt_match is not None:
#                     obj.set_matched()
#                 obj_gt_match_in_frame.append(obj_gt_match)
#             eval_metrics = calculate_eval_metrics_in_frame(dets_in_frame, obj_gt_match_in_frame)
#         if viz:
#             visualize_3d_loc_in_frame(im_name, dets_in_frame, obj_gt_match_in_frame, eval_metrics)
#             # visualize_gt_dets(im_name, dets_gt_in_frame)
#         eval_metrics_all_frames.append(eval_metrics)
#
#     return eval_metrics_all_frames


def print_eval_results(eval_metrics_all_frames):

    print("==============================")
    print("  Get evaluation from %s frames." % len(eval_metrics_all_frames))

    flag_list = []
    n_obj_list = []
    n_obj_near_list = []
    depth_ae_list = []
    depth_pct_list = []
    x_ae_list = []
    dist_ae_list = []
    depth_ae_near_list = []
    depth_pct_near_list = []
    x_ae_near_list = []
    dist_ae_near_list = []
    n_obj_near_30_list = []
    dist_ae_near_30_list = []
    n_obj_far_list = []
    dist_ae_far_list = []
    for metrics in eval_metrics_all_frames:
        if metrics['flag']:
            flag_list.append(True)
            n_obj_list.append(metrics['n_obj'])
            depth_ae_list.append(metrics['depth_ae'])
            depth_pct_list.append(metrics['depth_pct'])
            x_ae_list.append(metrics['x_ae'])
            dist_ae_list.append(metrics['dist_ae'])
            if metrics['depth_ae_near'] is not None:
                n_obj_near_list.append(metrics['n_obj'])
                depth_ae_near_list.append(metrics['depth_ae_near'])
                depth_pct_near_list.append(metrics['depth_pct_near'])
                x_ae_near_list.append(metrics['x_ae_near'])
                dist_ae_near_list.append(metrics['dist_ae_near'])
            if metrics['dist_ae_near_30'] is not None:
                n_obj_near_30_list.append(metrics['n_obj'])
                dist_ae_near_30_list.append(metrics['dist_ae_near_30'])
            if metrics['dist_ae_far'] is not None:
                n_obj_far_list.append(metrics['n_obj'])
                dist_ae_far_list.append(metrics['dist_ae_far'])

    print("  %s frames have valid metrics." % len(flag_list))

    # n_obj_list = np.array(n_obj_list)
    depth_ae_list = np.array(depth_ae_list)
    depth_pct_list = np.array(depth_pct_list)
    x_ae_list = np.array(x_ae_list)
    dist_ae_list = np.array(dist_ae_list)
    depth_ae_near_list = np.array(depth_ae_near_list)
    depth_pct_near_list = np.array(depth_pct_near_list)
    x_ae_near_list = np.array(x_ae_near_list)
    dist_ae_near_list = np.array(dist_ae_near_list)
    dist_ae_near_30_list = np.array(dist_ae_near_30_list)
    dist_ae_far_list = np.array(dist_ae_far_list)

    # print("  Depth AE: %.4f m (%.4f" % (np.average(depth_ae_list), np.average(depth_pct_list)*100) + " %).")
    # print("  Depth AE (near): %.4f m (%.4f" % (np.average(depth_ae_near_list), np.average(depth_pct_near_list)*100) + " %).")

    if len(n_obj_list) != 0:
        # print("  Depth AE: %.4f m (%.4f" % (np.average(depth_ae_list, weights=n_obj_list),
        #                                     np.average(depth_pct_list, weights=n_obj_list)*100) + " %).")
        # print("  X AE: %.4f m." % (np.average(x_ae_list, weights=n_obj_list)))
        dist_ae_all = np.average(dist_ae_list, weights=n_obj_list)
        dist_std_all = np.sqrt(np.average((dist_ae_list - dist_ae_all) ** 2, weights=n_obj_list))
        print("  Dist AE: %.4f m (%.4f)." % (dist_ae_all, dist_std_all))
    else:
        print("  No valid object.")
    if len(n_obj_near_list) != 0:
        # print("  Depth AE (near): %.4f m (%.4f" % (np.average(depth_ae_near_list, weights=n_obj_near_list),
        #                                            np.average(depth_pct_near_list, weights=n_obj_near_list)*100) + " %).")
        # print("  X AE (near): %.4f m." % (np.average(x_ae_near_list, weights=n_obj_near_list)))
        dist_ae_near_all = np.average(dist_ae_near_list, weights=n_obj_near_list)
        dist_std_near_all = np.sqrt(np.average((dist_ae_near_list - dist_ae_near_all) ** 2, weights=n_obj_near_list))
        print("  Dist AE (near): %.4f m (%.4f)." % (dist_ae_near_all, dist_std_near_all))
    else:
        print("  No valid near object.")
    if len(n_obj_near_30_list) != 0:
        dist_ae_near_30_all = np.average(dist_ae_near_30_list, weights=n_obj_near_30_list)
        dist_std_near_30_all = np.sqrt(np.average((dist_ae_near_30_list - dist_ae_near_30_all) ** 2, weights=n_obj_near_30_list))
        print("  Dist AE (near 30): %.4f m (%.4f)." % (dist_ae_near_30_all, dist_std_near_30_all))
    else:
        print("  No valid near 30 object.")
    if len(n_obj_far_list) != 0:
        dist_ae_far_all = np.average(dist_ae_far_list, weights=n_obj_far_list)
        dist_std_far_all = np.sqrt(np.average((dist_ae_far_list - dist_ae_far_all) ** 2, weights=n_obj_far_list))
        print("  Dist AE (far): %.4f m (%.4f)." % (dist_ae_far_all, dist_std_far_all))
    else:
        print("  No valid far object.")

    print("==============================")
