import os

from utils.utils import calculate_iou_btw_obj_and_bbox


def read_traj(tracking_file_name, dets):
    """
    read tracking results from txt, including set tracking id for detections and return tracklets
    :param data_root: dataset root directory
    :param dets: detections, list of objects
    :return: tracklets <list> (frame_id, obj_id)
    """
    # set tracking txt file directory
    if not os.path.exists(tracking_file_name):
        print("Warning: No tracking results found!")
        return

    n_frames = len(dets)
    # TODO: a better n_tracklets
    n_tracklets = 2 * n_frames  # set a maximal value of number of tracklets
    tracklets = [None] * n_tracklets
    max_tracklet_id = -1

    with open(tracking_file_name) as f:
        data = f.readlines()
    for line in data:
        if line is not None:
            linelist = line.rstrip().split(',')
            frame_id = int(float(linelist[0]))
            obj_tracklet_id = int(float(linelist[1]))
            left = int(float(linelist[3]))
            top = int(float(linelist[4]))
            width = int(float(linelist[5]))
            height = int(float(linelist[6]))
            score = float(linelist[11])
            category = int(float(linelist[12]))
            dist2cam = float(linelist[13])

            # print(frame_id, obj_tracklet_id, left, top, width, height, score, category)

            bbox = {'x1': left, 'x2': left + width, 'y1': top, 'y2': top + height}
            if dets[frame_id] is not None:
                # detection exists in this frame, find matched detection id
                obj_id = match_obj(dets[frame_id], bbox, obj_tracklet_id)

                # update max tracklet id
                if obj_tracklet_id > max_tracklet_id:
                    max_tracklet_id = obj_tracklet_id

                if obj_id is None:
                    # no matched detction
                    continue
                else:
                    # matched detection found, append tuple (frame_id, obj_id)
                    try:
                        tracklets[obj_tracklet_id].append((frame_id, obj_id))
                    except:
                        tracklets[obj_tracklet_id] = [(frame_id, obj_id)]
            else:
                # no detection in this frame, drop this tracking line
                continue

    return tracklets[:max_tracklet_id+1]


def match_obj(dets_in_frame, bbox, obj_tracking_id):
    """
    match the given bbox with the detections in the corresponding frame
    :param dets_in_frame: detections in this frame
    :param bbox: given bounding box
    :param obj_tracking_id: given tracking id
    :return: matched object id in dets_in_frame, None if no matched
    """
    # find the detection with the max iou with tracking bbox
    max_iou = 0.5
    obj_bbox_match = None

    for obj_id, obj in enumerate(dets_in_frame):
        # iterate over each detection result
        iou = calculate_iou_btw_obj_and_bbox(obj, bbox)
        if iou > max_iou:
            max_iou = iou
            obj_bbox_match = obj_id

    # set matched object's tracking id
    if obj_bbox_match is not None:
        dets_in_frame[obj_bbox_match].set_tracklet_id(obj_tracking_id)

    return obj_bbox_match


if __name__ == '__main__':

    DATE = '2011_09_26'
    DRIVE = '0091'
    FOLDER_NAME = DATE + "_drive_" + DRIVE + "_sync"
    BASEDIR = "/mnt/disk1/kitti-dataset/raw_data/"
    DATA_ROOT = BASEDIR + DATE + "/" + FOLDER_NAME

    read_traj(DATA_ROOT)
