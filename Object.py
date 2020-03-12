import numpy as np

from cfg import Params
from utils.projections import project_by_depth, project_by_ground_plane


class Object:
    def __init__(self, category, left, top, width, height, corner_3d=None,
                 score=None, mask=None, x_3d=None, y_3d=None, z_3d=None, trun=None, oclu=None, alpha=None,
                 matched=False, encode_mask=False):
        # from object detection
        self.category = category
        self.left = left
        self.top = top
        self.width = width
        self.height = height
        self.x1_2d = left
        self.x2_2d = left + width
        self.y1_2d = top
        self.y2_2d = top + height
        self.score = score

        # from mrcnn
        self.mask = mask
        self.encode_mask = encode_mask
        self.matched = matched  # Ture is there is a matching with ground truth
        self.matched_id = None  # matched ground truth id

        # from 3d gt
        self.x_3d = x_3d
        self.y_3d = y_3d
        self.z_3d = z_3d
        self.trun = trun
        self.oclu = oclu
        self.alpha = alpha
        self.corner_3d = corner_3d

        # from depthmap
        self.depth = None
        self.depth_conf = None

        # from gpe
        self.ground_plane = None

        # from 2d to 3d projection
        self.x_gnd_2d = int((self.x1_2d + self.x2_2d) / 2.0)
        self.y_gnd_2d = self.y2_2d
        self.x_3d_proj_d = None
        self.y_3d_proj_d = None
        self.z_3d_proj_d = None
        self.x_3d_proj_gp = None
        self.y_3d_proj_gp = None
        self.z_3d_proj_gp = None

        # from tracking
        self.tracklet_id = None
        self.x_3d_trk = None
        self.y_3d_trk = None
        self.z_3d_trk = None

        self.x_3d_final = None
        self.y_3d_final = None
        self.z_3d_final = None

        # from evaluation
        self.eval_metrics = {}

    def __str__(self):
        return "Object <%s>: depth = %s, eval_metrics = %s" % (self.category, self.depth, self.eval_metrics)

    def set_mask(self, mask):
        self.mask = mask

    def set_depth(self, depth, depth_conf=None):
        self.depth = depth
        self.depth_conf = depth_conf

    def set_matched(self, matched_id):
        """set matched=True when this detection is matched with a ground truth"""
        self.matched = True
        self.matched_id = matched_id

    def set_tracklet_id(self, tracklet_id):
        self.tracklet_id = tracklet_id

    def set_tracklet_refine_loc(self, x_3d, y_3d, z_3d, apply_final=True):
        self.x_3d_trk = x_3d
        self.y_3d_trk = y_3d
        self.z_3d_trk = z_3d

        if apply_final:
            self.x_3d_final = self.x_3d_trk
            self.y_3d_final = self.y_3d_trk
            self.z_3d_final = self.z_3d_trk

    def set_eval_metrics(self, metric_name, metric_value):
        self.eval_metrics[metric_name] = metric_value

    def project_by_depth(self, K, apply_final=True):
        """
        project 2d points to 3d using camera intrinsics and depth from depthmap
        :param K: intrinsic matrix
        :return: pc_3d: object 3d point in camera coordinates, depth_conf
        """
        puv = np.array([self.x_gnd_2d, self.y_gnd_2d])
        puv = puv.reshape((2, 1))
        pc_3d = project_by_depth(puv, K, self.depth)
        pc_3d = np.squeeze(pc_3d)

        self.x_3d_proj_d = pc_3d[0]
        self.y_3d_proj_d = pc_3d[1]
        self.z_3d_proj_d = pc_3d[2]

        if apply_final:
            self.x_3d_final = self.x_3d_proj_d
            self.y_3d_final = self.y_3d_proj_d
            self.z_3d_final = self.z_3d_proj_d

        return pc_3d, self.depth_conf

    def project_by_ground_plane(self, K, ground_plane=None, apply_final=True):
        """
        project 2d points to 3d using camera intrinsics and ground plane
        :param K: intrinsic matrix
        :param ground_plane: ground plane for this object
        :return: pc_3d: object 3d point in camera coordinates
        """
        # input ground plane has higher priority
        if ground_plane is None:
            ground_plane = self.ground_plane
        if ground_plane is None:
            raise ValueError("Ground plane does not exist!")

        puv = np.array([self.x_gnd_2d, self.y_gnd_2d])
        puv = puv.reshape((2, 1))
        pc_3d = project_by_ground_plane(puv, K, ground_plane)
        pc_3d = np.squeeze(pc_3d)

        self.x_3d_proj_gp = pc_3d[0]
        self.y_3d_proj_gp = pc_3d[1]
        self.z_3d_proj_gp = pc_3d[2]

        if apply_final:
            if self.depth_conf < Params['DEPTH_CONF_THRES']:
                self.x_3d_final = self.x_3d_proj_gp
                self.y_3d_final = self.y_3d_proj_gp
                self.z_3d_final = self.z_3d_proj_gp
            elif self.z_3d_final > Params['MAX_CONF_DEPTH']:
                self.x_3d_final = self.x_3d_proj_gp
                self.y_3d_final = self.y_3d_proj_gp
                self.z_3d_final = self.z_3d_proj_gp

        return pc_3d
