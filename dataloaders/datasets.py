import pykitti


class KITTI_raw:
    def __init__(self, basedir, date, drive):
        self.data = pykitti.raw(basedir, date, drive)
        self.original_shape = (1242, 375)
        self.operate_folder_name = 'image_02'
        self.image_folder_name = 'data'
        self.image_ext = 'png'
        self.dets_name = 'mrcnn_dets.txt'
        self.mask_folder_name = 'masks_obj'
        self.depthmaps_name = "depthmap/disparities_pp.npy"
        self.segment_folder_name = 'masks_seg'
        self.tracking_name = 'tracking.txt'

        self.dets_save_name = 'dets_3d'
        self.dets_viz_save_name = 'dets_3d_viz'


class KITTI_tracking:
    def __init__(self, basedir, drive):
        # TODO: finish configurations for kitti tracking dataset
        self.data = pykitti.tracking(basedir, drive)
        self.original_shape = (1242, 375)
        self.operate_folder_name = 'image_02'
        self.image_folder_name = 'data'
        self.image_ext = 'png'
        self.dets_name = 'mrcnn_dts.txt'
        self.mask_folder_name = 'mrcnn_masks'
        self.depthmaps_name = "depthmap/disparities_pp.npy"
        self.segment_folder_name = 'masks_seg'
        self.tracking_name = 'tracking.txt'

        self.dets_save_name = 'dets_3d'
        self.dets_viz_save_name = 'dets_3d_viz'
