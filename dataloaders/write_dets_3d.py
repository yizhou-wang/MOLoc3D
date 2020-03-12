import os
from tqdm import tqdm

from utils.visualization import visualize_dets_3d


def save_dets_3d(data_root, result_dir, dets, dataset, viz_flag=False):
    dets_save_name = dataset.dets_save_name
    dets_viz_save_name = dataset.dets_viz_save_name
    dets_3d_dir = os.path.join(result_dir, dataset.operate_folder_name, dets_save_name)
    dets_3d_viz_dir = os.path.join(result_dir, dataset.operate_folder_name, dets_viz_save_name)

    if not os.path.exists(dets_3d_dir):
        os.makedirs(dets_3d_dir)

    n_frame = len(dets)
    for fid in range(n_frame):
        output_file = dets_3d_dir + '/{0:010}.txt'.format(fid)
        with open(output_file, 'w'):
            pass

    print("Saving results ...")
    for fid, dets_in_frame in enumerate(tqdm(dets)):
        if dets_in_frame is not None:
            output_file = dets_3d_dir + '/{0:010}.txt'.format(fid)
            for obj in dets_in_frame:
                category = obj.category
                # if obj.category == 'Van':
                #     category = 'Car'
                # class_name, trun, occl, alpha, left, top, right, bottom, height, width, length, x_3d, y_3d, z_3d, rot_y, score
                if obj.trun is None:
                    obj.trun = -1
                if obj.oclu is None:
                    obj.oclu = -1
                if obj.alpha is None:
                    obj.alpha = -1
                if obj.depth_conf < 0.5:
                    output_list = category, obj.trun, obj.oclu, obj.alpha, \
                                  obj.x1_2d, obj.y1_2d, obj.x2_2d, obj.y2_2d, \
                                  -1000, -1000, -1000, \
                                  obj.x_3d_proj_gp, obj.y_3d_proj_gp, obj.z_3d_proj_gp, -1000, obj.score
                else:
                    output_list = category, obj.trun, obj.oclu, obj.alpha, \
                                  obj.x1_2d, obj.y1_2d, obj.x2_2d, obj.y2_2d, \
                                  -1000, -1000, -1000, \
                                  obj.x_3d_proj_d, obj.y_3d_proj_d, obj.z_3d_proj_d, -1000, obj.score
                out_str = ' '.join([str(item) for item in output_list])
                with open(output_file, 'a+') as f:
                    f.write(out_str + '\n')
    if viz_flag:
        # print("Visualizing %s" % frame_name)
        if not os.path.exists(dets_3d_viz_dir):
            os.makedirs(dets_3d_viz_dir)
        visualize_dets_3d(data_root, result_dir, dataset, dets)

