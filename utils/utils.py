import numpy as np
import random
import colorsys


def norm(vectors, ord=None):
    """
    calculate norm values for input vectors
    :param vectors: n vectors with d-dim <numpy>[d x n]
    :param ord: norm order, default=None (norm2)
    :return: norm values <numpy>[n]
    """
    norms = np.linalg.norm(vectors, ord=ord, axis=0)
    # assert norms.shape[0] == vectors.shape[1]
    return norms


def normalize(vectors, ord=None):
    """
    normalize vectors
    :param vectors: n vectors with d-dim <numpy>[d x n]
    :param ord: norm order, default=None (norm2)
    :return: normalized vectors
    """
    norms = norm(vectors, ord=ord)  # <numpy>[n]

    if type(norms) is np.ndarray:
        # remove zero norm by one
        norms[norms == 0] = 1
        norms_tile = np.tile(norms, (vectors.shape[0], 1)).T
    else:
        if norms == 0:
            norms = 1
        norms_tile = np.tile(norms, (vectors.shape[0],))

    vectors_norm = vectors / norms_tile
    # print(vectors.shape)
    # print(vectors_norm.shape)
    assert vectors_norm.shape == vectors.shape
    return vectors_norm


def dist_btw_two_points(p1, p2):
    p1 = np.array(p1)
    p2 = np.array(p2)
    return norm(p1 - p2)


def get_iou(bb1, bb2):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.

    Parameters
    ----------
    bb1 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x1, y1) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner
    bb2 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x1, y1) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner

    Returns
    -------
    float
        in [0, 1]
    """
    assert bb1['x1'] < bb1['x2']
    assert bb1['y1'] < bb1['y2']
    assert bb2['x1'] < bb2['x2']
    assert bb2['y1'] < bb2['y2']

    # determine the coordinates of the intersection rectangle
    x_left = max(bb1['x1'], bb2['x1'])
    y_top = max(bb1['y1'], bb2['y1'])
    x_right = min(bb1['x2'], bb2['x2'])
    y_bottom = min(bb1['y2'], bb2['y2'])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # compute the area of both AABBs
    bb1_area = (bb1['x2'] - bb1['x1']) * (bb1['y2'] - bb1['y1'])
    bb2_area = (bb2['x2'] - bb2['x1']) * (bb2['y2'] - bb2['y1'])

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou


def calculate_iou_btw_obj(obj, obj_gt):
    bb1 = {'x1': obj.x1_2d, 'x2': obj.x2_2d, 'y1': obj.y1_2d, 'y2': obj.y2_2d}
    bb2 = {'x1': obj_gt.x1_2d, 'x2': obj_gt.x2_2d, 'y1': obj_gt.y1_2d, 'y2': obj_gt.y2_2d}
    return get_iou(bb1, bb2)


def calculate_iou_btw_obj_and_bbox(obj, bbox):
    bb1 = {'x1': obj.x1_2d, 'x2': obj.x2_2d, 'y1': obj.y1_2d, 'y2': obj.y2_2d}
    bb2 = bbox
    return get_iou(bb1, bb2)


def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / float(N)


def running_mean_pad(x, N):
    x_pad = np.pad(x, (N // 2, N - 1 - N // 2), mode='edge')
    res = running_mean(x_pad, N)
    return res


def point_project_to_plane(point, plane):
    # TODO: write multiple points projection
    a, b, c, d = plane
    x, y, z = point
    l2_norm = a * a + b * b + c * c
    x_new = x - a * (a * x + b * y + c * z + d) / l2_norm
    y_new = y - b * (a * x + b * y + c * z + d) / l2_norm
    z_new = z - c * (a * x + b * y + c * z + d) / l2_norm

    return [x_new, y_new, z_new]


def random_colors(N, bright=True):
    """
    Generate random colors.
    To get visually distinct colors, generate them in HSV space then
    convert to RGB.
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(colors)
    return colors


def random_colors_cv2(N):
    colors = np.random.randint(0, 255, size=(N, 3))
    colors_out = []
    for color in colors:
        color = tuple(map(int, color))
        colors_out.append(color)
    return colors_out
