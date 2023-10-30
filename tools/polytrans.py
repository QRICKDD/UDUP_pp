import torch
import numpy as np
def poly2bbox(polygon):
    assert len(polygon) % 2 == 0
    polygon = np.array(polygon, dtype=np.float32)
    x = polygon[::2]
    y = polygon[1::2]
    return np.array([min(x), min(y), max(x), max(y)])

def bbox2poly(bbox)->np.array:
    assert len(bbox) == 4
    x1, y1, x2, y2 = bbox
    poly = np.array([x1, y1, x2, y1, x2, y2, x1, y2])
    return poly

def crop_img(src_img, box, long_edge_pad_ratio=0.4, short_edge_pad_ratio=0.2):
    """Crop text region given the bounding box which might be slightly padded.
    The bounding box is assumed to be a quadrangle and tightly bound the text
    region.

    Args:
        src_img (np.array): The original image.
        box (list[float | int]): Points of quadrangle.
        long_edge_pad_ratio (float): The ratio of padding to the long edge. The
            padding will be the length of the short edge * long_edge_pad_ratio.
            Defaults to 0.4.
        short_edge_pad_ratio (float): The ratio of padding to the short edge.
            The padding will be the length of the long edge *
            short_edge_pad_ratio. Defaults to 0.2.

    Returns:
        np.array: The cropped image.
    """
    assert len(box) == 8
    assert 0. <= long_edge_pad_ratio < 1.0
    assert 0. <= short_edge_pad_ratio < 1.0

    h, w = src_img.shape[:2]
    points_x = np.clip(np.array(box[0::2]), 0, w)
    points_y = np.clip(np.array(box[1::2]), 0, h)

    box_width = np.max(points_x) - np.min(points_x)
    box_height = np.max(points_y) - np.min(points_y)
    shorter_size = min(box_height, box_width)

    if box_height < box_width:
        horizontal_pad = long_edge_pad_ratio * shorter_size
        vertical_pad = short_edge_pad_ratio * shorter_size
    else:
        horizontal_pad = short_edge_pad_ratio * shorter_size
        vertical_pad = long_edge_pad_ratio * shorter_size

    left = np.clip(int(np.min(points_x) - horizontal_pad), 0, w)
    top = np.clip(int(np.min(points_y) - vertical_pad), 0, h)
    right = np.clip(int(np.max(points_x) + horizontal_pad), 0, w)
    bottom = np.clip(int(np.max(points_y) + vertical_pad), 0, h)

    dst_img = src_img[top:bottom, left:right]

    return dst_img
"""
quad = bbox2poly(poly2bbox(polygon)).tolist()
rec_inputs.append(crop_img(img, quad))

注意这里img是（610，727，3）
"""