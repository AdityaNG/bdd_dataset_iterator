import os
import numpy as np
import yaml
import cv2
import pandas as pd
from PIL import Image
import math

DATASET_BASE = "~/Datasets/Depth_Dataset_Bengaluru"
DEFAULT_CALIB = os.path.join(
    DATASET_BASE, "calibration/pocoX3/calib.yaml"
)
DEFAULT_DATASET = os.path.join(DATASET_BASE, "1658384707877")


def rgb_seg_to_class(seg_frame, color_2_class):
    seg_frame = cv2.cvtColor(seg_frame, cv2.COLOR_BGR2RGB)
    seg_frame_class = np.zeros(
        [
            seg_frame.shape[0],
            seg_frame.shape[1],
        ],
        dtype=int,
    )

    for color in color_2_class:
        seg_frame_class[
            np.all(seg_frame == np.array(color), axis=-1)
        ] = color_2_class[color]

    return seg_frame_class


def get_item_between_timestamp(csv_dat, start_ts, end_ts, fault_delay=0.5):
    """
    Return frame between two given timestamps
    Raise exception if delta between start_ts and
        minimum_ts is greater than fault_delay
    Raise exception if delta between end_ts and
        maximum_ts is greater than fault_delay
    """
    ts_dat = csv_dat[csv_dat["Timestamp"].between(start_ts, end_ts)]
    minimum_ts = min(ts_dat["Timestamp"])
    if abs(minimum_ts - start_ts) > fault_delay:
        raise Exception("out of bounds: |minimum_ts - start_ts|>fault_delay")
    maximum_ts = max(ts_dat["Timestamp"])
    if abs(maximum_ts - end_ts) > fault_delay:
        raise Exception("out of bounds: |maximum_ts - end_ts|>fault_delay")
    return ts_dat


def parse_rot(rot):
    rot = rot.replace("[", "").replace("]", "").replace("\n", "")
    rot = rot.split()
    rot = np.array(rot).astype(np.float32).reshape(3, 3)
    return rot


def semantic_pc_to_numpy(semantic_pc):
    """
    semantic_pc: (N, 4)
        (x, y, z, class_id)

    Returns:
        (N * 4)
    """
    assert (
        semantic_pc.shape[1] == 4
    ), "semantic_pc must have 4 columns, but got {}".format(
        semantic_pc.shape[1]
    )
    return semantic_pc.reshape(-1)


def numpy_to_semantic_pc(semantic_pc_numpy):
    """
    semantic_pc_numpy: (N * 4)
        (x, y, z, class_id)

    Returns:
        (N, 4)
    """
    assert (
        semantic_pc_numpy.shape[0] % 4 == 0
    ), "semantic_pc_numpy must have a multiple of 4 rows, but got {}".format(
        semantic_pc_numpy.shape[0]
    )
    return semantic_pc_numpy.reshape(-1, 4)


def semantic_pc_to_colors_and_pc(semantic_pc, class_2_color):
    """
    semantic_pc: (N, 4)
        (x, y, z, class_id)

    Returns:
        points: (N, 3)
        colors: (N, 3)
    """
    points = semantic_pc[:, :3]
    colors = np.array(
        [class_2_color[class_id] for class_id in semantic_pc[:, 3]]
    )

    # substitute black with white

    # colors[
    #     (
    #         colors[:, 0] == 0 &
    #         colors[:, 1] == 0 &
    #         colors[:, 2] == 0
    #     ), :
    # ] = 255

    return points, colors


def rotate_points(points, angles):
    """
    Rotate the set of points by the given euler angles.

    points: numpy array of shape (N, 3)
        The array containing N points with 3D coordinates (x, y, z).
    a, b, c: float
        The euler angles in degrees

    Returns:
    numpy array of shape (N, 3)
        The rotated points.
    """
    a, b, c = angles

    # Convert the angles from degrees to radians
    a = math.radians(a)
    b = math.radians(b)
    c = math.radians(c)

    # Create the rotation matrices
    rotation_matrix_a = np.array(
        [
            [1, 0, 0],
            [0, math.cos(a), -math.sin(a)],
            [0, math.sin(a), math.cos(a)],
        ]
    )
    rotation_matrix_b = np.array(
        [
            [math.cos(b), 0, math.sin(b)],
            [0, 1, 0],
            [-math.sin(b), 0, math.cos(b)],
        ]
    )
    rotation_matrix_c = np.array(
        [
            [math.cos(c), -math.sin(c), 0],
            [math.sin(c), math.cos(c), 0],
            [0, 0, 1],
        ]
    )

    # Rotate the points using the rotation matrices
    rotated_points = np.dot(points, rotation_matrix_a.T)
    rotated_points = np.dot(rotated_points, rotation_matrix_b.T)
    rotated_points = np.dot(rotated_points, rotation_matrix_c.T)

    return rotated_points
