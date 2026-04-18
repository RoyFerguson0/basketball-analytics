""" 
A module providing utility functions for bounding box calculations and measurements.

This module contains helper functions for working with bounding boxes, including calculations for centers, widths and distances between points.
"""


def get_center_of_bbox(bbox):
    """ 
    Calculate the center point of a bounding box.

    Args:
        bbox (tuple): Bounding box coordinates format (x1, y1, x2, y2).

    Returns:
        tuple: The (x_center, y_center) coordinates of the bounding box center.
    """
    x1, y1, x2, y2 = bbox
    return int((x1 + x2) / 2), int((y1 + y2) / 2)


def get_bbox_width(bbox):
    """ 
    Calculate the width of a bounding box.

    Args:
        bbox (tuple): Bounding box coordinates format (x1, y1, x2, y2).

    Returns:
        int: The width of the bounding box.
    """
    return bbox[2] - bbox[0]


def measure_distance(point1, point2):
    """ 
    Calculate the Euclidean distance between two points.

    Args:
        point1 (tuple): The (x, y) coordinates of the first point.
        point2 (tuple): The (x, y) coordinates of the second point.

    Returns:
        float: The Euclidean distance between the two points.
    """
    return ((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2) ** 0.5


def measure_xy_distance(point1, point2):
    """ 
    Calculate the seperate horizontal (x) and vertical (y) distances between two points.

    Args:
        point1 (tuple): The (x, y) coordinates of the first point.
        point2 (tuple): The (x, y) coordinates of the second point.

    Returns:
        tuple: The (x_distance, y_distance) between the points.
    """
    return point1[0] - point2[0], point1[1] - point2[1]


def get_foot_position(bbox):
    """
    Calculate the position of the bottom center point of a bounding box.

    Args:
        bbox (tuple): Bounding box coordinates format (x1, y1, x2, y2).

    Returns:
        tuple: The (x, y) coordinates of the foot position (bottom center) of the bounding box.
    """
    x1, y1, x2, y2 = bbox
    return int((x1 + x2) / 2), int(y2)
