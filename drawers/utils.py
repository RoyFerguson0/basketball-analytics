""" 
TODO - Docs
"""
import cv2
import numpy as np
from utils import get_center_of_bbox, get_bbox_width, get_foot_position


def draw_triangle(frame, bbox, colour, conf=None):
    """ 
    TODO - Docs
    """
    y = int(bbox[1])
    x_center, y_center = get_center_of_bbox(bbox)

    triangle_points = np.array([
        [x_center, y],
        [x_center - 10, y - 20],
        [x_center + 10, y - 20]
    ])

    cv2.drawContours(
        image=frame,
        contours=[triangle_points],
        contourIdx=0,
        color=colour,
        thickness=cv2.FILLED
    )

    cv2.drawContours(
        image=frame,
        contours=[triangle_points],
        contourIdx=0,
        color=(0, 0, 0),
        thickness=2
    )

    if conf:
        conf_text = f"{conf:.2f} conf"
        cv2.putText(
            frame,
            conf_text,
            (x_center + 10, y + 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            2
        )

    return frame


def draw_ellipse(frame, bbox, colour, track_id=None, conf=None):
    """ 
    TODO - Docs
    """
    y2 = int(bbox[3])
    x_center, y_center = get_center_of_bbox(bbox)
    width = get_bbox_width(bbox)

    cv2.ellipse(
        img=frame,
        center=(x_center, y2),
        axes=(int(width), int(0.35*width)),
        angle=0.0,
        startAngle=-45,
        endAngle=235,
        color=colour,
        thickness=2,
        lineType=cv2.LINE_4
    )

    rectangle_width = 40
    rectangle_height = 20
    x1_rect = x_center - rectangle_width // 2
    x2_rect = x_center + rectangle_width // 2
    y1_rect = (y2 - rectangle_height // 2) + int(0.35*width)
    y2_rect = (y2 + rectangle_height // 2) + int(0.35*width)

    if track_id:
        cv2.rectangle(
            frame,
            (int(x1_rect), int(y1_rect)),
            (int(x2_rect), int(y2_rect)),
            colour,
            cv2.FILLED
        )

        x1_text = x1_rect + 12
        if track_id > 99:
            x1_text -= 10

        cv2.putText(
            frame,
            f"{track_id}",
            (int(x1_text), int(y1_rect + 15)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            2
        )

    if conf:
        conf_text = f"{conf:.2f} conf"
        x1_text = x2_rect + 12

        cv2.putText(
            frame,
            conf_text,
            (int(x1_text), int(y2_rect)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            2
        )

    return frame
