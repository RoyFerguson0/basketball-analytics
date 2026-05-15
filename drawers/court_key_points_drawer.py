import numpy as np
import cv2


class CourtKeypointDrawer:
    """ 
    A drawer class responsible for drawing court keypoints on a sequence of frames.

    Attributes:
        keypoint_colour (str): The color used to draw the keypoints, represented as a hexadecimal string.
        min_keypoint_confidence (float): The minimum confidence required for a keypoint to be drawn.
        min_keypoints_to_draw (int): The minimum number of keypoints required to draw them.
    """

    def __init__(self, keypoint_colour="#ff2c2c", min_keypoint_confidence=0.45, min_keypoints_to_draw=4):
        """ 
        Initialize the CourtKeypointDrawer instance with the specified attributes for drawing court keypoints.

        Args:
            keypoint_colour (str, optional): The color used to draw the keypoints, represented as a hexadecimal string. Defaults to "#ff2c2c".
            min_keypoint_confidence (float, optional): The minimum confidence required for a keypoint to be drawn. Defaults to 0.45.
            min_keypoints_to_draw (int, optional): The minimum number of keypoints required to draw them. Defaults to 4.
        """
        self.keypoint_colour = keypoint_colour
        self.min_keypoint_confidence = min_keypoint_confidence
        self.min_keypoints_to_draw = min_keypoints_to_draw

    def hex_to_bgr(self, hex_colour):
        """ 
        Convert a hexadecimal color string to a BGR tuple.

        Args:
            hex_colour (str): The color represented as a hexadecimal string (e.g., "#ff2c2c").

        Returns:
            tuple: A tuple representing the color in BGR format (e.g., (44, 44, 255)).
        """
        hex_colour = hex_colour.lstrip('#')
        r, g, b = tuple(int(hex_colour[i:i+2], 16) for i in (0, 2, 4))
        return (b, g, r)

    def extract_drawable_keypoints(self, keypoints, frame_width, frame_height):
        """ 
        Extract keypoints that meet the confidence threshold and are within the frame boundaries.

        Args:
            keypoints (list): A list of keypoints, where each keypoint is a dictionary containing 'x', 'y', and 'confidence'.
            frame_width (int): The width of the video frame.
            frame_height (int): The height of the video frame.

        Returns:
            list: A list of drawable keypoints.
        """
        # If keypoints is None or has no detections, return an empty list
        if keypoints is None or keypoints.xy.shape[0] == 0:
            return []

        # Get all keypoints and confidence scores
        all_xy = keypoints.xy.cpu().numpy()
        all_conf = keypoints.conf.cpu().numpy() if keypoints.conf is not None else None

        if all_xy.shape[0] == 0:
            return []

        # If confidence scores are available, select the keypoints from the detection with the highest average confidence
        if all_conf is not None and all_conf.shape[0] > 0:
            detection_idx = int(np.argmax(np.mean(all_conf, axis=1)))
            points_xy = all_xy[detection_idx]
            points_conf = all_conf[detection_idx]
        else:
            # If confidence scores are not available, select the keypoints from the first detection
            detection_idx = 0
            points_xy = all_xy[detection_idx]
            points_conf = np.ones((points_xy.shape[0],), dtype=np.float32)

        # Filter keypoints based on confidence and frame boundaries
        # Only keypoints with confidence above the threshold and within the frame boundaries will be drawn
        drawable_points = []
        for point_idx, (point, conf) in enumerate(zip(points_xy, points_conf)):
            if conf < self.min_keypoint_confidence:
                continue
            x, y = int(point[0]), int(point[1])
            if x <= 0 or y <= 0 or x >= frame_width or y >= frame_height:
                continue
            drawable_points.append((point_idx, int(x), int(y), conf))

        return drawable_points

    def draw(self, frames, court_keypoints):
        """ 
        Draws court keypoints on a given list of frames based on the provided court keypoints information.

        Args:
            frames (list): A list of video frames (images) to draw on.
            court_keypoints (list): A list of court keypoints for each frame, where each element is a list of keypoints corresponding to that frame.

        Returns:
            list: A list of video frames with the court keypoints drawn on them.
        """
        point_colour = self.hex_to_bgr(self.keypoint_colour)

        output_frames = []
        for index, frame in enumerate(frames):
            annotated_frame = frame.copy()

            keypoints = court_keypoints[index]

            frame_height, frame_width = annotated_frame.shape[:2]
            drawable_keypoints = self.extract_drawable_keypoints(
                keypoints, frame_width, frame_height)

            # Skip drawing when too few reliable points are visible in the frame
            if len(drawable_keypoints) < self.min_keypoints_to_draw:
                output_frames.append(annotated_frame)
                continue

            for point_idx, x, y, conf in drawable_keypoints:
                cv2.circle(annotated_frame, (x, y), 6, point_colour, -1)
                cv2.putText(
                    img=annotated_frame,
                    text=f"{point_idx} ({conf:.2f})",
                    org=(x + 6, y - 6),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.45,
                    color=(255, 255, 255),
                    thickness=1,
                    lineType=cv2.LINE_AA
                )
            output_frames.append(annotated_frame)

        return output_frames
