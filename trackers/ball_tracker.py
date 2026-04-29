from ultralytics import YOLO
from utils import read_stub, save_stub
import supervision as sv
import numpy as np
import pandas as pd


class BallTracker:
    """ 
    A class that handles basketball detection and tracking using YOLO.

    This class provides methods to detect the basketball in video frames, process the detections in batches for improved performance and refine the tracking results throughout filtering and interpolation.
    """

    def __init__(self, model_path: str):
        """ 
        Initializes the BallTracker with the specified YOLO model path.

        Args:
            model_path (str): The file path to the YOLO model weights for basketball detection.
        """
        self.model = YOLO(model_path)

    def detect_frames(self, frames):
        """ 
        Detect the basketball in a sequence of video frames using the model while batch processing.

        Args:
            frames (list): A list of video frames, where each frame is a numpy array.

        Returns:
            list: YOLO detection results for each frame.
        """
        batch_size = 20
        detections = []
        for i in range(0, len(frames), batch_size):
            batch_frames = frames[i:i + batch_size]
            batch_detections = self.model.predict(
                batch_frames, conf=0.5, save=False)
            detections.extend(batch_detections)
        return detections

    def get_object_tracks(self, frames, read_from_stub=False, stub_path=None):
        """ 
        Get basketball tracking results for a sequence of video frames with optional caching.

        Args:
            frames (list): A list of video frames, where each frame is a numpy array.
            read_from_stub (bool, optional): Whether to read detections from a cached stub file. Defaults to False.
            stub_path (str, optional): The file path to the cached stub file containing detections. Required if read_from_stub is True.

        Returns:
            list: List of dictionaries containing basketball tracking information for each frame.
        """

        tracks = read_stub(read_from_stub, stub_path)
        if tracks:
            if len(tracks) == len(frames):
                print("Reading from stub file [Ball].")
                return tracks

        detections = self.detect_frames(frames)

        tracks = []

        for frame_num, detection in enumerate(detections):
            cls_names = detection.names

            # Invert the class names dictionary from person: 0 to 0: person
            cls_names_inv = {v: k for k, v in cls_names.items()}

            # Convert the supervision Detection format
            detection_supervision = sv.Detections.from_ultralytics(detection)

            tracks.append({})
            choose_bbox = None
            max_confidence = 0

            for frame_detection in detection_supervision:
                bbox = frame_detection[0].tolist()
                confidence = frame_detection[2]
                cls_id = frame_detection[3]

                if cls_id == cls_names_inv['Ball']:
                    # In case there are multiple ball detections, we choose the one with the highest confidence score
                    if max_confidence < confidence:
                        max_confidence = confidence
                        choose_bbox = bbox

            if choose_bbox:
                tracks[frame_num][1] = {
                    'bbox': choose_bbox,
                    'conf': max_confidence.tolist()
                }

        save_stub(stub_path, tracks)
        return tracks

    def remove_wrong_detections(self, ball_positions):
        """ 
        Filter out wrong basketball detections based on the maximum distance movement allowed between frames, which is adjusted based on the number of frames between detections.

        Args:
            ball_postions (list): A list of detected ball positions across frames.

        Returns:
            list: A list of ball positions with wrong detections removed/filtered out.
        """
        maximum_allowed_distance = 25
        last_good_frame_index = -1
        for i in range(len(ball_positions)):
            current_bbox = ball_positions[i].get(1, {}).get('bbox', [])

            # if that is no detection for the ball in the current frame
            if len(current_bbox) == 0:
                continue

            # First detection of the ball
            if last_good_frame_index == -1:
                last_good_frame_index = i
                continue

            last_good_box = ball_positions[last_good_frame_index].get(
                1, {}).get('bbox', [])
            frame_gap = i - last_good_frame_index
            adjusted_max_distance = maximum_allowed_distance * frame_gap

            # Calculate the distance between the distance between the last good bbox and the current position
            if np.linalg.norm(np.array(np.array(last_good_box[:2]) - np.array(current_bbox[:2]))) > adjusted_max_distance:
                # more that maximum distance ignore it by overwriting the bbox with an empty dict which is empty detection
                ball_positions[i] = {}
            else:
                last_good_frame_index = i

        return ball_positions

    def interpolate_ball_positions(self, ball_positions):
        """ 
        Interpolate missing ball positions across frames to create smooth tracking results.

        Args:
            ball_positions (list): A list of detected ball positions across frames, which may contain missing values.

        Returns:
            list: A list of ball positions with missing values interpolated for smoother tracking (Filling in the gaps).
        """
        ball_positions = [x.get(1, {}).get('bbox', []) for x in ball_positions]

        df_ball_positions = pd.DataFrame(
            ball_positions, columns=["x1", "y1", "x2", "y2"])

        # Interpolate missing values
        df_ball_positions = df_ball_positions.interpolate()
        # Backward fill to handle any remaining NaN values at the beginning
        df_ball_positions = df_ball_positions.bfill()

        ball_positions = [{1: {"bbox": x}}
                          for x in df_ball_positions.to_numpy().tolist()]

        return ball_positions
