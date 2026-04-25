from ultralytics import YOLO
import supervision as sv
from utils import read_stub, save_stub


class PlayerTracker:
    """ 
    A class that handles player detection and tracking using YOLO and ByteTrack.

    This class combines YOLO for object detection with ByteTrack to track and maintain identities of players across frames while also processing the detections in batches for improved performance.
    """

    def __init__(self, model_path: str):
        """ 
        Initialize the player tracker with a specified YOLO model and a ByteTrack tracker.

        Args:
            model_path (str): The file path to the YOLO model weights.
        """
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack()

    def detect_frames(self, frames):
        """ 
        Detect players is a sequence of video frames using the YOLO model while batch processing.

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
        Get player tracking results for a sequence of video frames with optional caching.

        Args:
            frames (list): A list of video frames, where each frame is a numpy array.

            read_from_stub (bool, optional): Whether to read detections from a cached stub file. Defaults to False.

            stub_path (str, optional): The file path to the cached stub file containing detections. Required if read_from_stub is True.

        Returns:
            list: List of dictionaries containing player tracking information for each frame. Each dictionary maps player IDs to their corresponding bounding box coordinates and confidence scores.
        """

        tracks = read_stub(read_from_stub, stub_path)
        if tracks:
            if len(tracks) != len(frames):
                return tracks

        detections = self.detect_frames(frames)

        tracks = []

        for frame_num, detection in enumerate(detections):
            cls_names = detection.names

            # Invert the class names dictionary from person: 0 to 0: person
            cls_names_inv = {v: k for k, v in cls_names.items()}

            # Convert the supervision Detection format
            detection_supervision = sv.Detections.from_ultralytics(detection)

            # Track Objects
            detection_with_tracks = self.tracker.update_with_detections(
                detections=detection_supervision)

            tracks.append({})

            for frame_detection in detection_with_tracks:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]
                track_id = frame_detection[4]

                if cls_id == cls_names_inv['Player']:
                    tracks[frame_num][track_id] = {
                        'bbox': bbox,
                        'conf': frame_detection[2].tolist()
                    }

        save_stub(stub_path, tracks)
        return tracks
