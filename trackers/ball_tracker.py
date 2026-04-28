from ultralytics import YOLO
from utils import read_stub, save_stub
import supervision as sv


class BallTracker:
    """ 
    TODO - Docs
    """

    def __init__(self, model_path: str):
        """ 
        TODO - Docs
        """
        self.model = YOLO(model_path)

    def detect_frames(self, frames):
        """ 
        TODO - Docs
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
        TODO - Docs
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
