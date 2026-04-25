import os

import torch
from ultralytics import YOLO
from utils import read_video, save_video
from trackers import PlayerTracker


def main():

    # Read video
    video_frames = read_video("input_videos/video_1.mp4")

    # Initialise Tracker
    player_tracker = PlayerTracker("models/detector.pt")

    # Run Detectors
    player_tracks = player_tracker.get_object_tracks(
        frames=video_frames,
        read_from_stub=True,
        stub_path=os.path.join("stubs", "player_tracks.pkl")
    )

    print(player_tracks)

    # Save Video
    save_video(video_frames, "output_videos/video_1_output.mp4")


if __name__ == "__main__":
    main()
