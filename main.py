import os

import torch
from ultralytics import YOLO
from utils import read_video, save_video
from trackers import PlayerTracker, BallTracker
from drawers import PlayerTracksDrawer, BallTracksDrawer


def main():

    # Read video
    video_frames = read_video("input_videos/video_1.mp4")

    # Initialise Tracker
    player_tracker = PlayerTracker("models/detector.pt")
    ball_tracker = BallTracker("models/detector.pt")

    # Run Detectors
    player_tracks = player_tracker.get_object_tracks(
        frames=video_frames,
        read_from_stub=True,
        stub_path="stubs/player_track_stubs.pkl"
    )

    ball_tracks = ball_tracker.get_object_tracks(
        frames=video_frames,
        read_from_stub=True,
        stub_path="stubs/ball_track_stubs.pkl"
    )

    # Draw Output
    # Initialise Drawers
    player_tracks_drawer = PlayerTracksDrawer()
    ball_tracks_drawer = BallTracksDrawer()

    # Draw Players Tracks
    output_video_frames = player_tracks_drawer.draw(
        video_frames=video_frames,
        tracks=player_tracks
    )

    # Draw Ball Tracks
    output_video_frames = ball_tracks_drawer.draw(
        video_frames=output_video_frames,
        tracks=ball_tracks
    )

    # Save Video
    save_video(output_video_frames, "output_videos/video_1_output.mp4")


if __name__ == "__main__":
    main()
