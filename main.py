import os

import torch
from ultralytics import YOLO
from utils import read_video, save_video
from trackers import PlayerTracker, BallTracker
from drawers import PlayerTracksDrawer, BallTracksDrawer
from team_assigner import TeamAssigner


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

    # Remove wrong ball detections
    ball_tracks = ball_tracker.remove_wrong_detections(ball_tracks)

    # Interpolate ball tracks
    ball_tracks = ball_tracker.interpolate_ball_positions(ball_tracks)

    # Assign Player Teams
    team_assigner = TeamAssigner()
    player_assignment = team_assigner.get_player_teams_across_frames(
        video_frames=video_frames,
        player_tracks=player_tracks,
        read_from_stub=True,
        stub_path="stubs/player_assignment_stubs.pkl"
    )

    # Draw Output
    # Initialise Drawers
    player_tracks_drawer = PlayerTracksDrawer()
    ball_tracks_drawer = BallTracksDrawer()

    # Draw Players Tracks
    output_video_frames = player_tracks_drawer.draw(
        video_frames=video_frames,
        tracks=player_tracks,
        player_assignment=player_assignment
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
