import os

import torch
from ultralytics import YOLO
from court_keypoint_detector import CourtKeypointDetector
from utils import read_video, save_video
from trackers import PlayerTracker, BallTracker
from drawers import PlayerTracksDrawer, BallTracksDrawer, CourtKeypointDrawer, TacticalViewDrawer
from team_assigner import TeamAssigner
from ball_aquisition import BallAquisitionDetector
from tactical_view_converter import TacticalViewConverter


def main():

    # Read video
    video_frames = read_video("input_videos/video_3.mp4")

    # Initialise Tracker
    player_tracker = PlayerTracker("models/detector.pt")
    ball_tracker = BallTracker("models/detector.pt")
    court_keypoint_detect = CourtKeypointDetector(
        "models/court_keypoint_detector.pt")

    # Run Detectors
    player_tracks = player_tracker.get_object_tracks(
        frames=video_frames,
        read_from_stub=False,
        stub_path="stubs/player_track_stubs.pkl"
    )

    ball_tracks = ball_tracker.get_object_tracks(
        frames=video_frames,
        read_from_stub=False,
        stub_path="stubs/ball_track_stubs.pkl"
    )

    court_keypoints = court_keypoint_detect.get_court_keypoints(
        frames=video_frames,
        read_from_stub=False,
        stub_path="stubs/court_keypoint_stubs.pkl"
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
        read_from_stub=False,
        stub_path="stubs/player_assignment_stubs.pkl"
    )

    # Ball Aquisition
    ball_aquisition_detector = BallAquisitionDetector()
    ball_aquisition = ball_aquisition_detector.detect_ball_possession(
        player_tracks=player_tracks,
        ball_tracks=ball_tracks
    )

    # Tactical View Converter
    tactical_view_converter = TacticalViewConverter(
        court_image_path="./images/basketball_court.png")
    court_keypoints = tactical_view_converter.validate(
        court_keypoints)

    # Draw Output
    # Initialise Drawers
    player_tracks_drawer = PlayerTracksDrawer()
    ball_tracks_drawer = BallTracksDrawer()
    court_keypoint_drawer = CourtKeypointDrawer()
    tactical_view_drawer = TacticalViewDrawer()

    # Draw Players Tracks
    output_video_frames = player_tracks_drawer.draw(
        video_frames=video_frames,
        tracks=player_tracks,
        player_assignment=player_assignment,
        ball_aquisition=ball_aquisition
    )

    # Draw Ball Tracks
    output_video_frames = ball_tracks_drawer.draw(
        video_frames=output_video_frames,
        tracks=ball_tracks
    )

    # Draw Court Keypoints
    output_video_frames = court_keypoint_drawer.draw(
        frames=output_video_frames,
        court_keypoints=court_keypoints
    )

    # Draw Tactical View
    output_video_frames = tactical_view_drawer.draw(
        video_frames=output_video_frames,
        court_image_path=tactical_view_converter.court_image_path,
        width=tactical_view_converter.width,
        height=tactical_view_converter.height,
        tactical_court_keypoints=tactical_view_converter.key_points
    )

    # Save Video
    save_video(output_video_frames, "output_videos/video_output.mp4")


if __name__ == "__main__":
    main()
