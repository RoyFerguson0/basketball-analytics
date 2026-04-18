import torch
from ultralytics import YOLO
from utils import read_video, save_video


def main():

    # Read video
    video_frames = read_video("input_videos/video_1.mp4")

    # Save Video
    save_video(video_frames, "output_videos/video_1_output.mp4")


if __name__ == "__main__":
    main()
