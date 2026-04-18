""" 
A module for reading and writing video files.

This module provides utility functions to load video frames into memory and save processed frames back to a video files, with support for common video formats.
"""

import cv2
import os


def read_video(video_path):
    """ 
    Read all frames from a video file into memory.

    Args:
        video_path (str): The path to the video file.

    Returns:
        list: A list of frames read from the video, where each frame is a numpy array.
    """
    cap = cv2.VideoCapture(video_path)
    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames


def save_video(output_video_frames, output_video_path, fps=24):
    """ 
    Save a sequence of frames to a video file.

    Creates necessary directories if they do not exist and writes frames using mp4v codec.

    Args:
        output_video_frames (list): A list of frames to be saved, where each frame is a numpy array.
        output_video_path (str): The path where the output video will be saved.
        fps (int, optional): Frames per second for the output video. Defaults to 24.
    """
    if not output_video_frames:
        print("No frames to save.")
        return

    if not os.path.exists(os.path.dirname(output_video_path)):
        os.makedirs(os.path.dirname(output_video_path))

    # First frame dimensions (height, width, channels)
    height, width, _ = output_video_frames[0].shape

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    for frame in output_video_frames:
        out.write(frame)

    out.release()
