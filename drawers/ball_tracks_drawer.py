from drawers import draw_triangle


class BallTracksDrawer:
    """ 
    A drawer class that is responsible for drawing the ball tracks on the video frames.

    Attributes:
        ball_pointer_colour (tuple): The colour of the pointer that indicates the ball position. (BGR format)
    """

    def __init__(self, ball_pointer_colour=(0, 255, 0)):
        """ 
        Initialize the BallTracksDrawer instance with the specified ball pointer colour.

        Args:
            ball_pointer_colour (tuple): The colour of the pointer that indicates the ball position. (BGR format)
        """
        self.ball_pointer_colour = ball_pointer_colour

    def draw(self, video_frames, tracks):
        """ 
        Draws the ball pointers on each video frame based on the provided tracks information.

        Args:
            video_frames (list): A list of video frames (images) to draw on.
            tracks (list): A list of dictionaries containing the ball tracking information for each frame.

        Returns:
            list: A list of video frames with the ball tracks pointers drawn on them.
        """

        output_video_frames = []
        for frame_num, frame in enumerate(video_frames):
            frame = frame.copy()

            ball_dict = tracks[frame_num]

            # Draw ball tracks
            for track_id, ball in ball_dict.items():
                if ball["bbox"] is None:
                    continue

                frame = draw_triangle(
                    frame=frame,
                    bbox=ball["bbox"],
                    colour=self.ball_pointer_colour,
                    conf=ball.get("conf", None)
                )

            output_video_frames.append(frame)

        return output_video_frames
