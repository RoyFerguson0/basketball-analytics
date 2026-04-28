from drawers import draw_triangle


class BallTracksDrawer:
    """ 
    TODO - Docs
    """

    def __init__(self):
        """ 
        TODO - Docs
        """
        self.ball_pointer_colour = (0, 255, 0)

    def draw(self, video_frames, tracks):
        """ 
        TODO - Docs
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
