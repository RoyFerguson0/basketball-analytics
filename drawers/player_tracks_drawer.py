from drawers import draw_ellipse


class PlayerTracksDrawer:
    """ 
    TODO - Docs
    """

    def __init__(self):
        """ 
        TODO - Docs
        """
        pass

    def draw(self, video_frames, tracks):
        """ 
        TODO - Docs
        """
        output_video_frames = []
        for frame_num, frame in enumerate(video_frames):
            frame = frame.copy()

            player_dict = tracks[frame_num]

            # Draw players tracks
            for track_id, player in player_dict.items():

                frame = draw_ellipse(
                    frame=frame,
                    bbox=player["bbox"],
                    colour=(0, 0, 255),
                    track_id=track_id,
                    conf=player.get("conf", None)
                )

            output_video_frames.append(frame)

        return output_video_frames
