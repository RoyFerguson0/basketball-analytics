from drawers import draw_ellipse


class PlayerTracksDrawer:
    """ 
    TODO - Docs
    """

    def __init__(self, team_1_colour=[255, 245, 238], team_2_colour=[128, 0, 0]):
        """ 
        TODO - Docs
        """
        self.default_player_team_id = 1

        self.team_1_colour = team_1_colour
        self.team_2_colour = team_2_colour

    def draw(self, video_frames, tracks, player_assignment):
        """ 
        TODO - Docs
        """
        output_video_frames = []
        for frame_num, frame in enumerate(video_frames):
            frame = frame.copy()

            player_dict = tracks[frame_num]

            player_assignment_for_frame = player_assignment[frame_num]

            # Draw players tracks
            for track_id, player in player_dict.items():
                team_id = player_assignment_for_frame.get(
                    track_id, self.default_player_team_id)

                if team_id == 1:
                    colour = self.team_1_colour
                else:
                    colour = self.team_2_colour

                frame = draw_ellipse(
                    frame=frame,
                    bbox=player["bbox"],
                    colour=colour,
                    track_id=track_id,
                    conf=player.get("conf", None)
                )

            output_video_frames.append(frame)

        return output_video_frames
