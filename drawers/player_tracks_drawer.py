from drawers import draw_ellipse, draw_triangle


class PlayerTracksDrawer:
    """ 
    A class responsible for drawing the player tracks and ball possession indicators on the video frames.

    Attributes:
        default_player_team_id (int): The default team ID to assign to a player if their team cannot be determined.
        team_1_colour (list): The BGR colour to use for players assigned to team 1.
        team_2_colour (list): The BGR colour to use for players assigned to team 2.
    """

    def __init__(self, team_1_colour=[255, 245, 238], team_2_colour=[128, 0, 0]):
        """ 
        Initialize the PlayerTracksDrawer instance with the specified team colours.

        Args:
            team_1_colour (list): The BGR colour to use for players assigned to team 1. Defaults to [255, 245, 238].
            team_2_colour (list): The BGR colour to use for players assigned to team 2. Defaults to [128, 0, 0].
        """
        self.default_player_team_id = 1

        self.team_1_colour = team_1_colour
        self.team_2_colour = team_2_colour

    def draw(self, video_frames, tracks, player_assignment, ball_aquisition):
        """ 
        Draws the player tracks and ball possession indicators on each video frame.

        Args:
            video_frames (list): A list of video frames (images) to draw on.
            tracks (list): A list of dictionaries containing the player tracking information for each corresponding frame.
            player_assignment (list): A list of dictionaries indicating the team assignment for each player in the corresponding frame.
            ball_aquisition (list): A list indicating which player has possession of the ball in each corresponding frame.

        Returns:
            list: A list of video frames with the player tracks and ball possession indicators drawn on them.
        """
        output_video_frames = []
        for frame_num, frame in enumerate(video_frames):
            frame = frame.copy()

            player_dict = tracks[frame_num]

            player_assignment_for_frame = player_assignment[frame_num]

            player_id_has_ball = ball_aquisition[frame_num]

            # Draw players tracks
            for track_id, player in player_dict.items():
                team_id = player_assignment_for_frame.get(
                    track_id, self.default_player_team_id)

                if team_id == 1:
                    colour = self.team_1_colour
                else:
                    colour = self.team_2_colour

                if track_id == player_id_has_ball:
                    frame = draw_triangle(
                        frame=frame,
                        bbox=player["bbox"],
                        colour=(0, 0, 255),
                        conf=None
                    )

                frame = draw_ellipse(
                    frame=frame,
                    bbox=player["bbox"],
                    colour=colour,
                    track_id=track_id,
                    conf=player.get("conf", None),
                    team=None
                )

            output_video_frames.append(frame)

        return output_video_frames
