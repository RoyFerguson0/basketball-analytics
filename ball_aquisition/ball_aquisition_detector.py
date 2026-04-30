from utils import measure_distance, get_center_of_bbox


class BallAquisitionDetector:
    """ 
    TODO - Docs
    """

    def __init__(self):
        """ 
        TODO - Docs
        """
        # Maximum distance allowed for someone to have ball possession
        self.possession_threshold = 50
        self.min_frames = 11  # Minimum number of frames to confirm ball possession
        # Minimum containment ratio to confirm ball possession overlap of ball and player bounding boxes
        self.containment_threshold = 0.8

    def get_key_basketball_player_assignment_points(self, player_bbox, ball_center):
        """ 
        TODO - Docs
        """
        ball_center_x = ball_center[0]
        ball_center_y = ball_center[1]

        x1, y1, x2, y2 = player_bbox
        width = x2 - x1
        height = y2 - y1

        output_points = []

        # Check if the ball center is within the vertical bounds of the player bounding box
        if ball_center_y > y1 and ball_center_y < y2:
            output_points.append((x1, ball_center_y))
            output_points.append((x2, ball_center_y))

        # Check if the ball center is within the horizontal bounds of the player bounding box
        if ball_center_x > x1 and ball_center_x < x2:
            output_points.append((ball_center_x, y1))
            output_points.append((ball_center_x, y2))

        output_points += [
            (x1, y1),  # Top-left corner
            (x2, y1),  # Top-right corner
            (x1, y2),  # Bottom-left corner
            (x2, y2),  # Bottom-right corner
            (x1 + width // 2, y1),  # Top Center
            (x1 + width // 2, y2),   # Bottom Center
            (x1, y1 + height // 2),  # Left Center
            (x2, y1 + height // 2)   # Right Center
        ]
        return output_points

    def find_minimum_distance_to_ball(self, ball_center, player_bbox):
        """ 
        TODO - Docs
        """
        key_points = self.get_key_basketball_player_assignment_points(
            player_bbox, ball_center)

        return min(measure_distance(ball_center, key_point) for key_point in key_points)

    def calculate_ball_containment_ratio(self, player_bbox, ball_bbox):
        """ 
        TODO - Docs
        """
        px1, py1, px2, py2 = player_bbox
        bx1, by1, bx2, by2 = ball_bbox

        ball_area = (bx2 - bx1) * (by2 - by1)
        player_area = (px2 - px1) * (py2 - py1)

        intersection_x1 = max(px1, bx1)
        intersection_y1 = max(py1, by1)
        intersection_x2 = min(px2, bx2)
        intersection_y2 = min(py2, by2)

        if intersection_x2 < intersection_x1 or intersection_y2 < intersection_y1:
            return 0  # No overlap

        intersection_area = (intersection_x2 - intersection_x1) * \
            (intersection_y2 - intersection_y1)
        ball_area = (bx2 - bx1) * (by2 - by1)

        return intersection_area / ball_area if ball_area > 0 else 0

    def find_best_candiate_for_possession(self, ball_center, player_tracks, ball_bbox):
        """ 
        TODO - Docs
        """
        high_containment_players = []  # List of players with high containment ratios
        # List of players with regular distance but within possession threshold
        regular_distance_players = []

        for player_id, player_info in player_tracks.items():
            # Get player bounding box
            player_bbox = player_info.get("bbox", [])
            if not player_bbox:
                continue

            # Calculate containment ratio and minimum distance to the ball for the current player
            containment = self.calculate_ball_containment_ratio(
                player_bbox, ball_bbox)
            min_distance = self.find_minimum_distance_to_ball(
                ball_center, player_bbox)

            # Classify players based on containment ratio and distance to the ball
            if containment > self.containment_threshold:
                high_containment_players.append((player_id, min_distance))
            else:
                regular_distance_players.append((player_id, min_distance))

        # First priority: Players with high containment ratios
        if high_containment_players:
            # Player with highest containment
            best_candiate = max(high_containment_players, key=lambda x: x[1])
            return best_candiate[0]

        # Second priority: Players with the smallest distance to the ball within the possession threshold
        if regular_distance_players:
            best_candiate = min(regular_distance_players, key=lambda x: x[1])
            if best_candiate[1] < self.possession_threshold:
                return best_candiate[0]

        return -1  # No suitable candidate found

    def detect_ball_possession(self, player_tracks, ball_tracks):
        """ 
        TODO - Docs
        """
        num_frames = len(ball_tracks)
        # Initialize with -1 indicating no possession
        possession_list = [-1] * num_frames
        # Dictionary to track consecutive possession counts for each player
        consecutive_possession_count = {}

        for frame_num in range(num_frames):
            # Get ball information for the current frame
            ball_info = ball_tracks[frame_num].get(1, {})
            if not ball_info:
                continue

            # Get ball bounding box and calculate its center
            ball_bbox = ball_info.get("bbox", [])
            if not ball_bbox:
                continue

            ball_center = get_center_of_bbox(ball_bbox)

            # Find the best candidate for ball possession in the current frame
            best_player_id = self.find_best_candiate_for_possession(
                ball_center=ball_center,
                player_tracks=player_tracks[frame_num],
                ball_bbox=ball_bbox
            )

            # Update possession list and consecutive counts
            if best_player_id != -1:
                number_of_consecutive_frames = consecutive_possession_count.get(
                    best_player_id, 0) + 1
                consecutive_possession_count[best_player_id] = number_of_consecutive_frames

                if consecutive_possession_count[best_player_id] >= self.min_frames:
                    possession_list[frame_num] = best_player_id
            else:
                # Reset counts for all players if no one has possession in this frame
                consecutive_possession_count = {}

        return possession_list
