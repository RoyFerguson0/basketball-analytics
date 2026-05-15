from copy import deepcopy
from utils import measure_distance, get_foot_position
import numpy as np
import cv2
from .homography import Homography


class TacticalViewConverter:
    """
    Converts player and keypoint positions from a broadcast camera view to a tactical (top-down) basketball court view.

    This class uses homography transformations to map detected court keypoints and player positions from the camera frame to a standardized tactical view, enabling consistent spatial analysis regardless of camera angle or orientation.

    Key features:
    - Validates detected keypoints for geometric consistency.
    - Handles mirrored court mappings to resolve left/right ambiguity.
    - Maintains temporal stability in mapping mode selection.
    - Transforms player positions to tactical coordinates for downstream analytics.
    """

    def __init__(self, court_image_path):
        self.court_image_path = court_image_path

        # The dimensions of the tactical court image (pixels)
        self.width = 300
        self.height = 161

        # The real-world dimensions of a standard basketball court (meters)
        self.actual_width_in_meters = 28
        self.actual_height_in_meters = 15

        # Key parameters for validation and stability
        # Minimum confidence for a keypoint to be considered reliable
        self.min_keypoint_confidence = 0.45
        # Minimum number of valid keypoints required to compute a homography
        self.min_keypoints_for_homography = 4
        # Minimum spread (in pixels) required among detected keypoints to avoid degenerate homographies
        self.min_keypoint_spread_x = 40
        self.min_keypoint_spread_y = 25
        # Margin for preferring temporal stability in mapping mode selection
        self.mapping_stability_margin = 1.0
        # Penalty added to the score if the detected side (left/right) does not match the tactical side
        self.side_mismatch_penalty = 2.0

        # Mirrored index map: maps each keypoint index to its horizontally flipped counterpart.
        # This is needed because the camera view may be mirrored (left/right ambiguity),
        # so we must test both the identity and mirrored mappings to find the best fit.
        self.mirrored_index_map = {
            0: 15, 1: 14, 2: 13, 3: 12, 4: 11, 5: 10,
            6: 6, 7: 7,
            8: 16, 9: 17,
            10: 5, 11: 4, 12: 3, 13: 2, 14: 1, 15: 0,
            16: 8, 17: 9,
        }

        # Tracks the last successful mapping mode ('identity' or 'mirrored') to ensure consistency across frames.
        # This helps avoid flickering between modes when both are similarly good.
        self.last_mapping_mode = "identity"  # 'identity' or 'mirrored'

        # Define the reference keypoints on the tactical court image (in pixels),
        # corresponding to known real-world locations (corners, free throw lines, etc).
        # These are used as the target points for homography estimation.
        self.key_points = [
            # left edge (from top to bottom)
            (0, 0),
            (0, int((0.91/self.actual_height_in_meters)*self.height)),
            (0, int((5.18/self.actual_height_in_meters)*self.height)),
            (0, int((10/self.actual_height_in_meters)*self.height)),
            (0, int((14.1/self.actual_height_in_meters)*self.height)),
            (0, int(self.height)),

            # Middle line (bottom and top)
            (int(self.width/2), self.height),
            (int(self.width/2), 0),

            # Left Free throw line (bottom and top)
            (int((5.79/self.actual_width_in_meters)*self.width),
             int((5.18/self.actual_height_in_meters)*self.height)),
            (int((5.79/self.actual_width_in_meters)*self.width),
             int((10/self.actual_height_in_meters)*self.height)),

            # right edge (from bottom to top)
            (self.width, int(self.height)),
            (self.width, int((14.1/self.actual_height_in_meters)*self.height)),
            (self.width, int((10/self.actual_height_in_meters)*self.height)),
            (self.width, int((5.18/self.actual_height_in_meters)*self.height)),
            (self.width, int((0.91/self.actual_height_in_meters)*self.height)),
            (self.width, 0),

            # Right Free throw line (bottom and top)
            (int(((self.actual_width_in_meters-5.79)/self.actual_width_in_meters)
                 * self.width), int((5.18/self.actual_height_in_meters)*self.height)),
            (int(((self.actual_width_in_meters-5.79)/self.actual_width_in_meters)
                 * self.width), int((10/self.actual_height_in_meters)*self.height)),
        ]

    def validate_keypoints(self, keypoints_list):
        """
        Remove geometrically inconsistent keypoints from each frame's detection.

        For each detected keypoint, compare the ratio of distances to two other keypoints
        with the expected ratio from the tactical court. If the error is too large,
        mark the keypoint as invalid (set to zero).

        Args:
            keypoints_list: List of keypoint detection objects for each frame.
        Returns:
            List of keypoint detection objects with invalid keypoints zeroed out.
        """
        keypoints_list = deepcopy(keypoints_list)
        for frame_idx, frame_keypoints in enumerate(keypoints_list):
            frame_keypoints = frame_keypoints.xy.tolist()[0]

            # Find indices of detected (nonzero) keypoints
            detected_indicies = [i for i, kp in enumerate(
                frame_keypoints) if kp[0] > 0 and kp[1] > 0]

            if len(detected_indicies) < 3:
                continue

            invalid_keypoints = []

            for i in detected_indicies:
                # skip keypoints (0, 0) which are not detected
                if frame_keypoints[i][0] == 0 and frame_keypoints[i][1] == 0:
                    continue

                # Pick two other detected keypoints for geometric ratio check
                other_indicies = [
                    idx for idx in detected_indicies if idx != i and idx not in invalid_keypoints]

                if len(other_indicies) < 2:
                    continue

                j, k = other_indicies[:2]

                # Compute distances in detected and tactical space
                detected_ij = measure_distance(
                    frame_keypoints[i], frame_keypoints[j])
                detected_ik = measure_distance(
                    frame_keypoints[i], frame_keypoints[k])

                tactical_ij = measure_distance(
                    self.key_points[i], self.key_points[j])
                tactical_ik = measure_distance(
                    self.key_points[i], self.key_points[k])

                # Compare the ratio of distances to expected tactical ratio
                if tactical_ij > 0 or tactical_ik > 0:
                    proportion_detected = detected_ij / \
                        detected_ik if detected_ik > 0 else float('inf')
                    proportion_tactical = tactical_ij / \
                        tactical_ik if tactical_ik > 0 else float('inf')

                    error = (proportion_detected -
                             proportion_tactical) / proportion_tactical
                    error = abs(error)

                    # If error is too large, mark as invalid
                    if error > 0.8:
                        keypoints_list[frame_idx].xy[0][i] *= 0
                        keypoints_list[frame_idx].xyn[0][i] *= 0
                        invalid_keypoints.append(i)

        return keypoints_list

    def extract_reliable_keypoints(self, frame_keypoints):
        """
        Select the most reliable detection in a frame and return its keypoints and indices of confident keypoints.

        Args:
            frame_keypoints: Detection object for a single frame (may contain multiple detections).
        Returns:
            keypoints_xy: (N, 2) array of keypoint coordinates for the best detection.
            valid_indices: List of indices for keypoints with sufficient confidence and nonzero location.
        """
        if frame_keypoints is None or frame_keypoints.xy.shape[0] == 0:
            return None, []

        all_xy = frame_keypoints.xy.cpu().numpy()
        all_conf = frame_keypoints.conf.cpu().numpy(
        ) if frame_keypoints.conf is not None else None

        if all_xy.shape[0] == 0:
            return None, []

        if all_conf is not None and all_conf.shape[0] > 0:
            # Choose detection with highest average keypoint confidence
            detection_idx = int(np.argmax(np.mean(all_conf, axis=1)))
            keypoints_xy = all_xy[detection_idx]
            keypoints_conf = all_conf[detection_idx]
            valid_indices = [i for i, (kp, conf) in enumerate(zip(
                keypoints_xy, keypoints_conf)) if conf >= self.min_keypoint_confidence and kp[0] > 0 and kp[1] > 0]
        else:
            keypoints_xy = all_xy[0]
            valid_indices = [i for i, kp in enumerate(
                keypoints_xy) if kp[0] > 0 and kp[1] > 0]

        return keypoints_xy, valid_indices

    def map_indices_for_mode(self, valid_indices, mode):
        """
        Map keypoint indices according to the current mode.
        If mode is 'mirrored', use the mirrored index map to flip indices horizontally.
        """
        if mode == "identity":
            return valid_indices
        return [self.mirrored_index_map.get(i, i) for i in valid_indices]

    def build_homography_for_mode(self, detected_keypoints, valid_indices, mode):
        """
        Attempt to build a homography for the given mode (identity or mirrored).
        Returns None if the keypoints are too clustered (degenerate case).
        Returns (homography, reprojection_error, mode) on success.
        """
        # Gather source points from detected keypoints
        source_points = np.array(
            [detected_keypoints[i] for i in valid_indices], dtype=np.float32
        )

        # Require sufficient spread in detected keypoints to avoid unstable homographies
        if (np.ptp(source_points[:, 0]) < self.min_keypoint_spread_x or
                np.ptp(source_points[:, 1]) < self.min_keypoint_spread_y):
            return None

        # Map indices for the current mode (identity or mirrored)
        mapped_indices = self.map_indices_for_mode(valid_indices, mode)
        target_points = np.array(
            [self.key_points[i] for i in mapped_indices], dtype=np.float32
        )

        try:
            # Fit homography and compute reprojection error
            homography = Homography(source_points, target_points)
            reprojection = homography.transform_points(source_points)
            reprojection_error = float(
                np.mean(np.linalg.norm(reprojection - target_points, axis=1)))
            return homography, reprojection_error, mode
        except (ValueError, cv2.error):
            return None

    def get_side_consistency_penalty(self, detected_keypoints, valid_indices, mode, frame_width):
        """
        Penalize mappings where the detected side (left/right) does not match the tactical side.
        This helps avoid mirrored mappings when the camera view is not actually mirrored.
        """
        if frame_width is None or frame_width <= 0:
            return 0.0

        source_points = np.array(
            [detected_keypoints[i] for i in valid_indices], dtype=np.float32
        )

        mapped_indices = self.map_indices_for_mode(valid_indices, mode)
        target_points = np.array(
            [self.key_points[i] for i in mapped_indices], dtype=np.float32
        )

        # Determine if the detected and tactical points are on the right side
        source_is_right = float(
            np.mean(source_points[:, 0])) > (frame_width / 2.0)
        target_is_right = float(
            np.mean(target_points[:, 0])) > (self.width / 2.0)

        # If the sides do not match, add a penalty
        if source_is_right != target_is_right:
            return self.side_mismatch_penalty
        return 0.0

    def select_best_homography(self, detected_keypoints, valid_indices, frame_width):
        """
        Try both mapping modes (identity and mirrored) and select the best homography.
        Uses reprojection error and side consistency penalty to score candidates.
        Prefers the previous mode if both are similarly good (temporal stability).
        Returns (homography, reprojection_error, selected_mode) or None if no valid mapping.
        """
        # Try the last used mode first for temporal stability
        modes = [self.last_mapping_mode]
        if self.last_mapping_mode != "identity":
            modes.append("identity")
        if self.last_mapping_mode != "mirrored":
            modes.append("mirrored")

        candidates = []
        for mode in modes:
            result = self.build_homography_for_mode(
                detected_keypoints, valid_indices, mode
            )
            if result is not None:
                homography, reprojection_error, _ = result
                total_score = reprojection_error + self.get_side_consistency_penalty(
                    detected_keypoints,
                    valid_indices,
                    mode,
                    frame_width
                )
                candidates.append(
                    (homography, total_score, mode, reprojection_error)
                )

        if not candidates:
            return None

        if len(candidates) == 1:
            homography, _, mode, reproj_error = candidates[0]
            return homography, reproj_error, mode

        # Sort by total score (lower is better)
        candidates.sort(key=lambda x: x[1])
        best, second = candidates[0], candidates[1]

        # Prefer previous mode if both are similarly good (within margin)
        if (second[1] - best[1]) < self.mapping_stability_margin:
            for candidate in candidates:
                if candidate[2] == self.last_mapping_mode:
                    return candidate[0], candidate[3], candidate[2]

        return best[0], best[3], best[2]

    def transform_players_to_tactical_view(self, keypoints_list, player_tracks):
        """
        For each frame, transform all player positions from camera coordinates to tactical view coordinates.
        Uses the best available homography for each frame, and skips frames with insufficient keypoints.

        Args:
            keypoints_list: List of keypoint detection objects for each frame.
            player_tracks: List of dicts mapping player_id to player data (with bbox) for each frame.
        Returns:
            List of dicts mapping player_id to tactical view coordinates for each frame.
        """
        tactical_player_positions = []

        for frame_idx, (frame_keypoints, frame_tracks) in enumerate(zip(keypoints_list, player_tracks)):
            # Initialize empty dictionary for this frame
            tactical_positions = {}

            # Extract reliable keypoints for this frame
            detected_keypoints, valid_indices = self.extract_reliable_keypoints(
                frame_keypoints)

            # Need at least 4 valid keypoints for a reliable homography
            if len(valid_indices) < self.min_keypoints_for_homography:
                tactical_player_positions.append(tactical_positions)
                continue

            # Get frame width if available (for side consistency check)
            frame_width = None
            if frame_keypoints is not None and hasattr(frame_keypoints, "orig_shape"):
                frame_width = frame_keypoints.orig_shape[1]

            # Select best homography for this frame
            selected = self.select_best_homography(
                detected_keypoints, valid_indices, frame_width)

            if selected is None:
                tactical_player_positions.append(tactical_positions)
                continue

            homography, _, selected_mode = selected
            self.last_mapping_mode = selected_mode  # Update for temporal stability

            try:
                # Transform each player's position (use foot position from bbox)
                for player_id, player_data in frame_tracks.items():
                    bbox = player_data["bbox"]
                    player_position = np.array([
                        get_foot_position(bbox)])

                    # Transform to tactical view coordinates
                    tactical_position = homography.transform_points(
                        player_position)

                    # If tactical position is not in the tactical view, skip
                    if tactical_position[0][0] < 0 or tactical_position[0][0] > self.width or tactical_position[0][1] < 0 or tactical_position[0][1] > self.height:
                        continue

                    tactical_positions[player_id] = tactical_position[0].tolist(
                    )

            except (ValueError, cv2.error) as e:
                # If transformation fails, skip this frame
                pass

            tactical_player_positions.append(tactical_positions)

        return tactical_player_positions
