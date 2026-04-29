import cv2
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from utils import read_stub, save_stub


class TeamAssigner:
    """
    A class that assigns players to teams based on the colour of their jersey colours using visual analysis.

    The class uses a pre-trained vision model to classify players into teams based on their appearance in the video frames. Maintains a consistent team assignment for each player across frames.

    Attributes:
        player_team_dict (dict): A dictionary mapping player IDs to their assigned team IDs.
        team_1_class_name (str): The class name representing the first team's jersey colour.
        team_2_class_name (str): The class name representing the second team's jersey colour.
    """

    def __init__(self, team_1_class_name="white shirt", team_2_class_name="dark blue shirt"):
        """
        Initializes that TeamAssigner with the specified team jersey descriptions.
        """
        self.player_team_dict = {}

        self.team_1_class_name = team_1_class_name
        self.team_2_class_name = team_2_class_name

    def load_model(self):
        """
        Loads the pre-trained vision model and processor for classifying player jersey colours.
        """
        self.model = CLIPModel.from_pretrained("patrickjohncyh/fashion-clip")
        self.processor = CLIPProcessor.from_pretrained(
            "patrickjohncyh/fashion-clip")

    def get_player_colour(self, frame, bbox):
        """
        Analyses the players jersey colour within the specified bounding box in the given video frame and classifies it as either team 1 or team 2 based on the pre-trained model's predictions.

        Args:
            frame (numpy.ndarray): The video frame containing the player.
            bbox (tuple): The bounding box coordinates (x1, y1, x2, y2) for the player in the frame.

        Returns:
            str: The classified team jersey colour/description.
        """
        image = frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]

        # Convert bgr to rgb to pil image
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_image)

        classes = [self.team_1_class_name, self.team_2_class_name]

        inputs = self.processor(
            text=classes,
            images=pil_image,
            return_tensors="pt",
            padding=True
        )

        outputs = self.model(**inputs)
        logits_per_image = outputs.logits_per_image
        probs = logits_per_image.softmax(dim=1)

        class_name = classes[probs.argmax(dim=1)[0]]
        return class_name

    def get_player_team(self, frame, player_bbox, player_id):
        """
        Get the team assignment for a player, using the cached results if available.

        Args:
            frame (numpy.ndarray): The video frame containing the player.
            player_bbox (tuple): The bounding box coordinates (x1, y1, x2, y2) for the player in the frame.
            player_id (int): The Unique Identifier of the player.

        Returns:
            int: The assigned team ID (1 or 2).
        """

        if player_id in self.player_team_dict:
            return self.player_team_dict[player_id]

        player_colour = self.get_player_colour(frame, player_bbox)

        team_id = 2
        if player_colour == self.team_1_class_name:
            team_id = 1

        self.player_team_dict[player_id] = team_id

        return team_id

    def get_player_teams_across_frames(self, video_frames, player_tracks, read_from_stub=False, stub_path=None):
        """
        Processes all video frames and assigns teams to players across frames, with optional caching.

        Args:
            video_frames (list): List of video frames to process.
            player_tracks (list): List of dictionaries containing player tracking information for each frame.
            read_from_stub (bool, optional): Whether to read cached team assignments from a stub file.
            stub_path (str, optional): The file path for the stub file to read from or write to. 

        Returns:
            list: A list of dictionaries mapping player IDs to team IDs for each frame.
        """

        player_assignment = read_stub(
            read_from_stub=read_from_stub, stub_path=stub_path)
        if player_assignment:
            if len(player_assignment) == len(video_frames):
                return player_assignment

        self.load_model()

        player_assignment = []

        for frame_num, player_track in enumerate(player_tracks):
            player_assignment.append({})

            if frame_num % 50 == 0:
                self.player_team_dict = {}

            for player_id, track in player_track.items():
                team = self.get_player_team(
                    video_frames[frame_num], track["bbox"], player_id)

                player_assignment[frame_num][player_id] = team

        save_stub(stub_path, player_assignment)
        return player_assignment
