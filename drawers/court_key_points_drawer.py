import supervision as sv


class CourtKeypointDrawer:
    def __init__(self):
        self.keypoint_colour = "#ff2c2c"

    def draw(self, frames, court_keypoints):
        # sv.VertexAnnotator is used to draw the keypoints on the frames
        vertex_annotator = sv.VertexAnnotator(
            color=sv.Color.from_hex(self.keypoint_colour),
            radius=8
        )
        # sv.VertexLabelAnnotator is used to draw the labels of the keypoints on the frames
        vertex_label_annotator = sv.VertexLabelAnnotator(
            color=sv.Color.from_hex(self.keypoint_colour),
            text_color=sv.Color.WHITE,
            text_scale=0.5,
            text_thickness=1
        )

        output_frames = []
        for index, frame in enumerate(frames):
            annotated_frame = frame.copy()

            keypoints = court_keypoints[index]
            annotated_frame = vertex_annotator.annotate(
                scene=annotated_frame,
                key_points=keypoints
            )

            keypoints_numpy = keypoints.cpu().numpy()
            annotated_frame = vertex_label_annotator.annotate(
                scene=annotated_frame,
                key_points=keypoints_numpy,
            )
            output_frames.append(annotated_frame)

        return output_frames
