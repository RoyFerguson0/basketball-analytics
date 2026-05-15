import cv2


class TacticalViewDrawer:
    def __init__(self):
        self.start_x = 20
        self.start_y = 40

    def draw(self, video_frames, court_image_path, width, height, tactical_court_keypoints,):
        court_image = cv2.imread(court_image_path)
        court_image = cv2.resize(court_image, (width, height))

        output_video_frames = []
        for frame_idx, frame in enumerate(video_frames):
            frame = frame.copy()

            x1 = self.start_x
            y1 = self.start_y
            x2 = x1 + width
            y2 = y1 + height

            alpha = 0.6

            # Overlay the court image onto the video frame with transparency
            overlay = frame[y1: y2, x1: x2].copy()
            cv2.addWeighted(
                src1=court_image,
                alpha=alpha,
                src2=overlay,
                beta=1 - alpha,
                gamma=0,
                dst=frame[y1: y2, x1: x2]
            )

            # Draw tactical court keypoints
            for keypoint_index, keypoint in enumerate(tactical_court_keypoints):
                x, y = keypoint
                x += self.start_x
                y += self.start_y

                cv2.circle(
                    img=frame,
                    center=(x, y),
                    radius=5,
                    color=(0, 0, 255),
                    thickness=-1
                )

                cv2.putText(
                    img=frame,
                    text=f"{keypoint_index}",
                    org=(x, y),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.5,
                    color=(0, 255, 0),
                    thickness=2,
                )
            output_video_frames.append(frame)

        return output_video_frames
