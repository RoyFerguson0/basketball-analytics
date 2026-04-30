from ultralytics import YOLO
import torch


def main():
    model = YOLO("runs/pose/train2/weights/best.pt")

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    results = model.predict(
        "input_videos/video_1.mp4", device=device, save=True)

    print(results[0])
    print("=============")
    for box in results[0].keypoints:
        print(box)


if __name__ == "__main__":
    main()
