import torch
from ultralytics import YOLO


def main():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # model = YOLO("models/yolo26x.pt")
    model = YOLO("models/detector.pt")

    results = model.predict(
        source="input_videos/video_1.mp4",
        save=True,
        device=device
    )

    print(results)

    print("---------")
    for box in results[0].boxes:
        print(box)


if __name__ == "__main__":
    main()
