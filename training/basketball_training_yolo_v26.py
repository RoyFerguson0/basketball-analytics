from pathlib import Path

from ultralytics import YOLO
from roboflow import Roboflow
from dotenv import dotenv_values
import torch
import gc


def get_project_root() -> Path:
    """Returns project root folder."""
    return Path(__file__).parent.parent


def main():
    print(get_project_root() / ".env")
    secrets = dotenv_values(get_project_root() / ".env")

    rf = Roboflow(api_key=secrets["ROBOFLOW_API_KEY"])
    project = rf.workspace(
        "workspace-5ujvu").project("basketball-players-fy4c2-vfsuv")
    version = project.version(17)
    dataset = version.download("yolo26")

    # Optional Cleanup to free up memory before training
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    print(f"Using device: {device}")

    model = YOLO("models/yolo26x.pt")

    model.train(
        task="detect",
        data=f"{dataset.location}/data.yaml",
        epochs=100,
        imgsz=640,
        batch=8,
        workers=4,
        device=device,
        plots=True
    )


if __name__ == "__main__":
    main()
