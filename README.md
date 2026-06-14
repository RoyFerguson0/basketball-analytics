# Basketball Video Analysis System

Computer vision pipeline for basketball game analysis. The project detects and tracks players and the ball from broadcast footage, identifies critical court landmarks, maps image coordinates to a top-down tactical view using homography transformation, assigns teams via zero-shot classification, and computes player speed and distance metrics in real-world units.

This repository is designed to show an end-to-end applied Computer Vision workflow rather than a single model demo. It combines object detection, multi-object tracking, homography-based perspective transformation, zero-shot learning for team assignment, and temporal analysis into one cohesive video-processing pipeline.

## Demo Video

**[Watch the demo video](assets/quick_demo_vid.mp4)**

[![Watch the demo](assets/thumbnail.png)](assets/quick_demo_vid.mp4)

## What the Project Does

- Detects players and the ball using a custom YOLOv8 model.
- Tracks detections across frames with multi-object tracking (ByteTrack/DeepSORT), including ball interpolation during occlusions.
- Identifies critical court keypoint landmarks (focal points, arcs, lines) for structural analysis.
- Applies **homography transformation** to convert image-space coordinates into a top-down tactical pitch view.
- Assigns players to teams using **zero-shot classification** based on visual features, without needing pre-labeled team data.
- Estimates ball possession using proximity heuristics and temporal logic.
- Computes instantaneous player speed and cumulative distance traveled.
- Detects and visualizes key tactical events: passes and interceptions.

## Technical Highlights

### Object Detection and Tracking

The pipeline starts with a YOLOv8 detector trained on basketball-specific classes (`player`, `ball`). Detections are passed through a multi-object tracker to maintain stable Player IDs across frames. The tracker handles ball detections with specific interpolation logic; when the ball is occluded (e.g., by a player's body), its position is estimated using kinematic interpolation to ensure continuous possession analysis.

### Court Keypoint Detection & Homography Transformation

To enable accurate tactical analysis, the system first detects court keypoints (such as the basket center, free-throw line, and court boundaries) in each frame. Using these keypoints, a **homography matrix** is computed to map the perspective-corrected video frame to a standardized, top-down 2D tactical view. This transformation allows distance and speed calculations to be performed in real-world metric units (meters/feet) rather than arbitrary pixels.

### Zero-Shot Team Classification

Instead of relying on detector attributes or pre-trained team embeddings, the project uses a **zero-shot classifier** to assign players to teams. By analyzing visual features (such as jersey color distribution and spatial clustering relative to the court geometry), the system infers team affiliation dynamically. This approach is robust against changing lighting conditions and does not require prior knowledge of the specific teams playing.

### Ball Possession Estimation

Ball possession is assigned using a proximity-based heuristic combined with temporal smoothing. The system calculates the distance between the ball's center and the nearest player's foot position. If the distance falls below a dynamic threshold and remains consistent over consecutive frames, possession is assigned to that player. Team possession is then aggregated to visualize control over time.

### Speed and Distance Calculation

Using the homography-transformed coordinates and the video's frame rate, the pipeline computes:

- **Instantaneous Speed**: The speed of each player in km/h, derived from the distance traveled between transformed coordinates.
- **Cumulative Distance**: The total distance traveled by each player in meters throughout the video segment.

### Tactical Event Detection

The pipeline includes logic to detect high-level events:

- **Pass Detection**: Identified by a rapid change in ball trajectory followed by the ball coming to rest near a different player.
- **Interception Detection**: Identified when the ball's trajectory intersects with a player who is not the ball carrier.

## Pipeline Overview

1.  Load the input video and extract frames.
2.  Run YOLOv8 detection for `player` and `ball` on each frame.
3.  Track players and the ball across frames using a multi-object tracker.
4.  Detect court keypoints and compute the **homography matrix** for perspective transformation.
5.  Project player and ball positions from image space to tactical (top-down) space.
6.  Assign team labels to each player using the **zero-shot classifier**.
7.  Estimate ball possession based on proximity and movement logic.
8.  Calculate player speed and distance from tactical coordinates.
9.  Detect passes and interceptions.
10. Render all visualizations (tracks, tactical view, metrics, events) on the original footage.
11. Save the annotated output video.

## Repository Structure

```bash
.
├── main.py                      # Orchestrates the full analysis pipeline.
├── input_videos/                # Directory for input basketball footage.
├── output_videos/               # Directory for annotated output videos.
├── models/                      # Pre-trained YOLOv8 weights and models.
├── trackers/                    # YOLO detection, multi-object tracking, and interpolation.
├── court_keypoint_detector/     # Detects court landmarks and structural points.
├── view_transformer/            # Homography transformation from image to tactical view.
├── team_assigner/               # Zero-shot classifier for team assignment.
├── possession_estimator/        # Logic for ball possession tracking.
├── pass_and_interception_detector/ # Detects passing and interception events.
├── speed_and_distance_calculator/ # Computes player speed and distance metrics.
├── drawers/                     # Visualization utilities for tracks, events, and metrics.
├── utils/                       # Video processing and geometry helper functions.
└── pyproject.toml               # Project dependencies.
```

## Installation

This project uses `uv` for environment management and Python packages listed in `pyproject.toml`.

```bash
# Clone the repository
git clone <repository-url>
cd basketball-analysis

# Install dependencies
uv sync
```

The project expects a CUDA-capable PyTorch setup for optimal performance. The environment is configured to use PyTorch with CUDA support.

## Running the Analysis

1.  Place a basketball video in the `input_videos/` directory.
2.  Ensure the video path in `main.py` points to your input file.
3.  Run the analysis:

```bash
uv run main.py
```

The annotated output video will be written to `output_videos/output_video.mp4`.

## Output Visualizations

The generated video includes multiple layers of information:

- **Tracking Data**: Player and ball bounding boxes with stable IDs.
- **Tactical View**: A converted overhead view of the court for spatial analysis.
- **Team Assignment**: Visual indicators of team membership (colors/icons).
- **Possession Overlay**: Color-coded zones indicating which team currently has the ball.
- **Performance Metrics**: Player speed and distance labels.
- **Event Markers**: Highlights for passes and interceptions.

## Notes on the Implementation

- **Homography**: Court keypoints are manually annotated in a reference frame or detected automatically to compute the initial homography matrix. This matrix is applied to subsequent frames assuming the camera angle remains relatively fixed (standard for broadcast footage).
- **Zero-Shot Classification**: The team assignment relies on visual embeddings rather than pre-trained class labels, making it adaptable to any game without retraining.
- **Ball Interpolation**: The ball tracker uses Kalman filtering or spline interpolation to handle short-term occlusions, ensuring that possession logic remains robust even when the ball is momentarily hidden.
- **Coordinate Transformation**: All speed and distance calculations are performed in the tactical coordinate space (meters) to ensure accuracy independent of camera zoom level.

## Why This Project is Useful

This is a compact but comprehensive example of a real Computer Vision system that goes beyond simple inference. It demonstrates advanced techniques including **homography-based perspective correction**, **zero-shot learning** for semantic grouping, **temporal tracking**, and **metric estimation**. This pipeline serves as an excellent reference for applied machine learning and computer vision roles, showing how to integrate multiple complex modules into a coherent analytical tool for sports analytics.
