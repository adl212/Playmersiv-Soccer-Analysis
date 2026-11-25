# Playmersiv Soccer Analysis

This project turns soccer broadcast or training-camera frames into:

- live ball and player detections,
- camera-motion-compensated tracks,
- normalized positions on a 1050 x 680 top-down pitch,
- optional goal-event state and confirmation, and
- OpenCV windows for previewing the camera view and pitch map.

The repository has no separate goal-model or web API. It combines two YOLO
models, Kalman tracking, optical-flow/camera homographies, pitch
homographies, and a small rule-based goal state machine.

## Quick start

```bash
pip install -r requirements.txt
python main.py
```

Edit the video, model, and runtime settings at the top of `main.py`.
The current defaults assume that the included weights and video files are in
the repository root.

## Configuring the video

Set the input and starting frame near the top of `main.py`:

```python
VIDEO_PATH = "18eBfrhGS3BurZIDah6IcXVfDnbUaE5Jk.mp4"
FIRST_FRAME = 95000
```

- `VIDEO_PATH` accepts a relative path, such as a video in the repository
  root, or an absolute path to another local file.
- `FIRST_FRAME` skips to that frame before processing starts. Set it to `0`
  to begin at the beginning of the video.
- Local `.mp4`, `.mov`, `.avi`, and other formats supported by OpenCV
  `cv2.VideoCapture` can be used.
- For RTSP or other live-stream URLs, assign the stream URL to `VIDEO_PATH`,
  for example `rtsp://host:554/live/stream`. URL support depends on the
  OpenCV build and network access. Live streams should normally use
  `FIRST_FRAME = 0`.

Model paths can also be configured beside the video settings:

```python
MODEL_PATH = "soccer_yolov8n_best (1).pt"
LANDMARK_MODEL_PATH = "best_v7.pt"
```

If a file is not in the repository root, use its absolute path. The pitch is fixed to `1050 x 680` by
`PITCH_WIDTH` and `PITCH_HEIGHT`; leave these unchanged unless the pitch
visualization dimensions are intentionally changed.

Controls:

- `q`: quit
- `r`: reset the ball/player trackers, camera motion, field calibration, and
  goal detector

Run tests with:

```bash
pytest
```

## Models and algorithms

| Component | What it is | Main use |
| --- | --- | --- |
| `soccer_yolov8n_best (1).pt` | YOLOv8 object-detection weights | Finds the ball and player boxes. The default in `main.py` is `soccer_yolov8n_best (1).pt`. |
| `best_v7.pt` | YOLO pose/detection weights with named pitch keypoints | Finds semantic pitch landmarks and supplies correspondences for the absolute image-to-pitch homography. |
| `BallTracker` | OpenCV 4-state Kalman filter | Smooths ball position and predicts through brief missed detections. |
| `PlayerTracker` | Per-player Kalman filters plus nearest-neighbor assignment | Tracks player ground-contact points and maintains IDs across frames. |
| `CameraMotionEstimator` | Semantic landmark/feature tracking with RANSAC homographies | Estimates how the previous frame maps into the current frame. |
| `LandmarkDetector` | Pose-keypoint extraction plus RANSAC homography | Estimates the absolute mapping from camera pixels to canonical pitch pixels. |
| `FieldMapper` | Homography composition and normalized coordinate mapping | Applies camera motion and maps image points to normalized `(x, y)` values in `[0, 1]`. |
| `GoalDetector` | Rule-based state machine; not a trained model | Proposes shots, detects goal-area evidence and kickoffs, and confirms a goal when evidence is combined. |

There is also a manual fallback in `ManualCalibration`: it lets a user select
image landmarks and assign their canonical pitch coordinates.

## How the full stack works

The important coordinate systems are:

1. **Camera pixels**: `(x, y)` positions in each source frame.
2. **Previous-frame/current-frame pixels**: camera-motion homography
   coordinates.
3. **Canonical pitch pixels**: the 1050 x 680 pitch used by `FieldMapper`.
4. **Normalized pitch coordinates**: `(x, y)` in `[0, 1]`; `(0, 0)` is the
   top-left and `(1, 1)` is the bottom-right.

For every frame, `main.py` performs this sequence:

```text
Video frame
  -> YOLO object detector
     -> ball center (camera pixels)
     -> player bottom centers (camera pixels)
  -> Landmark pose detector, every 15 frames
     -> semantic pitch landmarks
     -> absolute image-to-pitch homography
  -> Camera motion estimator, every frame
     -> previous-frame-to-current-frame homography
     -> updates Kalman states and pitch mapping
  -> Ball/player trackers
     -> camera-compensated, persistent tracks
  -> FieldMapper
     -> normalized top-down player/ball positions
  -> GoalDetector
     -> shot/goal/kickoff state
  -> OpenCV camera and pitch visualizations
```

The landmark model creates or refreshes the absolute field calibration.
`CameraMotionEstimator` handles movement between its sparse landmark updates,
using tracked semantic landmarks first and generic optical-flow features as a
fallback. `FieldMapper.update_camera_motion()` then propagates the old
image-to-pitch homography into the new frame.

Kalman trackers reject implausible detections and predict positions while a
target is temporarily missing. The player track stores its image position,
velocity, ID, detection/missed status. The ball track similarly stores a
camera-pixel position and velocity.

## Public functions for an API or external display

The most useful integration points are:

### `BallDetector`

```python
from detect import BallDetector

detector = BallDetector(model_path="soccer_yolov8n_best (1).pt",
                        confidence=0.50)
ball, players = detector.get_detections(frame, player_class_id=0)
```

- `ball`: highest-confidence `(x, y)` ball center, or `None`
- `players`: player bottom-center `(x, y)` points

### `LandmarkDetector`

```python
from LandmarkDetector import LandmarkDetector

landmarks = LandmarkDetector(
    model_path="best_v7.pt",
    pitch_width=1050,
    pitch_height=680,
)
calibration = landmarks.calibrate(frame)
```

Use `calibration["success"]` before reading
`calibration["homography"]`. `calibration["observations"]` contains the named
semantic landmarks with image coordinates and confidence scores.

### `CameraMotionEstimator`

```python
camera = CameraMotionEstimator()
camera_result = camera.update(frame, landmarks=observations)
```

Read `camera_result["homography"]` only when
`camera_result["success"]` is true. It maps previous-frame pixels to
current-frame pixels.

### `BallTracker` and `PlayerTracker`

```python
from BallTracker import BallTracker
from player_tracker import PlayerTracker

ball_tracker = BallTracker()
player_tracker = PlayerTracker(max_distance=100, max_missed=15)

ball_position = ball_tracker.update(ball_detection, camera_H=H_camera)
player_tracks = player_tracker.update(player_positions, camera_H=H_camera)
```

Pass no stale Kalman-only value as a fresh detection. `None` means that there
was no accepted YOLO observation in the current frame.

### `FieldMapper`

```python
from field_mapper import FieldMapper

field_mapper = FieldMapper(pitch_width=1050, pitch_height=680)
field_mapper.H_image_to_field = calibration["homography"].copy()

# Each frame, after a successful camera update:
field_mapper.update_camera_motion(H_camera)
pitch_ball = field_mapper.image_to_field(ball_position)
pitch_player = field_mapper.image_to_field(player_position)
```

`image_to_field()` returns normalized `(x, y)` values or `None`. The
`map_track_to_field()` helper also checks that the result is inside the pitch.

### `GoalDetector`

```python
from GoalDetector import GoalDetector

goal_detector = GoalDetector(fps=30)
goal_result = goal_detector.update(
    frame_number=frame_number,
    observed_ball_field_position=pitch_ball,
    ball_missed_frames=ball_tracker.missed_frames,
)
```

The result is a dict containing `goal`, `goal_side`, `event_frame`,
`confidence`, `state`, `reason`, and `goal_area_evidence`. Goal confirmation
requires goal-area evidence followed by an observed center-spot restart; it is
not referee-level goal-line verification.

## Suggested live-stream API

For a frame-processing endpoint or WebSocket service, create one processor
object and keep its trackers, field mapper, camera estimator, and goal detector
alive across frames. Do not recreate them for every request.

A small processor loop could be structured as:

```python
last_observations = {}

def process_frame(frame, frame_number):
    global last_observations

    ball, players = detector.get_detections(frame, player_class_id=0)

    calibration = None
    if field_mapper.H_image_to_field is None or frame_number % 15 == 0:
        calibration = landmark_detector.calibrate(frame)
        if calibration["success"]:
            last_observations = calibration["observations"]

    observations = (
        calibration["observations"]
        if calibration is not None and calibration["success"]
        else last_observations
    )
    camera_result = camera.update(frame, landmarks=observations)
    H_camera = (
        camera_result["homography"]
        if camera_result["success"]
        else None
    )

    if H_camera is not None:
        field_mapper.update_camera_motion(H_camera)
    if calibration is not None and calibration["success"]:
        field_mapper.H_image_to_field = calibration["homography"].copy()

    ball_position = ball_tracker.update(ball, camera_H=H_camera)
    player_tracks = player_tracker.update(players, camera_H=H_camera)

    ball_field = (
        field_mapper.image_to_field(ball_position)
        if ball_position is not None else None
    )
    ball_observed_field = (
        field_mapper.image_to_field(ball)
        if ball is not None else None
    )
    player_fields = [
        {
            "id": track["id"],
            "field_position": field_mapper.image_to_field(track["position"]),
            "detected": track["detected"],
            "missed": track["missed"],
        }
        for track in player_tracks
    ]

    goal_result = goal_detector.update(
        frame_number,
        ball_observed_field,
        ball_tracker.missed_frames,
    )

    return {
        "frame": frame_number,
        "ball": ball_field,
        "players": player_fields,
        "goal": goal_result,
        "camera": {
            "success": camera_result["success"],
            "source": camera_result["source"],
            "motion": camera_result["motion"],
        },
    }
```

The payload is already suitable for JSON after converting NumPy values to
Python floats/ints. Expose it over WebSockets or Server-Sent Events for live
display, or publish it as a REST response for polled/recorded frames. If the
display runs in another process or browser, send the normalized pitch
coordinates rather than the raw image pixels.

Useful display helpers:

- `PitchVisualizer.show(normalized_position)` draws the ball on the canonical
  pitch and returns an OpenCV image.
- `visualization.draw_pitch(players, ball=..., ball_observed=..., trail=...)`
  provides a compact normalized-pitch renderer.
- `CameraMotionVisualizer.visualize(...)` draws detections, predictions,
  motion vectors, and camera status on the source frame.

## Important limitations

- The field calibration is a ground-plane homography. The center of an aerial
  ball is not on the ground plane, so airborne ball positions can be
  visually misplaced.
- Goal events are heuristic: no goalposts or goal-line model is present.
- Goal-mouth and kickoff thresholds in `GoalDetector.__init__()` are starting
  values and should be tuned for a specific camera, frame rate, and dataset.
- A landmark calibration or camera estimate can fail. API payloads should
  therefore distinguish a missing field position from a real position at
  `(0, 0)`.
- JSON output should include tracking IDs, confidence or detection status, and
  frame timestamps when used by an external display; the example above is the
  core data contract.
