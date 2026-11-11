Dataset on [Kaggle](https://www.kaggle.com/datasets/adl212/playmersiv-ai-camera-footage/versions/1)
# Soccer Camera Mapping

A starter project that maps YOLO player/ball detections from a moving single camera
onto a normalized 2D soccer pitch.

## 1. Install

```bash
python -m venv .venv
# Windows:
.venv\\Scripts\\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

Put your trained YOLO weights at `models/best.pt` (or supply another path).

## 2. Identify your model's class IDs

Run:

```bash
yolo predict model=models/best.pt source=data/sample.mp4
```

Or inspect them in Python:

```python
from ultralytics import YOLO
model = YOLO("models/best.pt")
print(model.names)
```

Set the player and ball class IDs when running the program. For example, if
`model.names` is `{0: 'ball',  1: 'player'}`, use `--player-classes 1 --ball-class 0`.

## 3. Calibrate a match

Start with a frame where several soccer markings are visible:

```bash
python -m src.field_calibration data/sample.mp4 --output calibrations/sample.json
```

Click at least four points in the video window. Use points that can be identified
on the soccer pitch (for example, field corners, penalty-area corners, halfway-line
intersections, or center-circle/halfway-line intersections).

After each click, enter that point's normalized field coordinates in the terminal:
- x = 0 is one goal line; x = 1 is the opposite goal line.
- y = 0 is one touchline; y = 1 is the opposite touchline.

The normalized coordinates do not assert actual pitch dimensions. Use the same
orientation consistently. Four points must not all lie on one line. More well-spread
points are better than four points clustered together.

Press `r` in the image window to restart point selection, or `q` to finish. The
script will calculate and save an image-to-field homography.

## 4. Run the visualization

```bash
python main.py \
  --video data/sample.mp4 \
  --weights models/best.pt \
  --calibration calibrations/sample.json \
  --player-classes 1 \
  --ball-class 0
```

For multiple player class IDs, use e.g. `--player-classes 1 2`.

Press `q` to quit. The left window shows the source video with detections; the
right window shows the normalized top-down pitch.

## What is and isn't automated

- YOLO detects players and the ball; ByteTrack assigns player IDs.
- The ball uses a separate lightweight constant-velocity Kalman filter.
- `field_marking_detection.py` extracts candidate edges/lines/circles. It does
  **not** decide that a line belongs to soccer. Multi-sport line classification
  remains an assisted geometric-recognition problem.
- Calibration is manual in this first version. It is specific to a match/camera
  setup and should be checked when the camera angle changes substantially.
- Camera motion is estimated from background features. This can fail on low-texture
  turf, large pans, motion blur, or when too little static background is visible.
  The code skips bad estimates instead of applying obviously unreliable transforms.
- A planar homography maps ground-plane points. Use player foot points, not box
  centers. An airborne ball's image center is not its ground location.
- Only visible detections are plotted. The system cannot recover players outside
  the camera view.

## Important calibration limitation

A single homography assumes the field is approximately planar and the camera view
is projective. If the camera tilts/zooms, moves substantially, or has strong lens
distortion, recalibration or a more advanced camera model may be needed.
