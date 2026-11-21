Dataset on [Kaggle](https://www.kaggle.com/datasets/adl212/playmersiv-ai-camera-footage/versions/1)
**predict.py** currently visualizes how the ball is tracked including camera movement

## Goal event detection

The `goal-detection` branch adds a lightweight, rule-based goal-event detector.

### How it works

1. Tracks **observed YOLO ball detections** in normalized pitch coordinates and estimates the short-term trajectory.
2. Creates a shot candidate when the trajectory projects toward the left or right goal mouth.
3. Looks for the ball reaching the goal-line region, or disappearing shortly after a projected shot near that region.
4. Confirms the event only after a center-spot restart is observed (ball remains near the center briefly, then moves away).

The detector has no additional trained model. `GoalDetector.py` contains the state machine, and `main.py` displays its state on both the camera and top-down pitch windows. A confirmed event is also printed to the terminal.

### Configuration

The initial thresholds are in `GoalDetector.__init__`:
- `goal_mouth_y`: normalized vertical goal-mouth interval.
- `shot_speed_threshold`: minimum normalized pitch speed for a shot candidate.
- `max_projection_seconds`: maximum time horizon for the projected trajectory.
- `candidate_timeout_seconds`: maximum time allowed between the shot candidate and restart.
- `kickoff_center_radius`, `kickoff_stationary_frames`, and `kickoff_move_threshold`: restart detection thresholds.

These are starting values, not universal soccer constants. Tune them against labeled clips from the camera angle and frame rate used in your dataset.

### Important limitations

- A pitch homography maps the ground plane. The center of an airborne ball is not on that plane, so aerial shots can be misplaced.
- The goal mouth is approximated from normalized pitch coordinates; the current landmark model does not explicitly detect the goalposts or goal line.
- A ball near the goal or disappearing from view is not proof that it crossed the line. The kickoff is supporting evidence, not ground-truth confirmation.
- Expect false positives and missed goals until the thresholds are calibrated and evaluated on labeled match clips.

Run the project as usual with the existing video and model paths configured in `main.py`. Press `q` to quit and `r` to reset trackers and the goal detector.

Synthetic unit tests are in `tests/test_goal_detector.py` and can be run with `pytest`.
