
import cv2
import numpy as np

from CameraMotion import CameraMotionEstimator
from BallTracker import BallTracker
from CameraMotionVisualizer import CameraMotionVisualizer
from detect import BallDetector
from field_mapper import FieldMapper
from manual_calibration import ManualCalibration
from pitch_visualizer import PitchVisualizer


# ============================================================
# CONFIGURATION
# ============================================================

VIDEO_PATH = "18eBfrhGS3BurZIDah6IcXVfDnbUaE5Jk.mp4"
MODEL_PATH = "soccer_yolov8n_best.pt"

PITCH_WIDTH = 1050
PITCH_HEIGHT = 680

BALL_CONFIDENCE = 0.25

SHOW_CAMERA_MOTION = True
SHOW_PITCH = True

# ============================================================
# INITIALIZATION
# ============================================================

cap = cv2.VideoCapture(VIDEO_PATH)
initial_frame = 39000

cap.set(cv2.CAP_PROP_POS_FRAMES, initial_frame)
if not cap.isOpened():
    raise RuntimeError(f"Could not open video: {VIDEO_PATH}")

camera = CameraMotionEstimator()

tracker = BallTracker()

detector = BallDetector(
    model_path=MODEL_PATH,
    confidence=BALL_CONFIDENCE
)

camera_visualizer = CameraMotionVisualizer()

field_mapper = FieldMapper(
    pitch_width=PITCH_WIDTH,
    pitch_height=PITCH_HEIGHT
)

pitch_visualizer = PitchVisualizer(
    width=PITCH_WIDTH,
    height=PITCH_HEIGHT
)

calibrator = ManualCalibration(
    pitch_width=PITCH_WIDTH,
    pitch_height=PITCH_HEIGHT
)


# ============================================================
# INITIAL MANUAL CALIBRATION
# ============================================================

ret, first_frame = cap.read()

if not ret:
    raise RuntimeError("Could not read initial video frame.")

print("\n===================================")
print("       SOCCER FIELD CALIBRATION")
print("===================================")

print("\nSelect at least four recognizable field landmarks.")
print("Press ENTER when finished.")
print("Press U to undo.")
print("Press R to reset.")
print("Press ESC to cancel.")

calibration_result = calibrator.calibrate(first_frame)

if calibration_result is None:
    cap.release()
    cv2.destroyAllWindows()
    raise RuntimeError("Field calibration cancelled.")

H_field, image_points, field_points = calibration_result

field_mapper.H_image_to_field = H_field

print("\nField calibration successful.")

# Restart video from beginning
cap.set(cv2.CAP_PROP_POS_FRAMES, initial_frame)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def transform_point(point, H):
    """
    Transform a point between video frames.
    """

    point = np.array(
        [[[point[0], point[1]]]],
        dtype=np.float32
    )

    transformed = cv2.perspectiveTransform(
        point,
        H
    )

    return transformed[0, 0]


def get_uncompensated_prediction(tracker):
    """
    Predict ball position without camera compensation.
    """

    if not tracker.initialized:
        return None

    F = tracker.kf.transitionMatrix

    state = tracker.kf.statePost.copy()

    prediction = F @ state

    return prediction[:2].flatten()


def get_compensated_prediction(tracker, H):
    """
    Predict ball position with camera compensation.

    Uses a temporary state so the actual tracker is not modified.
    """

    if not tracker.initialized or H is None:
        return None

    state = tracker.kf.statePost.copy()

    x, y, vx, vy = state.flatten()

    H = H.astype(np.float64)

    denominator = (
        H[2, 0] * x +
        H[2, 1] * y +
        H[2, 2]
    )

    if abs(denominator) < 1e-8:
        return None

    numerator_x = (
        H[0, 0] * x +
        H[0, 1] * y +
        H[0, 2]
    )

    numerator_y = (
        H[1, 0] * x +
        H[1, 1] * y +
        H[1, 2]
    )

    new_x = numerator_x / denominator
    new_y = numerator_y / denominator

    J = np.array([
        [
            (H[0, 0] * denominator -
             numerator_x * H[2, 0]) / denominator**2,

            (H[0, 1] * denominator -
             numerator_x * H[2, 1]) / denominator**2
        ],
        [
            (H[1, 0] * denominator -
             numerator_y * H[2, 0]) / denominator**2,

            (H[1, 1] * denominator -
             numerator_y * H[2, 1]) / denominator**2
        ]
    ])

    velocity = J @ np.array([vx, vy])

    compensated_state = np.array([
        [new_x],
        [new_y],
        [velocity[0]],
        [velocity[1]]
    ], dtype=np.float32)

    prediction = tracker.kf.transitionMatrix @ compensated_state

    return prediction[:2].flatten()


# ============================================================
# MAIN VIDEO LOOP
# ============================================================

frame_number = 0

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame_number += 1

    # --------------------------------------------------------
    # 1. YOLO BALL DETECTION
    # --------------------------------------------------------

    detection = detector.get_ball_detection(frame)

    # --------------------------------------------------------
    # 2. CAMERA MOTION ESTIMATION
    # --------------------------------------------------------

    camera_result = camera.update(frame)

    camera_success = camera_result["success"]

    H_camera = (
        camera_result["homography"]
        if camera_success
        else None
    )

    # --------------------------------------------------------
    # 3. CALCULATE PREDICTIONS
    # --------------------------------------------------------

    raw_prediction = get_uncompensated_prediction(
        tracker
    )

    compensated_prediction = get_compensated_prediction(
        tracker,
        H_camera
    )

    # Both predictions are now in current-frame coordinates.
    # The raw prediction is explicitly transformed for display.

    if raw_prediction is not None and camera_success:

        raw_prediction = transform_point(
            raw_prediction,
            H_camera
        )

    # --------------------------------------------------------
    # 4. UPDATE BALL TRACKER
    # --------------------------------------------------------

    ball_position = tracker.update(
        detection,
        camera_H=H_camera
    )

    # --------------------------------------------------------
    # 5. UPDATE FIELD MAPPING
    # --------------------------------------------------------

    if camera_success:

        field_mapper.update_camera_motion(
            H_camera
        )

    # --------------------------------------------------------
    # 6. MAP BALL TO FIELD COORDINATES
    # --------------------------------------------------------

    field_position = None

    if ball_position is not None:

        field_position = field_mapper.image_to_field(
            ball_position
        )

    # --------------------------------------------------------
    # 7. CAMERA MOTION VISUALIZATION
    # --------------------------------------------------------

    camera_view = frame.copy()

    if SHOW_CAMERA_MOTION:

        camera_view = camera_visualizer.visualize(
            frame=frame,
            camera_result=camera_result,
            detection=detection,
            raw_prediction=raw_prediction,
            compensated_prediction=compensated_prediction
        )

    # --------------------------------------------------------
    # 8. DRAW FINAL TRACKED BALL
    # --------------------------------------------------------

    if ball_position is not None:

        x, y = map(
            int,
            np.round(ball_position)
        )

        cv2.circle(
            camera_view,
            (x, y),
            5,
            (0, 255, 0),
            -1
        )

        cv2.putText(
            camera_view,
            "Tracked ball",
            (x + 10, y + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2
        )

    # --------------------------------------------------------
    # 9. TOP-DOWN PITCH VISUALIZATION
    # --------------------------------------------------------

    if SHOW_PITCH:

        pitch_view = pitch_visualizer.show(
            field_position
        )

        cv2.imshow(
            "Top Down Soccer Field",
            pitch_view
        )

    # --------------------------------------------------------
    # 10. DISPLAY CAMERA VIEW
    # --------------------------------------------------------

    cv2.imshow(
        "Camera Motion + Ball Tracking",
        camera_view
    )

    # --------------------------------------------------------
    # 11. DEBUG INFORMATION
    # --------------------------------------------------------

    if camera_success:

        motion = camera_result["motion"]

        print(
            f"Frame {frame_number} | "
            f"Pan X: {motion['translation_x']:.1f} | "
            f"Pan Y: {motion['translation_y']:.1f} | "
            f"Zoom: {motion['zoom_percent']:+.2f}% | "
            f"Rotation: {motion['rotation_degrees']:.2f} | "
            f"Inliers: {camera_result['inliers']}"
        )

    if field_position is not None:

        print(
            f"Ball field position: "
            f"X={field_position[0]:.3f}, "
            f"Y={field_position[1]:.3f}"
        )

    # --------------------------------------------------------
    # 12. USER INPUT
    # --------------------------------------------------------

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

    elif key == ord("r"):
        print("Resetting ball tracker.")
        tracker.reset()


# ============================================================
# CLEANUP
# ============================================================

cap.release()
cv2.destroyAllWindows()

print("\nVideo processing complete.")
