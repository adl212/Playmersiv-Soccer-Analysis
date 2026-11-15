
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
MODEL_PATH = "soccer_yolov8n_best (1).pt"

# Change this to match your model's player class ID.
PLAYER_CLASS_ID = 0

PITCH_WIDTH = 1050
PITCH_HEIGHT = 680

BALL_CONFIDENCE = 0.25

SHOW_CAMERA_MOTION = True
SHOW_PITCH = True

# Maximum distance between a predicted ball position
# and YOLO detection before rejecting that detection.
# This is configured in ball_tracker.py.


# ============================================================
# INITIALIZATION
# ============================================================

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    raise RuntimeError(f"Could not open video: {VIDEO_PATH}")

first_frame_number = 95000
cap.set(cv2.CAP_PROP_POS_FRAMES, first_frame_number)

# Camera motion estimator
camera = CameraMotionEstimator()


# Ball tracker
tracker = BallTracker()


# YOLO detector
detector = BallDetector(
    model_path=MODEL_PATH,
    confidence=BALL_CONFIDENCE
)

print("YOLO classes:", detector.model.names)


# Visualizers
camera_visualizer = CameraMotionVisualizer()

pitch_visualizer = PitchVisualizer(
    width=PITCH_WIDTH,
    height=PITCH_HEIGHT
)


# Field mapper
field_mapper = FieldMapper(
    pitch_width=PITCH_WIDTH,
    pitch_height=PITCH_HEIGHT
)


# Manual calibration
calibrator = ManualCalibration(
    pitch_width=PITCH_WIDTH,
    pitch_height=PITCH_HEIGHT
)


# ============================================================
# INITIAL MANUAL CALIBRATION
# ============================================================

ret, first_frame = cap.read()

if not ret:
    cap.release()
    raise RuntimeError("Could not read initial video frame.")


print("\n===================================")
print("       SOCCER FIELD CALIBRATION")
print("===================================")

print("\nSelect recognizable soccer field landmarks.")
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

# Restart video from the beginning.
cap.set(cv2.CAP_PROP_POS_FRAMES, first_frame_number)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def transform_point(point, H):
    """
    Transform a point from the previous frame
    into the current frame.

    H maps previous-frame coordinates
    to current-frame coordinates.
    """

    point_array = np.array(
        [[[point[0], point[1]]]],
        dtype=np.float32
    )

    transformed = cv2.perspectiveTransform(
        point_array,
        H
    )

    return transformed[0, 0]


def get_raw_prediction(tracker):
    """
    Predict ball position without camera compensation.

    The returned position is in previous-frame coordinates.
    """

    if not tracker.initialized:
        return None

    F = tracker.kf.transitionMatrix

    state = tracker.kf.statePost.copy()

    prediction = F @ state

    return prediction[:2].flatten()


def get_compensated_prediction(tracker, H):
    """
    Predict ball position after compensating for camera motion.

    Does not modify the actual Kalman filter state.
    The returned position is in current-frame coordinates.
    """

    if not tracker.initialized:
        return None

    state = tracker.kf.statePost.copy()

    x, y, vx, vy = state.flatten()

    # Without a valid homography, use the ordinary prediction.
    if H is None:
        prediction = tracker.kf.transitionMatrix @ state
        return prediction[:2].flatten()

    H = H.astype(np.float64)

    # Apply homography to position.
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

    # Calculate local Jacobian to transform velocity.
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

    prediction = (
        tracker.kf.transitionMatrix @ compensated_state
    )

    return prediction[:2].flatten()


def map_players_to_field(player_positions, field_mapper):
    """
    Convert player image coordinates into normalized field coordinates.

    Each player position is the bottom-center of their bounding box.

    Returns:
        List of normalized (x, y) field coordinates.
    """

    player_field_positions = []

    for player_position in player_positions:

        field_position = field_mapper.image_to_field(
            player_position
        )

        if field_position is None:
            continue

        field_x, field_y = field_position

        # Ignore positions outside the reference pitch.
        if (
            np.isfinite(field_x) and
            np.isfinite(field_y) and
            0 <= field_x <= 1 and
            0 <= field_y <= 1
        ):

            player_field_positions.append(
                (field_x, field_y)
            )

    return player_field_positions


def draw_players_on_pitch(pitch, player_positions):
    """
    Draw player positions onto the top-down pitch.
    """

    for field_x, field_y in player_positions:

        pixel_x = int(field_x * (PITCH_WIDTH - 1))
        pixel_y = int(field_y * (PITCH_HEIGHT - 1))

        cv2.circle(
            pitch,
            (pixel_x, pixel_y),
            8,
            (255, 100, 0),
            -1
        )

        cv2.circle(
            pitch,
            (pixel_x, pixel_y),
            8,
            (255, 255, 255),
            1
        )

    return pitch


# ============================================================
# MAIN VIDEO LOOP
# ============================================================

frame_number = 0

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame_number += 1


    # ========================================================
    # 1. YOLO DETECTION
    # ========================================================

    # Run the model once to detect both the ball and players.
    detection, player_positions = detector.get_detections(
        frame,
        player_class_id=PLAYER_CLASS_ID
    )


    # ========================================================
    # 2. CAMERA MOTION ESTIMATION
    # ========================================================

    # Estimate previous-frame -> current-frame homography.
    camera_result = camera.update(frame)

    camera_success = camera_result["success"]

    H_camera = (
        camera_result["homography"]
        if camera_success
        else None
    )


    # ========================================================
    # 3. BALL PREDICTIONS
    # ========================================================

    raw_prediction = get_raw_prediction(tracker)

    compensated_prediction = get_compensated_prediction(
        tracker,
        H_camera
    )

    # Transform raw prediction into current-frame coordinates
    # so both predictions can be displayed on the current frame.
    if raw_prediction is not None and H_camera is not None:

        raw_prediction = transform_point(
            raw_prediction,
            H_camera
        )


    # ========================================================
    # 4. UPDATE BALL TRACKER
    # ========================================================

    # The tracker transforms its internal state using H_camera
    # before predicting and correcting with the current detection.
    ball_position = tracker.update(
        detection,
        camera_H=H_camera
    )


    # ========================================================
    # 5. UPDATE FIELD HOMOGRAPHY
    # ========================================================

    # Update the current image -> field transformation.
    #
    # H_camera maps previous frame -> current frame.
    # Therefore:
    #
    # H_field_current =
    #     H_field_previous @ inverse(H_camera)

    if camera_success:

        field_mapper.update_camera_motion(
            H_camera
        )


    # ========================================================
    # 6. MAP PLAYER POSITIONS TO FIELD COORDINATES
    # ========================================================

    player_field_positions = map_players_to_field(
        player_positions,
        field_mapper
    )


    # ========================================================
    # 7. MAP BALL POSITION TO FIELD COORDINATES
    # ========================================================

    field_position = None

    if ball_position is not None:

        field_position = field_mapper.image_to_field(
            ball_position
        )

        if field_position is not None:

            field_x, field_y = field_position

            if not (
                np.isfinite(field_x) and
                np.isfinite(field_y) and
                0 <= field_x <= 1 and
                0 <= field_y <= 1
            ):
                field_position = None


    # ========================================================
    # 8. CAMERA MOTION VISUALIZATION
    # ========================================================

    camera_view = frame.copy()

    if SHOW_CAMERA_MOTION:

        camera_view = camera_visualizer.visualize(
            frame=frame,
            camera_result=camera_result,
            detection=detection,
            raw_prediction=raw_prediction,
            compensated_prediction=compensated_prediction
        )


    # ========================================================
    # 9. DRAW PLAYER DETECTIONS ON VIDEO
    # ========================================================

    # Draw the YOLO player bottom-center points.
    for player_x, player_y in player_positions:

        x = int(player_x)
        y = int(player_y)

        cv2.circle(
            camera_view,
            (x, y),
            4,
            (255, 100, 0),
            -1
        )


    # ========================================================
    # 10. DRAW TRACKED BALL ON VIDEO
    # ========================================================

    if ball_position is not None:

        x, y = map(
            int,
            np.round(ball_position)
        )

        cv2.circle(
            camera_view,
            (x, y),
            6,
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


    # ========================================================
    # 11. CREATE TOP-DOWN PITCH
    # ========================================================

    if SHOW_PITCH:

        # Create pitch and plot ball.
        pitch_view = pitch_visualizer.show(
            field_position
        )

        # Plot all detected players.
        pitch_view = draw_players_on_pitch(
            pitch_view,
            player_field_positions
        )

        # Display player count.
        cv2.putText(
            pitch_view,
            f"Players: {len(player_field_positions)}",
            (15, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        cv2.imshow(
            "Top Down Soccer Field",
            pitch_view
        )


    # ========================================================
    # 12. DISPLAY VIDEO
    # ========================================================

    cv2.imshow(
        "Camera Motion + Ball + Player Detection",
        camera_view
    )


    # ========================================================
    # 13. DEBUG INFORMATION
    # ========================================================

    if camera_success:

        motion = camera_result["motion"]

        print(
            f"Frame {frame_number} | "
            f"Players: {len(player_field_positions)} | "
            f"Pan X: {motion['translation_x']:.1f} | "
            f"Pan Y: {motion['translation_y']:.1f} | "
            f"Zoom: {motion['zoom_percent']:+.2f}% | "
            f"Rotation: {motion['rotation_degrees']:.2f} | "
            f"Inliers: {camera_result['inliers']}"
        )

    else:

        print(
            f"Frame {frame_number} | "
            f"Camera estimation failed | "
            f"Players: {len(player_field_positions)}"
        )

    if field_position is not None:

        print(
            f"Ball field position: "
            f"X={field_position[0]:.3f}, "
            f"Y={field_position[1]:.3f}"
        )


    # ========================================================
    # 14. USER INPUT
    # ========================================================

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
