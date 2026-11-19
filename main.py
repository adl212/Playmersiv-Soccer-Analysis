
import cv2
import numpy as np

from CameraMotion import CameraMotionEstimator
from BallTracker import BallTracker
from CameraMotionVisualizer import CameraMotionVisualizer
from detect import BallDetector
from field_mapper import FieldMapper
from LandmarkDetector import LandmarkDetector
from pitch_visualizer import PitchVisualizer
from player_tracker import PlayerTracker

# ============================================================
# CONFIGURATION
# ============================================================

VIDEO_PATH = "18eBfrhGS3BurZIDah6IcXVfDnbUaE5Jk.mp4"
MODEL_PATH = "soccer_yolov8n_best (1).pt"

LANDMARK_MODEL_PATH = "best_v7.pt"

# Re-estimate the field homography every 60 frames.
LANDMARK_UPDATE_INTERVAL = 30

# Search up to 120 frames for the first successful calibration.
INITIAL_CALIBRATION_SEARCH_FRAMES = 120

# Change this to match your model's player class ID.
PLAYER_CLASS_ID = 0

PITCH_WIDTH = 1050
PITCH_HEIGHT = 680

BALL_CONFIDENCE = 0.25

SHOW_CAMERA_MOTION = True
SHOW_PITCH = True

PLAYER_MAX_DISTANCE = 100
PLAYER_MAX_MISSED = 15
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

# Player tracker
player_tracker = PlayerTracker(
    max_distance=PLAYER_MAX_DISTANCE,
    max_missed=PLAYER_MAX_MISSED
)

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


# ============================================================
# AUTOMATED LANDMARK DETECTOR
# ============================================================

landmark_detector = LandmarkDetector(
    model_path=LANDMARK_MODEL_PATH,
    pitch_width=PITCH_WIDTH,
    pitch_height=PITCH_HEIGHT
)


# ============================================================
# INITIAL AUTOMATED CALIBRATION
# ============================================================

print("\n===================================")
print("     AUTOMATED FIELD CALIBRATION")
print("===================================")

H_field = None
calibrated_frame_number = None

for offset in range(INITIAL_CALIBRATION_SEARCH_FRAMES + 1):

    current_frame_number = first_frame_number + offset

    ret, calibration_frame = cap.read()

    if not ret:
        break

    (
        landmark_result,
        H,
        inlier_count,
        mean_error,
        detected_landmarks
    ) = landmark_detector.calibrate(calibration_frame)

    if H is None:
        print(
            f"Frame {current_frame_number}: "
            f"Calibration unsuccessful. Searching..."
        )
        continue

    H_field = H
    calibrated_frame_number = current_frame_number

    print("\nInitial calibration successful!")
    print(f"Frame: {calibrated_frame_number}")
    print(f"Inliers: {inlier_count}")
    print(f"Mean reprojection error: {mean_error:.2f}")

    break


if H_field is None:

    cap.release()
    cv2.destroyAllWindows()

    raise RuntimeError(
        "Could not automatically calibrate the field "
        f"within {INITIAL_CALIBRATION_SEARCH_FRAMES} frames."
    )


# Initialize the field mapper with the detected homography.
field_mapper.H_image_to_field = H_field


# Start processing from the frame used for calibration.
cap.set(
    cv2.CAP_PROP_POS_FRAMES,
    calibrated_frame_number
)

print("Starting video processing...")


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

    # Run YOLO once to detect both the ball and players.
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

    # Transform raw prediction into current-frame coordinates.
    if raw_prediction is not None and H_camera is not None:

        raw_prediction = transform_point(
            raw_prediction,
            H_camera
        )


    # ========================================================
    # 4. UPDATE BALL TRACKER
    # ========================================================

    ball_position = tracker.update(
        detection,
        camera_H=H_camera
    )


    # ========================================================
    # 5. UPDATE FIELD HOMOGRAPHY
    # ========================================================

    # --------------------------------------------------------
    # A. Propagate existing homography using camera movement
    # --------------------------------------------------------

    if camera_success:

        field_mapper.update_camera_motion(
            H_camera
        )


    # --------------------------------------------------------
    # B. Periodically recalibrate using automated landmarks
    # --------------------------------------------------------

    if frame_number % LANDMARK_UPDATE_INTERVAL == 0:

        (
            landmark_result,
            H_new,
            inlier_count,
            mean_error,
            detected_landmarks
        ) = landmark_detector.calibrate(frame)

        if H_new is not None:

            # Replace the propagated homography with the
            # newly estimated image-to-field homography.
            field_mapper.H_image_to_field = H_new

            print(
                f"\n[LANDMARK RECALIBRATION] "
                f"Frame: {frame_number} | "
                f"Inliers: {inlier_count} | "
                f"Mean error: {mean_error:.2f}"
            )

        else:

            # Keep the existing homography.
            print(
                f"\n[LANDMARK RECALIBRATION] "
                f"Frame: {frame_number} | "
                f"Failed. Retaining previous homography."
            )


    # ========================================================
    # 6. UPDATE PLAYER TRACKER
    # ========================================================

    # >>> NEW
    # Each player has an independent Kalman filter.
    #
    # Existing tracks are:
    # 1. Compensated for camera movement.
    # 2. Predicted into the current frame.
    # 3. Matched against current YOLO detections.
    # 4. Corrected when a matching detection is found.
    #
    # Players without detections retain their predicted positions.

    player_tracks = player_tracker.update(
        player_positions,
        camera_H=H_camera
    )


    # ========================================================
    # 7. MAP PLAYER POSITIONS TO FIELD COORDINATES
    # ========================================================

    # >>> UPDATED
    # Map Kalman-filtered positions instead of raw YOLO positions.

    player_field_positions = []

    for track in player_tracks:

        x, y = track["position"]

        field_position = field_mapper.image_to_field(
            (x, y)
        )

        if field_position is None:
            continue

        field_x, field_y = field_position

        if (
            np.isfinite(field_x) and
            np.isfinite(field_y) and
            0 <= field_x <= 1 and
            0 <= field_y <= 1
        ):

            player_field_positions.append({
                "id": track["id"],
                "position": (field_x, field_y),
                "detected": track["detected"],
                "missed": track["missed"]
            })


    # ========================================================
    # 8. MAP BALL POSITION TO FIELD COORDINATES
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
    # 9. CAMERA MOTION VISUALIZATION
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
    # 10. DRAW PLAYER DETECTIONS ON VIDEO
    # ========================================================

    # >>> UPDATED
    # Draw player bounding-box bottom-center points.

    for player_x, player_y in player_positions:

        cv2.circle(
            camera_view,
            (int(player_x), int(player_y)),
            4,
            (255, 100, 0),
            -1
        )


    # >>> NEW
    # Draw persistent player IDs on the video.

    for track in player_tracks:

        x, y = track["position"]

        color = (
            (0, 255, 0)
            if track["detected"]
            else (0, 255, 255)
        )

        cv2.putText(
            camera_view,
            f"Player {track['id']}",
            (int(x) + 8, int(y) - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1
        )


    # ========================================================
    # 11. DRAW TRACKED BALL ON VIDEO
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
    # 12. CREATE TOP-DOWN PITCH
    # ========================================================

    if SHOW_PITCH:

        pitch_view = pitch_visualizer.show(
            field_position
        )

        # >>> UPDATED
        # Draw all Kalman-filtered player positions.
        # Orange = currently detected.
        # Yellow = predicted because detection was missed.

        for player in player_field_positions:

            player_id = player["id"]

            field_x, field_y = player["position"]

            detected = player["detected"]

            pixel_x = int(
                field_x * (PITCH_WIDTH - 1)
            )

            pixel_y = int(
                field_y * (PITCH_HEIGHT - 1)
            )

            if detected:
                color = (255, 140, 0)
            else:
                color = (0, 255, 255)

            cv2.circle(
                pitch_view,
                (pixel_x, pixel_y),
                8,
                color,
                -1
            )

            cv2.putText(
                pitch_view,
                str(player_id),
                (pixel_x + 10, pixel_y - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1
            )


        # Player count
        cv2.putText(
            pitch_view,
            f"Players: {len(player_tracks)}",
            (15, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        # >>> NEW
        # Tracking legend

        cv2.circle(
            pitch_view,
            (20, 55),
            5,
            (255, 140, 0),
            -1
        )

        cv2.putText(
            pitch_view,
            "Detected",
            (35, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1
        )

        cv2.circle(
            pitch_view,
            (115, 55),
            5,
            (0, 255, 255),
            -1
        )

        cv2.putText(
            pitch_view,
            "Predicted",
            (130, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1
        )

        cv2.imshow(
            "Top Down Soccer Field",
            pitch_view
        )


    # ========================================================
    # 13. DISPLAY VIDEO
    # ========================================================

    cv2.imshow(
        "Camera Motion + Ball + Player Tracking",
        camera_view
    )


    # ========================================================
    # 14. DEBUG INFORMATION
    # ========================================================

    if camera_success:

        motion = camera_result["motion"]

        print(
            f"Frame {frame_number} | "
            f"Players: {len(player_tracks)} | "
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
            f"Players: {len(player_tracks)}"
        )

    if field_position is not None:

        print(
            f"Ball field position: "
            f"X={field_position[0]:.3f}, "
            f"Y={field_position[1]:.3f}"
        )


    # ========================================================
    # 15. USER INPUT
    # ========================================================

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

    elif key == ord("r"):

        print("Resetting ball and player trackers.")

        tracker.reset()

        # >>> NEW
        player_tracker = PlayerTracker(
            max_distance=PLAYER_MAX_DISTANCE,
            max_missed=PLAYER_MAX_MISSED
        )


# ============================================================
# CLEANUP
# ============================================================

cap.release()
cv2.destroyAllWindows()

print("\nVideo processing complete.")
