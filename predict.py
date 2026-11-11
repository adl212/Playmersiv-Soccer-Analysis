
import cv2
import numpy as np
from ultralytics import YOLO
from detect import BallDetector

from CameraMotion import CameraMotionEstimator
from BallTracker import BallTracker
from CameraMotionVisualizer import CameraMotionVisualizer

# --------------------------------------------------
# INITIALIZATION
# --------------------------------------------------

VIDEO_PATH = "18eBfrhGS3BurZIDah6IcXVfDnbUaE5Jk.mp4"

cap = cv2.VideoCapture(VIDEO_PATH)

camera = CameraMotionEstimator()

# Do not pass camera here because main loop handles it.
tracker = BallTracker()

visualizer = CameraMotionVisualizer()

detector = BallDetector(
    model_path="soccer_yolov8n_best.pt",
    confidence=0.25
)
# --------------------------------------------------
# MAIN LOOP
# --------------------------------------------------
cap.set(cv2.CAP_PROP_POS_FRAMES, 85000)
while True:

    ret, frame = cap.read()

    if not ret:
        break

    # ----------------------------------------------
    # 1. YOLO BALL DETECTION
    # ----------------------------------------------

    # Replace this with your existing YOLO detection.
    #
    # Expected format:
    # detection = (center_x, center_y)
    # or None if no ball is detected.

    detection = detector.get_ball_detection(frame)

    # ----------------------------------------------
    # 2. ESTIMATE CAMERA MOTION
    # ----------------------------------------------

    camera_result = camera.update(frame)

    camera_success = camera_result["success"]

    H = camera_result["homography"]

    # ----------------------------------------------
    # 3. GET PREVIOUS KALMAN STATE
    # ----------------------------------------------

    raw_prediction = None
    compensated_prediction = None

    if tracker.initialized:

        previous_state = tracker.kf.statePost.copy()

        F = tracker.kf.transitionMatrix

        # ------------------------------------------
        # 4. UNCOMPENSATED PREDICTION
        # ------------------------------------------

        raw_state = F @ previous_state

        raw_prediction = raw_state[:2].flatten()

        # ------------------------------------------
        # 5. CAMERA-COMPENSATED PREDICTION
        # ------------------------------------------

        if camera_success:

            # Temporarily transform a copy of the
            # previous state.

            original_state = tracker.kf.statePost.copy()
            original_covariance = tracker.kf.errorCovPost.copy()

            transformed = tracker._transform_state(H)

            if transformed:

                compensated_state = (
                    tracker.kf.transitionMatrix @
                    tracker.kf.statePost
                )

                compensated_prediction = (
                    compensated_state[:2].flatten()
                )

            # Restore original state. The tracker will
            # perform its actual update independently.
            tracker.kf.statePost = original_state
            tracker.kf.errorCovPost = original_covariance

    # ----------------------------------------------
    # 6. UPDATE BALL TRACKER
    # ----------------------------------------------

    ball_position = tracker.update(detection)

    # ----------------------------------------------
    # 7. VISUALIZATION
    # ----------------------------------------------

    output = visualizer.visualize(
        frame=frame,
        camera_result=camera_result,
        detection=detection,
        raw_prediction=raw_prediction,
        compensated_prediction=compensated_prediction
    )

    # Draw final tracked ball position.
    if ball_position is not None:

        x, y = map(int, ball_position)

        cv2.circle(
            output,
            (x, y),
            5,
            (0, 255, 0),
            -1
        )

        cv2.putText(
            output,
            "Final tracked position",
            (x + 10, y + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2
        )

    # ----------------------------------------------
    # 8. DISPLAY
    # ----------------------------------------------

    cv2.imshow("Camera Motion + Ball Tracking", output)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# --------------------------------------------------
# CLEANUP
# --------------------------------------------------

cap.release()
cv2.destroyAllWindows()
