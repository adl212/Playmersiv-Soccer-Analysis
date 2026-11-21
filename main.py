import cv2
import numpy as np

from CameraMotion import CameraMotionEstimator
from BallTracker import BallTracker
from CameraMotionVisualizer import CameraMotionVisualizer
from detect import BallDetector
from field_mapper import FieldMapper
from pitch_visualizer import PitchVisualizer
from player_tracker import PlayerTracker
from LandmarkDetector import LandmarkDetector
from GoalDetector import GoalDetector
from helper import (
    get_raw_prediction,
    get_compensated_prediction,
    transform_point,
    map_track_to_field,
    draw_landmarks,
)

# ------------------------------- CONFIG -------------------------------
VIDEO_PATH = "18eBfrhGS3BurZIDah6IcXVfDnbUaE5Jk.mp4"
MODEL_PATH = "soccer_yolov8n_best (1).pt"
LANDMARK_MODEL_PATH = "best_v7.pt"

FIRST_FRAME = 95000
PLAYER_CLASS_ID = 0
PITCH_WIDTH, PITCH_HEIGHT = 1050, 680
BALL_CONFIDENCE = 0.25
LANDMARK_CONFIDENCE = 0.20
KEYPOINT_CONFIDENCE = 0.35
LANDMARK_UPDATE_INTERVAL = 15

SHOW_CAMERA_MOTION = True
SHOW_PITCH = True
SHOW_LANDMARKS = True
PLAYER_MAX_DISTANCE = 100
PLAYER_MAX_MISSED = 15

# A fresh absolute field calibration is accepted only when its own
# geometric validation passes in LandmarkDetector.
# ----------------------------------------------------------------------

cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise RuntimeError(f"Could not open video: {VIDEO_PATH}")
cap.set(cv2.CAP_PROP_POS_FRAMES, FIRST_FRAME)

camera = CameraMotionEstimator()
ball_tracker = BallTracker()
player_tracker = PlayerTracker(
    max_distance=PLAYER_MAX_DISTANCE,
    max_missed=PLAYER_MAX_MISSED,
)
detector = BallDetector(model_path=MODEL_PATH, confidence=BALL_CONFIDENCE)
landmark_detector = LandmarkDetector(
    model_path=LANDMARK_MODEL_PATH,
    pitch_width=PITCH_WIDTH,
    pitch_height=PITCH_HEIGHT,
    confidence=LANDMARK_CONFIDENCE,
    keypoint_confidence=KEYPOINT_CONFIDENCE,
)
camera_visualizer = CameraMotionVisualizer()
pitch_visualizer = PitchVisualizer(width=PITCH_WIDTH, height=PITCH_HEIGHT)
field_mapper = FieldMapper(pitch_width=PITCH_WIDTH, pitch_height=PITCH_HEIGHT)
video_fps = cap.get(cv2.CAP_PROP_FPS)
if not np.isfinite(video_fps) or video_fps <= 1:
    video_fps = 30.0
goal_detector = GoalDetector(fps=video_fps)

print("Object detector classes:", detector.model.names)
print("Landmark model classes:", landmark_detector.model.names)

frame_number = 0
last_landmark_observations = {}
last_calibration = None

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_number += 1

        # Run the object detector every frame.
        ball_detection, player_positions = detector.get_detections(
            frame, player_class_id=PLAYER_CLASS_ID
        )

        # Run the landmark model sparsely. On intervening frames, the
        # CameraMotionEstimator tracks semantic points with optical flow.
        run_landmark_model = (
            field_mapper.H_image_to_field is None
            or frame_number == 1
            or frame_number % LANDMARK_UPDATE_INTERVAL == 0
        )
        calibration = None
        if run_landmark_model:
            calibration = landmark_detector.calibrate(frame)
            last_landmark_observations = calibration["observations"]
            last_calibration = calibration

        camera_result = camera.update(
            frame,
            landmarks=last_landmark_observations if run_landmark_model else None,
        )
        camera_success = camera_result["success"]
        H_camera = camera_result["homography"] if camera_success else None

        # The old image->pitch homography is propagated into this frame
        # using the inverse of the previous->current camera transform.
        if camera_success and field_mapper.H_image_to_field is not None:
            field_mapper.update_camera_motion(H_camera)

        # A fresh landmark calibration is absolute, so it corrects any
        # accumulated camera-motion drift. Do not overwrite with failed H.
        if calibration is not None and calibration["success"]:
            field_mapper.H_image_to_field = calibration["homography"].copy()

        raw_prediction = get_raw_prediction(ball_tracker)
        compensated_prediction = get_compensated_prediction(
            ball_tracker, H_camera
        )
        if raw_prediction is not None and H_camera is not None:
            raw_prediction = transform_point(raw_prediction, H_camera)

        ball_position = ball_tracker.update(
            ball_detection, camera_H=H_camera
        )
        player_tracks = player_tracker.update(
            player_positions, camera_H=H_camera
        )

        # Map player ground-contact points (bbox bottom-center) to the pitch.
        player_field_positions = []
        for track in player_tracks:
            mapped = map_track_to_field(track["position"], field_mapper)
            if mapped is not None:
                player_field_positions.append({
                    "id": track["id"],
                    "position": mapped,
                    "detected": track["detected"],
                    "missed": track["missed"],
                })

        # A ball center is not on the ground plane while airborne. This
        # simple visualization still maps the detector/tracker center;
        # interpret its pitch position cautiously during aerial flight.
        ball_field_position = (
            map_track_to_field(ball_position, field_mapper)
            if ball_position is not None else None
        )

        # Goal detection uses only a CURRENT YOLO ball observation for its
        # trajectory history; Kalman-only predictions are never treated as
        # observed ball positions. This avoids confirming goals from drift.
        observed_ball_field_position = (
            map_track_to_field(ball_detection, field_mapper)
            if ball_detection is not None else None
        )
        goal_result = goal_detector.update(
            frame_number=frame_number,
            observed_ball_field_position=observed_ball_field_position,
            player_field_positions=player_field_positions,
            ball_missed_frames=ball_tracker.missed_frames,
        )

        camera_view = frame.copy()
        if SHOW_LANDMARKS and last_landmark_observations:
            camera_view = draw_landmarks(camera_view, last_landmark_observations)

        if SHOW_CAMERA_MOTION:
            camera_view = camera_visualizer.visualize(
                frame=camera_view,
                camera_result=camera_result,
                detection=ball_detection,
                raw_prediction=raw_prediction,
                compensated_prediction=compensated_prediction,
            )

        for x, y in player_positions:
            cv2.circle(camera_view, (int(x), int(y)), 4, (255, 100, 0), -1)

        for track in player_tracks:
            x, y = track["position"]
            color = (0, 255, 0) if track["detected"] else (0, 255, 255)
            cv2.putText(
                camera_view, f"Player {track['id']}",
                (int(x) + 8, int(y) - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1,
            )

        if ball_position is not None:
            x, y = np.round(ball_position).astype(int)
            cv2.circle(camera_view, (x, y), 6, (0, 255, 0), -1)
            cv2.putText(
                camera_view, "Tracked ball", (x + 10, y + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2,
            )

        # Goal-event diagnostics on the camera view.
        status_color = (0, 255, 0) if goal_result["goal"] else (0, 200, 255)
        cv2.putText(
            camera_view,
            f"Goal detector: {goal_result['state']} | {goal_result['reason']}",
            (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.62, status_color, 2,
        )
        if goal_result["goal"]:
            cv2.putText(
                camera_view,
                f"GOAL CONFIRMED - {goal_result['goal_side'].upper()}",
                (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3,
            )
            print(
                f"GOAL CONFIRMED | frame={goal_result['event_frame']} | "
                f"side={goal_result['goal_side']} | "
                f"confidence={goal_result['confidence']:.2f}"
            )

        if SHOW_PITCH:
            pitch_view = pitch_visualizer.show(ball_field_position)
            for player in player_field_positions:
                fx, fy = player["position"]
                px = int(fx * (PITCH_WIDTH - 1))
                py = int(fy * (PITCH_HEIGHT - 1))
                color = (255, 140, 0) if player["detected"] else (0, 255, 255)
                cv2.circle(pitch_view, (px, py), 8, color, -1)
                cv2.putText(
                    pitch_view, str(player["id"]), (px + 10, py - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
                )
            cv2.putText(
                pitch_view,
                f"Goal: {goal_result['state']}",
                (15, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                status_color, 2,
            )
            cv2.putText(
                pitch_view,
                f"Players: {len(player_tracks)}",
                (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (255, 255, 255), 2,
            )
            cv2.imshow("Top Down Soccer Field", pitch_view)

        cv2.imshow("Camera Motion + Ball + Player Tracking", camera_view)

        if calibration is not None:
            print(
                f"Frame {frame_number}: landmark calibration "
                f"{'OK' if calibration['success'] else 'REJECTED'} | "
                f"points={len(calibration['observations'])} | "
                f"inliers={calibration['inliers']} | "
                f"mean_error={calibration['mean_error']} | "
                f"coverage={calibration['coverage']:.3f}"
            )

        if camera_success:
            motion = camera_result["motion"]
            print(
                f"Frame {frame_number} | camera={camera_result['source']} | "
                f"inliers={camera_result['inliers']} | "
                f"players={len(player_tracks)} | "
                f"pan=({motion['translation_x']:.1f},"
                f"{motion['translation_y']:.1f}) | "
                f"zoom={motion['zoom_percent']:+.2f}% | "
                f"rotation={motion['rotation_degrees']:.2f}"
            )
        else:
            print(
                f"Frame {frame_number} | camera estimation failed "
                f"({camera_result['source']}) | players={len(player_tracks)}"
            )

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("r"):
            print("Resetting object trackers and camera motion.")
            ball_tracker.reset()
            goal_detector.reset()
            player_tracker = PlayerTracker(
                max_distance=PLAYER_MAX_DISTANCE,
                max_missed=PLAYER_MAX_MISSED,
            )
            camera.reset()
            last_landmark_observations = {}
            field_mapper.H_image_to_field = None

finally:
    cap.release()
    cv2.destroyAllWindows()

print("Video processing complete.")
