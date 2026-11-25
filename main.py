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
import time
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

FIRST_FRAME = 95000#122900
PLAYER_CLASS_ID = 0
PITCH_WIDTH, PITCH_HEIGHT = 1050, 680
BALL_CONFIDENCE = 0.5
LANDMARK_CONFIDENCE = 0.20
KEYPOINT_CONFIDENCE = 0.35
LANDMARK_UPDATE_INTERVAL = 15
PROFILE_EVERY_N_FRAMES = 30

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

profile_stage_names = (
    "read", "detector", "landmarks", "camera", "tracking_mapping",
    "render", "console", "waitKey", "total",
)
profile_totals = dict.fromkeys(profile_stage_names, 0.0)
profile_frames = 0


def record_profile_stage(name, stage_start):
    profile_totals[name] += time.perf_counter() - stage_start


def report_profile():
    if profile_frames == 0:
        return
    averages_ms = {
        name: profile_totals[name] * 1000 / profile_frames
        for name in profile_stage_names
    }
    average_frame_ms = averages_ms["total"]
    approximate_fps = 1000 / average_frame_ms if average_frame_ms > 0 else 0
    details = " | ".join(
        f"{name}={averages_ms[name]:.1f}ms"
        for name in profile_stage_names
        if name != "total"
    )
    print(
        f"[PROFILE] avg over {profile_frames} frames: {details} | "
        f"total={average_frame_ms:.1f}ms ({approximate_fps:.1f} FPS)"
    )

frame_number = 0
last_landmark_observations = {}
last_calibration = None
last_goal_event = None

try:
    while True:
        frame_start = time.perf_counter()
        stage_start = frame_start
        ret, frame = cap.read()
        if not ret:
            break
        record_profile_stage("read", stage_start)
        frame_number += 1
        profile_frames += 1

        # Run the object detector every frame.
        stage_start = time.perf_counter()
        ball_detection, player_positions = detector.get_detections(
            frame, player_class_id=PLAYER_CLASS_ID
        )
        record_profile_stage("detector", stage_start)

        # Run the landmark model sparsely. On intervening frames, the
        # CameraMotionEstimator tracks semantic points with optical flow.
        run_landmark_model = (
            field_mapper.H_image_to_field is None
            or frame_number == 1
            or frame_number % LANDMARK_UPDATE_INTERVAL == 0
        )
        calibration = None
        stage_start = time.perf_counter()
        if run_landmark_model:
            calibration = landmark_detector.calibrate(frame)
            last_landmark_observations = calibration["observations"]
            last_calibration = calibration
        record_profile_stage("landmarks", stage_start)

        stage_start = time.perf_counter()
        camera_result = camera.update(
            frame,
            landmarks=last_landmark_observations if run_landmark_model else None,
        )
        record_profile_stage("camera", stage_start)
        camera_success = camera_result["success"]
        H_camera = camera_result["homography"] if camera_success else None

        stage_start = time.perf_counter()
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
            ball_missed_frames=ball_tracker.missed_frames,
        )
        if goal_result["goal"]:
            last_goal_event = {
                "side": goal_result["goal_side"],
                "frame": goal_result["event_frame"],
                "until": frame_number + int(3.0 * video_fps),
                "confidence": goal_result["confidence"],
            }
        goal_display_active = (
            last_goal_event is not None
            and frame_number <= last_goal_event["until"]
        )
        goal_label = (
            f"GOAL CONFIRMED - {last_goal_event['side'].upper()}"
            if goal_display_active else f"Goal detector: {goal_result['state']}"
        )
        record_profile_stage("tracking_mapping", stage_start)

        stage_start = time.perf_counter()
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
        status_color = (0, 255, 0) if goal_display_active else (0, 200, 255)
        cv2.putText(
            camera_view,
            f"{goal_label} | {goal_result['reason']}",
            (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.62, status_color, 2,
        )
        if goal_result["goal"]:
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
                f"{goal_label}",
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
        record_profile_stage("render", stage_start)

        stage_start = time.perf_counter()
        if calibration is not None:
            print(
                f"Frame {frame_number}: landmark calibration "
                f"{'OK' if calibration['success'] else 'REJECTED'} | "
                f"points={len(calibration['observations'])} | "
                f"inliers={calibration['inliers']} | "
                f"mean_error={calibration['mean_error']} | "
                f"coverage={calibration['coverage']:.3f}"
                f"time={time.perf_counter() - frame_start:.3f}s"
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
                f"time={time.perf_counter() - frame_start:.3f}s"
            )
        else:
            print(
                f"Frame {frame_number} | camera estimation failed "
                f"({camera_result['source']}) | players={len(player_tracks)}"
            )
        record_profile_stage("console", stage_start)

        stage_start = time.perf_counter()
        key = cv2.waitKey(1) & 0xFF
        record_profile_stage("waitKey", stage_start)
        record_profile_stage("total", frame_start)

        if frame_number % PROFILE_EVERY_N_FRAMES == 0:
            report_profile()
            for name in profile_stage_names:
                profile_totals[name] = 0.0
            profile_frames = 0

        if key == ord("q"):
            break
        if key == ord("r"):
            print("Resetting object trackers and camera motion.")
            ball_tracker.reset()
            goal_detector.reset()
            last_goal_event = None
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

report_profile()
print("Video processing complete.")
