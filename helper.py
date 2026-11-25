import cv2
import numpy as np


def transform_point(point, H):
    """Apply a 3x3 homography to one image point."""
    if point is None or H is None:
        return None
    p = np.asarray(point, dtype=np.float32).reshape(1, 1, 2)
    out = cv2.perspectiveTransform(p, np.asarray(H, dtype=np.float64))
    result = out.reshape(2)
    return result if np.all(np.isfinite(result)) else None


def get_raw_prediction(ball_tracker):
    """One-step Kalman prediction in the previous frame's coordinates."""
    if not ball_tracker.initialized:
        return None
    return (
        ball_tracker.kf.transitionMatrix @ ball_tracker.kf.statePost
    )[:2].flatten()


def get_compensated_prediction(ball_tracker, H):
    """Predict ball position in current-frame coordinates without mutating KF."""
    if not ball_tracker.initialized:
        return None
    state = ball_tracker.kf.statePost.copy().astype(np.float64)
    if H is None:
        return (ball_tracker.kf.transitionMatrix @ state)[:2].flatten()

    x, y, vx, vy = state.flatten()
    H = np.asarray(H, dtype=np.float64)
    w = H[2, 0] * x + H[2, 1] * y + H[2, 2]
    if abs(w) < 1e-9:
        return None

    nx = (H[0, 0] * x + H[0, 1] * y + H[0, 2]) / w
    ny = (H[1, 0] * x + H[1, 1] * y + H[1, 2]) / w
    J = np.array([
        [(H[0, 0] - nx * H[2, 0]) / w, (H[0, 1] - nx * H[2, 1]) / w],
        [(H[1, 0] - ny * H[2, 0]) / w, (H[1, 1] - ny * H[2, 1]) / w],
    ])
    velocity = J @ np.array([vx, vy])
    transformed = np.array([[nx], [ny], [velocity[0]], [velocity[1]]], np.float32)
    return (ball_tracker.kf.transitionMatrix @ transformed)[:2].flatten()


def map_track_to_field(position, field_mapper):
    mapped = field_mapper.image_to_field(position)
    if mapped is None:
        return None
    x, y = mapped
    if np.isfinite(x) and np.isfinite(y) and 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0:
        return float(x), float(y)
    return None


def draw_landmarks(frame, observations, inlier_names=None):
    """Draw semantic keypoints; inliers are green, other detections yellow."""
    output = frame.copy()
    inlier_names = set(inlier_names or [])
    for name, item in observations.items():
        x, y = np.round(item["image_point"]).astype(int)
        color = (0, 255, 0) if name in inlier_names else (0, 255, 255)
        cv2.circle(output, (x, y), 5, color, -1)
        cv2.putText(output, name, (x + 5, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)
    return output
