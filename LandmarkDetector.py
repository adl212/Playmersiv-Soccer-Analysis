import cv2
import numpy as np
from ultralytics import YOLO


# The user's trained pose model uses one 19-keypoint schema shared by
# three detection classes. Indices are global within result.keypoints.data.
CLASS_KEYPOINT_SCHEMA = {
    0: (0, [
        "center-circle-top",
        "center-circle-right",
        "center-circle-bottom",
        "center-circle-left",
        "center",
    ]),
    1: (5, [
        "big-left-bottom-left",
        "big-left-bottom-right",
        "big-left-top-right",
        "big-left-top-left",
        "left-semicircle-left",
        "left-semicircle-right",
        "left-pk",
    ]),
    2: (12, [
        "big-right-top-left",
        "big-right-top-right",
        "big-right-bottom-right",
        "big-right-bottom-left",
        "right-semicircle-left",
        "right-semicircle-right",
        "right-pk",
    ]),
}

# Canonical pitch coordinates in percentage units, x from left to right,
# y from top to bottom. The penalty-area and penalty-spot coordinates
# preserve the coordinate convention previously used in this project.
# Semicircle/box-line intersections are calculated from the standard
# 9.15 m arc radius, 16.5 m penalty-area depth, and 11 m penalty spot.
ARC_Y_TOP = 50.0 - (np.sqrt(9.15 ** 2 - (16.5 - 11.0) ** 2) / 68.0) * 100.0
ARC_Y_BOTTOM = 100.0 - ARC_Y_TOP

FIELD_POINTS_PERCENT = {
    "center-circle-top": (50.0, 40.0),
    "center-circle-right": (60.0, 50.0),
    "center-circle-bottom": (50.0, 60.0),
    "center-circle-left": (40.0, 50.0),
    "center": (50.0, 50.0),

    "big-left-bottom-left": (18.0, 79.647),
    "big-left-bottom-right": (18.0, 20.353),
    "big-left-top-right": (0.0, 20.353),
    "big-left-top-left": (0.0, 79.647),
    "left-semicircle-left": (18.0, ARC_Y_BOTTOM),
    "left-semicircle-right": (18.0, ARC_Y_TOP),
    "left-pk": (12.0, 50.0),

    "big-right-top-left": (100.0, 20.353),
    "big-right-top-right": (100.0, 79.647),
    "big-right-bottom-right": (82.0, 79.647),
    "big-right-bottom-left": (82.0, 20.353),
    "right-semicircle-left": (82.0, ARC_Y_TOP),
    "right-semicircle-right": (82.0, ARC_Y_BOTTOM),
    "right-pk": (88.0, 50.0),
}


class LandmarkDetector:
    """YOLO pose landmark detector and image-to-pitch homography estimator."""

    def __init__(
        self,
        model_path="best_v7.pt",
        pitch_width=1050,
        pitch_height=680,
        confidence=0.20,
        keypoint_confidence=0.35,
        ransac_threshold=10.0,
        min_inliers=4,
        max_mean_error=20.0,
        min_pitch_coverage=0.01,
        device=None,
    ):
        self.model = YOLO(model_path)
        self.pitch_width = int(pitch_width)
        self.pitch_height = int(pitch_height)
        self.confidence = float(confidence)
        self.keypoint_confidence = float(keypoint_confidence)
        self.ransac_threshold = float(ransac_threshold)
        self.min_inliers = int(min_inliers)
        self.max_mean_error = float(max_mean_error)
        self.min_pitch_coverage = float(min_pitch_coverage)
        self.device = device

        missing = set(FIELD_POINTS_PERCENT) - set(
            name for _, names in CLASS_KEYPOINT_SCHEMA.values() for name in names
        )
        if missing:
            raise ValueError(f"Missing keypoint schema entries: {sorted(missing)}")

    def detect(self, frame):
        """Return detected landmarks as a name -> observation dictionary."""
        kwargs = {
            "source": frame,
            "conf": self.confidence,
            "verbose": False,
        }
        if self.device is not None:
            kwargs["device"] = self.device

        result = self.model.predict(**kwargs)[0]
        observations = {}

        if result.keypoints is None or result.boxes is None:
            return observations

        keypoints = result.keypoints.data
        classes = result.boxes.cls.detach().cpu().numpy().astype(int)
        box_conf = result.boxes.conf.detach().cpu().numpy()

        points = keypoints.detach().cpu().numpy()

        for detection_index, class_id in enumerate(classes):
            if class_id not in CLASS_KEYPOINT_SCHEMA:
                continue

            start, names = CLASS_KEYPOINT_SCHEMA[class_id]
            det_points = points[detection_index]

            for local_index, name in enumerate(names):
                keypoint_index = start + local_index
                if keypoint_index >= len(det_points):
                    continue

                x, y = float(det_points[keypoint_index][0]), float(det_points[keypoint_index][1])
                kp_conf = (
                    float(det_points[keypoint_index][2])
                    if det_points.shape[-1] >= 3
                    else float(box_conf[detection_index])
                )
                score = min(kp_conf, float(box_conf[detection_index]))

                if (
                    score < self.keypoint_confidence
                    or not np.isfinite(x)
                    or not np.isfinite(y)
                ):
                    continue

                # If duplicate detections predict the same semantic point,
                # retain the highest-confidence observation.
                previous = observations.get(name)
                if previous is None or score > previous["score"]:
                    observations[name] = {
                        "name": name,
                        "image_point": np.array([x, y], dtype=np.float32),
                        "score": score,
                        "class_id": int(class_id),
                    }

        return observations

    def correspondences(self, observations):
        image_points, field_points, names = [], [], []
        for name, observation in observations.items():
            if name not in FIELD_POINTS_PERCENT:
                continue
            x_percent, y_percent = FIELD_POINTS_PERCENT[name]
            image_points.append(observation["image_point"])
            field_points.append([
                x_percent / 100.0 * self.pitch_width,
                y_percent / 100.0 * self.pitch_height,
            ])
            names.append(name)

        return (
            np.asarray(image_points, dtype=np.float32).reshape(-1, 2),
            np.asarray(field_points, dtype=np.float32).reshape(-1, 2),
            names,
        )

    def estimate_homography(self, image_points, field_points):
        """Estimate H: image pixels -> canonical pitch pixels."""
        image_points = np.asarray(image_points, dtype=np.float32).reshape(-1, 2)
        field_points = np.asarray(field_points, dtype=np.float32).reshape(-1, 2)

        if len(image_points) < self.min_inliers:
            return None, 0, None, None, 0.0

        H, inlier_mask = cv2.findHomography(
            image_points,
            field_points,
            cv2.RANSAC,
            self.ransac_threshold,
        )
        if H is None or inlier_mask is None or not np.all(np.isfinite(H)):
            return None, 0, None, None, 0.0

        if abs(H[2, 2]) < 1e-10:
            return None, 0, None, None, 0.0

        H = H / H[2, 2]
        inliers = inlier_mask.ravel().astype(bool)
        inlier_count = int(inliers.sum())
        if inlier_count < self.min_inliers:
            return None, inlier_count, None, inliers, 0.0

        projected = cv2.perspectiveTransform(
            image_points.reshape(-1, 1, 2), H
        ).reshape(-1, 2)
        errors = np.linalg.norm(projected - field_points, axis=1)
        mean_error = float(np.mean(errors[inliers]))

        # Reject degenerate sets clustered in a tiny part of the pitch.
        hull = cv2.convexHull(field_points[inliers].astype(np.float32))
        hull_area = float(cv2.contourArea(hull))
        coverage = hull_area / float(self.pitch_width * self.pitch_height)

        if mean_error > self.max_mean_error or coverage < self.min_pitch_coverage:
            return None, inlier_count, mean_error, inliers, coverage

        return H, inlier_count, mean_error, inliers, coverage

    def calibrate(self, frame, observations=None):
        """Detect (unless supplied), then estimate a field homography."""
        if observations is None:
            observations = self.detect(frame)

        image_points, field_points, names = self.correspondences(observations)
        H, inliers, mean_error, inlier_mask, coverage = self.estimate_homography(
            image_points, field_points
        )

        return {
            "observations": observations,
            "image_points": image_points,
            "field_points": field_points,
            "names": names,
            "homography": H,
            "inliers": inliers,
            "mean_error": mean_error,
            "inlier_mask": inlier_mask,
            "coverage": coverage,
            "success": H is not None,
        }

    @staticmethod
    def draw(frame, observations):
        output = frame.copy()
        for name, item in observations.items():
            x, y = np.round(item["image_point"]).astype(int)
            cv2.circle(output, (x, y), 5, (0, 255, 255), -1)
            cv2.putText(
                output, name, (x + 5, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1
            )
        return output
