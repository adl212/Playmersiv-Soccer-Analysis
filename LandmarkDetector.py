
import cv2
import numpy as np
from ultralytics import YOLO


# ============================================================
# KEYPOINT SCHEMA
# ============================================================

CLASS_KEYPOINT_SCHEMA = {
    0: (
        0,
        [
            "center-circle-top",
            "center-circle-right",
            "center-circle-bottom",
            "center-circle-left",
            "center",
        ],
    ),
    1: (
        5,
        [
            "big-left-bottom-left",
            "big-left-bottom-right",
            "big-left-top-right",
            "big-left-top-left",
            "left-semicircle-left",
            "left-semicircle-right",
            "left-pk",
        ],
    ),
    2: (
        12,
        [
            "big-right-top-left",
            "big-right-top-right",
            "big-right-bottom-right",
            "big-right-bottom-left",
            "right-semicircle-left",
            "right-semicircle-right",
            "right-pk",
        ],
    ),
}


# ============================================================
# CANONICAL PITCH LANDMARKS
# ============================================================

# Coordinates are percentages (0-100), not meters.
#
# The orientation is:
#   x = 0: left goal line
#   x = 100: right goal line
#   y = 0: top touchline
#   y = 100: bottom touchline
#
# The penalty-arc endpoints are deliberately excluded until
# their left/right physical correspondence is verified.

FIELD_POINTS_PERCENT = {
    # Center circle
    "center-circle-top": (50.0, 40.0),
    "center-circle-right": (60.0, 50.0),
    "center-circle-bottom": (50.0, 60.0),
    "center-circle-left": (40.0, 50.0),
    "center": (50.0, 50.0),

    # Left penalty box
    "big-left-bottom-left": (18, 79.647),
    "big-left-bottom-right": (18, 20.353),
    "big-left-top-right": (0.0, 20.353),
    "big-left-top-left": (0.0, 79.647),
    "left-pk": (12, 50.0),

    # Right penalty box
    "big-right-top-left": (100.0, 20.353),
    "big-right-top-right": (100.0, 79.647),
    "big-right-bottom-right": (82, 79.647),
    "big-right-bottom-left": (82, 20.353),
    "right-pk": (88, 50.0),
}


class LandmarkDetector:

    def __init__(
        self,
        model_path,
        pitch_width=1050,
        pitch_height=680,
        inference_confidence=0.15,
        center_keypoint_confidence=0.50,
        penalty_keypoint_confidence=0.80,
        penalty_box_detection_confidence=0.70,
        ransac_threshold=10.0,
        min_inliers=4,
        max_mean_error=20.0,
        min_pitch_coverage=0.01,
    ):

        self.model = YOLO(model_path)

        self.pitch_width = pitch_width
        self.pitch_height = pitch_height

        self.inference_confidence = inference_confidence

        self.center_keypoint_confidence = (
            center_keypoint_confidence
        )

        self.penalty_keypoint_confidence = (
            penalty_keypoint_confidence
        )

        self.penalty_box_detection_confidence = (
            penalty_box_detection_confidence
        )

        self.ransac_threshold = ransac_threshold
        self.min_inliers = min_inliers
        self.max_mean_error = max_mean_error
        self.min_pitch_coverage = min_pitch_coverage

        # Convert normalized 0-100 coordinates into the
        # pixel-coordinate system expected by FieldMapper.
        self.field_points = {
            name: np.array(
                [
                    x * self.pitch_width / 100.0,
                    y * self.pitch_height / 100.0,
                ],
                dtype=np.float32,
            )
            for name, (x, y) in FIELD_POINTS_PERCENT.items()
        }

    # ========================================================
    # MODEL INFERENCE
    # ========================================================

    def detect(self, frame):

        result = self.model.predict(
            frame,
            conf=self.inference_confidence,
            verbose=False,
        )[0]

        return result

    # ========================================================
    # EXTRACT IMAGE-TO-PITCH CORRESPONDENCES
    # ========================================================

    def extract_correspondences(self, result):

        if result.boxes is None or result.keypoints is None:

            return (
                np.empty((0, 2), dtype=np.float32),
                np.empty((0, 2), dtype=np.float32),
                [],
            )

        class_ids = result.boxes.cls.int().tolist()
        box_confidences = result.boxes.conf.tolist()
        keypoints_data = result.keypoints.data

        # Store only the strongest prediction for each
        # physical landmark.
        best_predictions = {}

        for instance_idx, class_id in enumerate(class_ids):

            if class_id not in CLASS_KEYPOINT_SCHEMA:
                continue

            box_confidence = box_confidences[instance_idx]

            # Penalty-box detections must have stronger
            # bounding-box confidence.
            if class_id in (1, 2):

                if (
                    box_confidence
                    < self.penalty_box_detection_confidence
                ):
                    continue

                required_keypoint_confidence = (
                    self.penalty_keypoint_confidence
                )

            else:

                required_keypoint_confidence = (
                    self.center_keypoint_confidence
                )

            start, names = CLASS_KEYPOINT_SCHEMA[class_id]

            keypoints = keypoints_data[instance_idx]

            for local_idx, name in enumerate(names):

                # Ignore penalty-arc endpoints until their
                # physical correspondence is verified.
                if name not in self.field_points:
                    continue

                global_idx = start + local_idx

                x, y, keypoint_confidence = (
                    keypoints[global_idx].tolist()
                )

                if (
                    keypoint_confidence
                    < required_keypoint_confidence
                ):
                    continue

                if not np.isfinite(x) or not np.isfinite(y):
                    continue

                if x == 0 and y == 0:
                    continue

                # Combine box and keypoint confidence for
                # duplicate landmark predictions.
                score = (
                    box_confidence * keypoint_confidence
                )

                if (
                    name not in best_predictions
                    or score > best_predictions[name]["score"]
                ):

                    best_predictions[name] = {
                        "score": score,
                        "image_point": (x, y),
                        "class_id": class_id,
                        "keypoint_confidence": keypoint_confidence,
                        "box_confidence": box_confidence,
                    }

        image_points = []
        field_points = []
        detected_landmarks = []

        for name, prediction in best_predictions.items():

            image_point = prediction["image_point"]

            image_points.append(image_point)
            field_points.append(self.field_points[name])

            detected_landmarks.append({
                "name": name,
                "image_point": image_point,
                "score": prediction["score"],
                "class_id": prediction["class_id"],
            })

        return (
            np.asarray(image_points, dtype=np.float32).reshape(-1, 2),
            np.asarray(field_points, dtype=np.float32).reshape(-1, 2),
            detected_landmarks,
        )

    # ========================================================
    # HOMOGRAPHY ESTIMATION
    # ========================================================

    def estimate_homography(
        self,
        image_points,
        field_points,
    ):

        if len(image_points) < self.min_inliers:

            return None, 0, None

        H, mask = cv2.findHomography(
            image_points,
            field_points,
            cv2.RANSAC,
            self.ransac_threshold,
        )

        if H is None or mask is None:
            return None, 0, None

        if not np.all(np.isfinite(H)):
            return None, 0, None

        inlier_mask = mask.ravel().astype(bool)

        inlier_count = int(inlier_mask.sum())

        if inlier_count < self.min_inliers:
            return None, inlier_count, None

        # Evaluate reprojection error using RANSAC inliers.
        projected = cv2.perspectiveTransform(
            image_points.reshape(-1, 1, 2),
            H,
        ).reshape(-1, 2)

        errors = np.linalg.norm(
            projected - field_points,
            axis=1,
        )

        mean_error = float(errors[inlier_mask].mean())

        if not np.isfinite(mean_error):
            return None, inlier_count, mean_error

        if mean_error > self.max_mean_error:
            return None, inlier_count, mean_error

        # Reject geometrically concentrated configurations.
        # This checks the spread of the corresponding canonical
        # pitch points, not their image-space distribution.
        inlier_field_points = field_points[inlier_mask]

        hull = cv2.convexHull(
            inlier_field_points.reshape(-1, 1, 2)
        )

        hull_area = cv2.contourArea(hull)

        total_pitch_area = (
            self.pitch_width * self.pitch_height
        )

        pitch_coverage = hull_area / total_pitch_area

        if pitch_coverage < self.min_pitch_coverage:
            return None, inlier_count, mean_error

        if abs(H[2, 2]) > 1e-8:
            H = H / H[2, 2]

        return H, inlier_count, mean_error

    # ========================================================
    # COMPLETE CALIBRATION
    # ========================================================

    def calibrate(self, frame):

        result = self.detect(frame)

        (
            image_points,
            field_points,
            detected_landmarks,
        ) = self.extract_correspondences(result)

        H, inlier_count, mean_error = (
            self.estimate_homography(
                image_points,
                field_points,
            )
        )

        return (
            result,
            H,
            inlier_count,
            mean_error,
            detected_landmarks,
        )

    # ========================================================
    # PROJECT DETECTED LANDMARKS ONTO THE TOP-DOWN PITCH
    # ========================================================

    def project_landmarks_to_pitch(
        self,
        detected_landmarks,
        H,
    ):

        if H is None:
            return []

        projected_landmarks = []

        for landmark in detected_landmarks:

            name = landmark["name"]
            image_point = landmark["image_point"]

            point = np.array(
                [[image_point]],
                dtype=np.float32,
            )

            projected = cv2.perspectiveTransform(
                point,
                H,
            )[0, 0]

            if not np.all(np.isfinite(projected)):
                continue

            # Convert to normalized 0-1 coordinates for
            # compatibility with FieldMapper output.
            normalized_x = (
                float(projected[0]) / self.pitch_width
            )

            normalized_y = (
                float(projected[1]) / self.pitch_height
            )

            if (
                0 <= normalized_x <= 1
                and 0 <= normalized_y <= 1
            ):

                projected_landmarks.append({
                    "name": name,
                    "position": (
                        normalized_x,
                        normalized_y,
                    ),
                })

        return projected_landmarks
