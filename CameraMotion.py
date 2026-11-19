
import cv2
import numpy as np


class CameraMotionEstimator:
    """
    Estimates previous-frame -> current-frame camera motion.

    Estimation priority:
        1. Fresh semantic pitch landmark correspondences.
        2. Tracked semantic pitch landmarks.
        3. General optical-flow feature correspondences.

    cumulative_H maps the current frame into the initial frame.
    """

    def __init__(
        self,
        max_features=1500,
        quality_level=0.01,
        min_distance=8,
        ransac_threshold=3.0,
        min_inliers=12,
        feature_interval=10,
        min_landmark_inliers=4,
        landmark_ransac_threshold=5.0,
        landmark_min_coverage=0.001,
        landmark_fb_threshold=1.5,
    ):
        self.max_features = max_features
        self.quality_level = quality_level
        self.min_distance = min_distance
        self.ransac_threshold = ransac_threshold
        self.min_inliers = min_inliers
        self.feature_interval = feature_interval

        self.min_landmark_inliers = min_landmark_inliers
        self.landmark_ransac_threshold = landmark_ransac_threshold
        self.landmark_min_coverage = landmark_min_coverage
        self.landmark_fb_threshold = landmark_fb_threshold

        self.prev_gray = None
        self.prev_pts = None
        self.prev_mask = None

        # Semantic landmark name -> previous-frame image coordinate.
        self.prev_landmarks = {}

        self.frame_count = 0

        self.cumulative_H = np.eye(3, dtype=np.float64)

        self.last_H = np.eye(3, dtype=np.float64)
        self.last_inliers = 0
        self.last_success = False
        self.last_source = "none"

    # ========================================================
    # BASIC UTILITIES
    # ========================================================

    def _to_gray(self, frame):

        if len(frame.shape) == 3:
            return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        return frame.copy()

    def _detect_features(self, gray, mask=None):

        return cv2.goodFeaturesToTrack(
            gray,
            maxCorners=self.max_features,
            qualityLevel=self.quality_level,
            minDistance=self.min_distance,
            mask=mask,
            blockSize=7
        )

    def _parse_landmarks(self, landmarks):
        """
        Convert LandmarkDetector output into:

            {
                landmark_name: (x, y)
            }
        """

        parsed = {}

        if landmarks is None:
            return parsed

        for landmark in landmarks:

            name = landmark.get("name")
            point = landmark.get("image_point")

            if name is None or point is None:
                continue

            point = np.asarray(point, dtype=np.float32)

            if point.shape != (2,):
                continue

            if not np.all(np.isfinite(point)):
                continue

            parsed[name] = (
                float(point[0]),
                float(point[1])
            )

        return parsed

    # ========================================================
    # SEMANTIC LANDMARK TRACKING
    # ========================================================

    def _track_landmarks(self, gray):
        """
        Track semantic pitch landmarks from the previous
        frame into the current frame using Lucas-Kanade.
        """

        if self.prev_gray is None or not self.prev_landmarks:
            return {}

        names = list(self.prev_landmarks.keys())

        old_points = np.array(
            [self.prev_landmarks[name] for name in names],
            dtype=np.float32
        ).reshape(-1, 1, 2)

        current_points, status, _ = cv2.calcOpticalFlowPyrLK(
            self.prev_gray,
            gray,
            old_points,
            None,
            winSize=(21, 21),
            maxLevel=3,
            criteria=(
                cv2.TERM_CRITERIA_EPS |
                cv2.TERM_CRITERIA_COUNT,
                30,
                0.01
            )
        )

        if current_points is None or status is None:
            return {}

        current_points = current_points.reshape(-1, 2)

        valid = status.flatten() == 1

        # Forward-backward consistency check.
        backward_points, backward_status, _ = (
            cv2.calcOpticalFlowPyrLK(
                gray,
                self.prev_gray,
                current_points.reshape(-1, 1, 2),
                None,
                winSize=(21, 21),
                maxLevel=3
            )
        )

        if backward_points is None or backward_status is None:
            return {}

        backward_points = backward_points.reshape(-1, 2)

        valid &= backward_status.flatten() == 1

        old_flat = old_points.reshape(-1, 2)

        fb_errors = np.linalg.norm(
            old_flat - backward_points,
            axis=1
        )

        valid &= fb_errors < self.landmark_fb_threshold

        tracked = {}

        for i, name in enumerate(names):

            if not valid[i]:
                continue

            point = current_points[i]

            if not np.all(np.isfinite(point)):
                continue

            tracked[name] = (
                float(point[0]),
                float(point[1])
            )

        return tracked

    # ========================================================
    # LANDMARK-BASED HOMOGRAPHY
    # ========================================================

    def _estimate_landmark_homography(
        self,
        previous_landmarks,
        current_landmarks,
        image_shape
    ):
        """
        Estimate previous-frame -> current-frame homography
        using matching semantic landmark names.
        """

        common_names = sorted(
            set(previous_landmarks.keys()) &
            set(current_landmarks.keys())
        )

        if len(common_names) < self.min_landmark_inliers:
            return None, 0

        previous_points = np.array(
            [previous_landmarks[name] for name in common_names],
            dtype=np.float32
        )

        current_points = np.array(
            [current_landmarks[name] for name in common_names],
            dtype=np.float32
        )

        # Reject spatially concentrated configurations.
        hull = cv2.convexHull(previous_points)

        hull_area = cv2.contourArea(hull)

        image_area = image_shape[0] * image_shape[1]

        if hull_area < image_area * self.landmark_min_coverage:
            return None, 0

        previous_points = previous_points.reshape(-1, 1, 2)
        current_points = current_points.reshape(-1, 1, 2)

        H, mask = cv2.findHomography(
            previous_points,
            current_points,
            cv2.RANSAC,
            self.landmark_ransac_threshold
        )

        if H is None or mask is None:
            return None, 0

        inlier_mask = mask.flatten().astype(bool)

        inliers = int(inlier_mask.sum())

        if inliers < self.min_landmark_inliers:
            return None, inliers

        if inliers / len(common_names) < 0.5:
            return None, inliers

        if not np.all(np.isfinite(H)):
            return None, inliers

        if abs(H[2, 2]) < 1e-8:
            return None, inliers

        H = H / H[2, 2]

        projected = cv2.perspectiveTransform(
            previous_points,
            H
        ).reshape(-1, 2)

        actual = current_points.reshape(-1, 2)

        errors = np.linalg.norm(
            projected[inlier_mask] - actual[inlier_mask],
            axis=1
        )

        if len(errors) == 0:
            return None, inliers

        if not np.all(np.isfinite(errors)):
            return None, inliers

        if np.median(errors) > self.landmark_ransac_threshold:
            return None, inliers

        return H, inliers

    # ========================================================
    # GENERAL OPTICAL-FLOW FALLBACK
    # ========================================================

    def _estimate_optical_flow(self, gray):
        """
        Estimate camera motion using general image features.

        Returns:
            H, inlier_count, current_feature_points
        """

        if (
            self.prev_pts is None
            or len(self.prev_pts) < self.min_inliers * 2
            or self.frame_count % self.feature_interval == 0
        ):
            # Important: use the previous frame's mask,
            # not the current frame's mask.
            self.prev_pts = self._detect_features(
                self.prev_gray,
                self.prev_mask
            )

        if self.prev_pts is None or len(self.prev_pts) < 8:
            return None, 0, None

        current_points, status, _ = cv2.calcOpticalFlowPyrLK(
            self.prev_gray,
            gray,
            self.prev_pts,
            None,
            winSize=(21, 21),
            maxLevel=3,
            criteria=(
                cv2.TERM_CRITERIA_EPS |
                cv2.TERM_CRITERIA_COUNT,
                30,
                0.01
            )
        )

        if current_points is None or status is None:
            return None, 0, None

        valid = status.flatten() == 1

        old = self.prev_pts[valid].reshape(-1, 2)
        new = current_points[valid].reshape(-1, 2)

        if len(old) < 8:
            return None, 0, None

        # Forward-backward consistency.
        backward_points, backward_status, _ = (
            cv2.calcOpticalFlowPyrLK(
                gray,
                self.prev_gray,
                new.reshape(-1, 1, 2),
                None
            )
        )

        if backward_points is None or backward_status is None:
            return None, 0, None

        backward_valid = backward_status.flatten() == 1

        old = old[backward_valid]
        new = new[backward_valid]

        backward_points = (
            backward_points.reshape(-1, 2)[backward_valid]
        )

        fb_errors = np.linalg.norm(
            old - backward_points,
            axis=1
        )

        good = fb_errors < 1.5

        old = old[good]
        new = new[good]

        if len(old) < 8:
            return None, 0, None

        H, mask = cv2.findHomography(
            old,
            new,
            cv2.RANSAC,
            self.ransac_threshold
        )

        if H is None or mask is None:
            return None, 0, None

        inliers = int(mask.sum())

        if inliers < self.min_inliers:
            return None, inliers, None

        if inliers / len(old) < 0.25:
            return None, inliers, None

        if not np.all(np.isfinite(H)):
            return None, inliers, None

        if abs(H[2, 2]) < 1e-8:
            return None, inliers, None

        H = H / H[2, 2]

        return H, inliers, new.reshape(-1, 1, 2)

    # ========================================================
    # MOTION DESCRIPTION
    # ========================================================

    def _extract_motion(self, H):

        A = H[:2, :2]

        scale = np.sqrt(abs(np.linalg.det(A)))

        rotation = np.degrees(
            np.arctan2(A[1, 0], A[0, 0])
        )

        return {
            "translation_x": float(H[0, 2]),
            "translation_y": float(H[1, 2]),
            "scale": float(scale),
            "zoom_percent": float((scale - 1.0) * 100),
            "rotation_degrees": float(rotation)
        }

    # ========================================================
    # MAIN UPDATE
    # ========================================================

    def update(self, frame, mask=None, landmarks=None):
        """
        Process a frame.

        Args:
            frame: BGR or grayscale image.
            mask: Optional feature-detection mask.
            landmarks: Optional list of semantic landmark
                       dictionaries from LandmarkDetector.

        Returns:
            homography: Previous frame -> current frame.
            source: landmarks-detected, landmarks-tracked,
                    optical-flow, or none.
        """

        gray = self._to_gray(frame)

        observed_landmarks = self._parse_landmarks(landmarks)

        # ----------------------------------------------------
        # Initialize.
        # ----------------------------------------------------

        if self.prev_gray is None:

            self.prev_gray = gray

            self.prev_mask = (
                mask.copy() if mask is not None else None
            )

            self.prev_pts = self._detect_features(
                gray,
                mask
            )

            self.prev_landmarks = observed_landmarks

            return self._result(
                np.eye(3),
                False,
                0,
                "none"
            )

        # ----------------------------------------------------
        # Track existing semantic landmarks.
        # ----------------------------------------------------

        tracked_landmarks = self._track_landmarks(gray)

        H = None
        inliers = 0
        source = "none"
        current_feature_points = None

        # Fresh detections take precedence over tracked
        # coordinates for the same semantic landmark.
        current_landmarks = tracked_landmarks.copy()
        current_landmarks.update(observed_landmarks)

        # ----------------------------------------------------
        # 1. Fresh detections + tracked semantic landmarks.
        # ----------------------------------------------------

        if observed_landmarks and self.prev_landmarks:

            H, inliers = self._estimate_landmark_homography(
                self.prev_landmarks,
                current_landmarks,
                gray.shape
            )

            if H is not None:
                source = "landmarks-detected"

        # ----------------------------------------------------
        # 2. Tracked semantic landmarks only.
        # ----------------------------------------------------

        if H is None and tracked_landmarks:

            H, inliers = self._estimate_landmark_homography(
                self.prev_landmarks,
                tracked_landmarks,
                gray.shape
            )

            if H is not None:
                source = "landmarks-tracked"

        # ----------------------------------------------------
        # 3. General optical-flow fallback.
        # ----------------------------------------------------

        if H is None:

            H, inliers, current_feature_points = (
                self._estimate_optical_flow(gray)
            )

            if H is not None:
                source = "optical-flow"

        # ----------------------------------------------------
        # Update semantic landmark state.
        # ----------------------------------------------------

        self.prev_landmarks = current_landmarks

        # ----------------------------------------------------
        # Handle failure.
        # ----------------------------------------------------

        if H is None:

            self.prev_gray = gray

            self.prev_mask = (
                mask.copy() if mask is not None else None
            )

            self.prev_pts = self._detect_features(
                gray,
                mask
            )

            self.last_success = False
            self.last_inliers = inliers
            self.last_source = "none"

            self.frame_count += 1

            return self._result(
                np.eye(3),
                False,
                inliers,
                "none"
            )

        # ----------------------------------------------------
        # Successful camera estimation.
        # ----------------------------------------------------

        H = H / H[2, 2]

        self.cumulative_H = (
            self.cumulative_H @ np.linalg.inv(H)
        )

        self.last_H = H
        self.last_inliers = inliers
        self.last_success = True
        self.last_source = source

        self.prev_gray = gray

        self.prev_mask = (
            mask.copy() if mask is not None else None
        )

        if current_feature_points is not None:
            self.prev_pts = current_feature_points
        else:
            self.prev_pts = self._detect_features(
                gray,
                mask
            )

        self.frame_count += 1

        return self._result(
            H,
            True,
            inliers,
            source
        )

    # ========================================================
    # FAILURE / RESULTS
    # ========================================================

    def _result(self, H, success, inliers, source):

        return {
            "homography": H,
            "cumulative_homography": self.cumulative_H.copy(),
            "motion": self._extract_motion(H),
            "inliers": inliers,
            "success": success,
            "source": source
        }

    def reset(self):

        self.prev_gray = None
        self.prev_pts = None
        self.prev_mask = None
        self.prev_landmarks = {}

        self.frame_count = 0

        self.cumulative_H = np.eye(3, dtype=np.float64)

        self.last_H = np.eye(3, dtype=np.float64)
        self.last_inliers = 0
        self.last_success = False
        self.last_source = "none"
