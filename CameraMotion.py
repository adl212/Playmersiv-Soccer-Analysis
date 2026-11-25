import cv2
import numpy as np
import time


class CameraMotionEstimator:
    """
    Estimates previous-frame -> current-frame motion.

    Semantic pitch landmarks are preferred. Between landmark detector
    updates, those same landmarks are tracked with Lucas-Kanade flow.
    Generic optical-flow features are only a fallback.
    """

    def __init__(
        self,
        max_features=500,
        quality_level=0.01,
        min_distance=8,
        ransac_threshold=4.0,
        min_inliers=6,
        feature_interval=20,
        fb_threshold=1.5,
        min_inlier_ratio=0.5,
        min_image_coverage=0.0015,
        max_median_error=5.0,
        max_frame_motion_ratio=0.2,
        profile_interval=30,
    ):
        self.max_features = max_features
        self.quality_level = quality_level
        self.min_distance = min_distance
        self.ransac_threshold = ransac_threshold
        self.min_inliers = min_inliers
        self.feature_interval = feature_interval
        self.fb_threshold = fb_threshold
        self.min_inlier_ratio = min_inlier_ratio
        self.min_image_coverage = min_image_coverage
        self.max_median_error = max_median_error
        self.max_frame_motion_ratio = max_frame_motion_ratio
        self.profile_interval = max(1, int(profile_interval))
        self._profile_totals = {}
        self._profile_frames = 0
        self._dilation_kernels = {}

        self.prev_gray = None
        self.prev_pts = None
        self.prev_mask = None
        self.prev_landmarks = {}
        self.frame_count = 0
        self.cumulative_H = np.eye(3, dtype=np.float64)

    @staticmethod
    def _gray(frame):
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame.copy()

    @staticmethod
    def _clean_landmarks(landmarks):
        clean = {}
        if not landmarks:
            return clean
        for name, item in landmarks.items():
            point = item.get("image_point") if isinstance(item, dict) else item
            if point is None:
                continue
            point = np.asarray(point, dtype=np.float32).reshape(2)
            if np.all(np.isfinite(point)):
                clean[name] = point
        return clean

    def _feature_mask(self, gray, landmarks, supplied_mask):
        if supplied_mask is not None:
            return supplied_mask.copy()
        points = np.asarray(list(landmarks.values()), dtype=np.float32)
        if len(points) >= 3:
            hull = cv2.convexHull(points.reshape(-1, 1, 2).astype(np.int32))
            mask = np.zeros(gray.shape, dtype=np.uint8)
            cv2.fillConvexPoly(mask, hull, 255)
            # Expand the landmark hull to include nearby grass texture.
            radius = max(25, int(min(gray.shape) * 0.06))
            kernel = self._dilation_kernels.get(radius)
            if kernel is None:
                kernel = cv2.getStructuringElement(
                    cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1)
                )
                self._dilation_kernels[radius] = kernel
            return cv2.dilate(mask, kernel)
        return None

    def _detect_features(self, gray, mask):
        return cv2.goodFeaturesToTrack(
            gray,
            maxCorners=self.max_features,
            qualityLevel=self.quality_level,
            minDistance=self.min_distance,
            mask=mask,
            blockSize=7,
        )

    def _track_points(self, old_points, current_gray):
        if old_points is None or len(old_points) == 0:
            return np.empty((0, 2), np.float32), np.empty((0, 2), np.float32)

        old = np.asarray(old_points, dtype=np.float32).reshape(-1, 1, 2)
        new, status, _ = cv2.calcOpticalFlowPyrLK(
            self.prev_gray, current_gray, old, None,
            winSize=(21, 21), maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        )
        if new is None or status is None:
            return np.empty((0, 2), np.float32), np.empty((0, 2), np.float32)

        good = status.ravel() == 1
        old_good = old.reshape(-1, 2)[good]
        new_good = new.reshape(-1, 2)[good]
        if len(old_good) == 0:
            return old_good, new_good

        back, back_status, _ = cv2.calcOpticalFlowPyrLK(
            current_gray, self.prev_gray, new_good.reshape(-1, 1, 2), None,
            winSize=(21, 21), maxLevel=3,
        )
        if back is None or back_status is None:
            return np.empty((0, 2), np.float32), np.empty((0, 2), np.float32)

        back = back.reshape(-1, 2)
        valid = (back_status.ravel() == 1) & (
            np.linalg.norm(old_good - back, axis=1) < self.fb_threshold
        )
        return old_good[valid], new_good[valid]

    def _estimate(self, old_points, new_points, min_inliers=None):
        min_inliers = self.min_inliers if min_inliers is None else min_inliers
        if len(old_points) < min_inliers:
            return None, 0, None

        H, mask = cv2.findHomography(
            np.asarray(old_points, np.float32),
            np.asarray(new_points, np.float32),
            cv2.RANSAC,
            self.ransac_threshold,
        )
        if H is None or mask is None or not np.all(np.isfinite(H)) or abs(H[2, 2]) < 1e-10:
            return None, 0, None

        H = H / H[2, 2]
        inliers = mask.ravel().astype(bool)
        count = int(inliers.sum())
        ratio = count / max(1, len(old_points))
        if count < min_inliers or ratio < self.min_inlier_ratio:
            return None, count, None

        frame_height, frame_width = self.prev_gray.shape[:2]
        frame_corners = np.array(
            [[
                [0, 0],
                [frame_width - 1, 0],
                [frame_width - 1, frame_height - 1],
                [0, frame_height - 1],
            ]],
            dtype=np.float32,
        )
        projected_corners = cv2.perspectiveTransform(
            frame_corners, H
        ).reshape(-1, 2)
        corner_displacements = np.linalg.norm(
            projected_corners - frame_corners.reshape(-1, 2), axis=1
        )
        max_corner_displacement = float(np.max(corner_displacements))
        max_allowed_displacement = (
            np.hypot(frame_width, frame_height) * self.max_frame_motion_ratio
        )
        if (
            not np.all(np.isfinite(projected_corners))
            or max_corner_displacement > max_allowed_displacement
        ):
            return None, count, None

        hull = cv2.convexHull(np.asarray(new_points, np.float32)[inliers])
        coverage = cv2.contourArea(hull) / max(
            1.0, float(self.prev_gray.shape[0] * self.prev_gray.shape[1])
        )
        if coverage < self.min_image_coverage:
            return None, count, None

        projected = cv2.perspectiveTransform(
            np.asarray(old_points, np.float32).reshape(-1, 1, 2), H
        ).reshape(-1, 2)
        errors = np.linalg.norm(projected - np.asarray(new_points), axis=1)
        median_error = float(np.median(errors[inliers]))
        if not np.isfinite(median_error) or median_error > self.max_median_error:
            return None, count, median_error

        return H, count, median_error

    def _result(self, H, success, inliers, source, median_error=None):
        if success:
            A = H[:2, :2]
            scale = float(np.sqrt(abs(np.linalg.det(A))))
            rotation = float(np.degrees(np.arctan2(A[1, 0], A[0, 0])))
            motion = {
                "translation_x": float(H[0, 2]),
                "translation_y": float(H[1, 2]),
                "scale": scale,
                "zoom_percent": (scale - 1.0) * 100.0,
                "rotation_degrees": rotation,
            }
        else:
            motion = {
                "translation_x": 0.0, "translation_y": 0.0,
                "scale": 1.0, "zoom_percent": 0.0,
                "rotation_degrees": 0.0,
            }
        return {
            "homography": H.copy() if success else np.eye(3),
            "cumulative_homography": self.cumulative_H.copy(),
            "motion": motion,
            "inliers": int(inliers),
            "median_error": median_error,
            "source": source,
            "success": bool(success),
        }

    @staticmethod
    def _record_timing(timings, name, stage_start):
        timings[name] = timings.get(name, 0.0) + (
            time.perf_counter() - stage_start
        ) * 1000

    def _finish_profile(self, timings, frame_start):
        timings["total"] = (time.perf_counter() - frame_start) * 1000
        for name, duration_ms in timings.items():
            total, count = self._profile_totals.get(name, (0.0, 0))
            self._profile_totals[name] = (total + duration_ms, count + 1)

        self._profile_frames += 1
        if self._profile_frames >= self.profile_interval:
            details = " | ".join(
                f"{name}={total / count:.2f}ms"
                for name, (total, count) in self._profile_totals.items()
            )
            print(
                f"[CAMERA PROFILE] avg over {self._profile_frames} updates: "
                f"{details}"
            )
            self._profile_totals.clear()
            self._profile_frames = 0

    def update(self, frame, mask=None, landmarks=None):
        frame_start = time.perf_counter()
        timings = {}

        stage_start = time.perf_counter()
        gray = self._gray(frame)
        self._record_timing(timings, "grayscale", stage_start)

        stage_start = time.perf_counter()
        observed = self._clean_landmarks(landmarks)
        self._record_timing(timings, "landmark_cleaning", stage_start)

        if self.prev_gray is None:
            self.prev_gray = gray
            self.prev_landmarks = observed
            stage_start = time.perf_counter()
            self.prev_mask = self._feature_mask(gray, observed, mask)
            self._record_timing(timings, "feature_mask", stage_start)
            stage_start = time.perf_counter()
            self.prev_pts = self._detect_features(gray, self.prev_mask)
            self._record_timing(timings, "feature_detection", stage_start)
            stage_start = time.perf_counter()
            result = self._result(np.eye(3), False, 0, "initialized")
            self._record_timing(timings, "result", stage_start)
            self._finish_profile(timings, frame_start)
            result["timings_ms"] = dict(timings)
            return result

        # Track semantic landmarks from previous frame.
        stage_start = time.perf_counter()
        tracked_old, tracked_new = self._track_points(
            list(self.prev_landmarks.values()), gray
        ) if self.prev_landmarks else (
            np.empty((0, 2), np.float32), np.empty((0, 2), np.float32)
        )
        self._record_timing(timings, "semantic_tracking", stage_start)

        stage_start = time.perf_counter()
        tracked_names = []
        if self.prev_landmarks and len(tracked_new):
            # Track in stable insertion order to retain semantic identity.
            names = list(self.prev_landmarks.keys())
            # _track_points may filter points; reproduce its validity mask
            # by matching the returned old coordinates to original points.
            for old_pt in tracked_old:
                distances = np.linalg.norm(
                    np.asarray(list(self.prev_landmarks.values())) - old_pt, axis=1
                )
                tracked_names.append(names[int(np.argmin(distances))])

        current_landmarks = {}
        for name, point in zip(tracked_names, tracked_new):
            current_landmarks[name] = point

        # Fresh detector results re-anchor semantic point positions.
        current_landmarks.update(observed)
        self._record_timing(timings, "landmark_association", stage_start)

        common = [
            name for name in self.prev_landmarks
            if name in current_landmarks
        ]
        source = "none"
        H, inliers, error = None, 0, None

        if len(common) >= self.min_inliers:
            old_semantic = np.asarray([self.prev_landmarks[n] for n in common], np.float32)
            new_semantic = np.asarray([current_landmarks[n] for n in common], np.float32)
            stage_start = time.perf_counter()
            H, inliers, error = self._estimate(old_semantic, new_semantic)
            self._record_timing(timings, "semantic_homography", stage_start)
            if H is not None:
                source = "landmarks-detected" if observed else "landmarks-tracked"

        # Fallback: optical flow features, detected using the PREVIOUS frame's mask.
        if H is None:
            if (
                self.prev_pts is None
                or len(self.prev_pts) < 8
                or self.frame_count % self.feature_interval == 0
            ):
                if self.prev_mask is None:
                    stage_start = time.perf_counter()
                    self.prev_mask = self._feature_mask(
                        self.prev_gray, self.prev_landmarks, None
                    )
                    self._record_timing(timings, "feature_mask", stage_start)
                stage_start = time.perf_counter()
                self.prev_pts = self._detect_features(self.prev_gray, self.prev_mask)
                self._record_timing(timings, "feature_detection", stage_start)

            stage_start = time.perf_counter()
            old, new = self._track_points(self.prev_pts, gray)
            self._record_timing(timings, "feature_tracking", stage_start)
            if len(old) >= 8:
                stage_start = time.perf_counter()
                H, inliers, error = self._estimate(old, new, min_inliers=8)
                self._record_timing(timings, "feature_homography", stage_start)
                if H is not None:
                    source = "optical-flow"

            if H is not None:
                self.prev_pts = new.reshape(-1, 1, 2)
            else:
                stage_start = time.perf_counter()
                feature_mask = self._feature_mask(gray, current_landmarks, mask)
                self._record_timing(timings, "feature_mask", stage_start)
                self.prev_mask = feature_mask
                stage_start = time.perf_counter()
                self.prev_pts = self._detect_features(
                    gray, feature_mask
                )
                self._record_timing(timings, "feature_detection", stage_start)

        success = H is not None
        if success:
            stage_start = time.perf_counter()
            try:
                self.cumulative_H = self.cumulative_H @ np.linalg.inv(H)
                self.cumulative_H /= self.cumulative_H[2, 2]
            except (np.linalg.LinAlgError, ZeroDivisionError):
                success = False
            self._record_timing(timings, "cumulative_transform", stage_start)

        stage_start = time.perf_counter()
        self.prev_gray = gray
        self.prev_landmarks = current_landmarks
        self.frame_count += 1
        self._record_timing(timings, "state_update", stage_start)

        if H is not None:
            self.prev_mask = mask.copy() if mask is not None else None

        stage_start = time.perf_counter()
        result = self._result(H if success else np.eye(3), success, inliers, source, error)
        self._record_timing(timings, "result", stage_start)
        self._finish_profile(timings, frame_start)
        result["timings_ms"] = dict(timings)
        return result

    def reset(self):
        self.prev_gray = None
        self.prev_pts = None
        self.prev_mask = None
        self.prev_landmarks = {}
        self.frame_count = 0
        self.cumulative_H = np.eye(3, dtype=np.float64)
        self._profile_totals.clear()
        self._profile_frames = 0
