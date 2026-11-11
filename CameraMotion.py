
import cv2
import numpy as np


class CameraMotionEstimator:
    """
    Estimates relative camera motion between consecutive video frames.

    Supports:
        - Pan
        - Tilt
        - Zoom
        - Rotation
        - Perspective changes

    Uses:
        - Shi-Tomasi feature detection
        - Lucas-Kanade optical flow
        - RANSAC homography estimation

    The returned homography maps points from the previous
    frame into the current frame.
    """

    def __init__(
        self,
        max_features=1500,
        quality_level=0.01,
        min_distance=8,
        ransac_threshold=3.0,
        min_inliers=12,
        feature_interval=10
    ):
        self.max_features = max_features
        self.quality_level = quality_level
        self.min_distance = min_distance
        self.ransac_threshold = ransac_threshold
        self.min_inliers = min_inliers
        self.feature_interval = feature_interval

        self.prev_gray = None
        self.prev_pts = None

        self.frame_count = 0

        # Maps the current frame into the initial reference frame.
        self.cumulative_H = np.eye(3, dtype=np.float64)

        self.last_H = np.eye(3, dtype=np.float64)
        self.last_inliers = 0
        self.last_success = False

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

    def _estimate_homography(self, prev_pts, curr_pts):
        H, mask = cv2.findHomography(
            prev_pts,
            curr_pts,
            cv2.RANSAC,
            self.ransac_threshold
        )

        if H is None or mask is None:
            return None, 0

        inliers = int(mask.sum())

        if inliers < self.min_inliers:
            return None, inliers

        # Reject invalid or numerically unstable transformations.
        if not np.all(np.isfinite(H)):
            return None, inliers

        if abs(H[2, 2]) < 1e-8:
            return None, inliers

        H = H / H[2, 2]

        return H, inliers

    def _extract_motion(self, H):
        """
        Extract approximate motion characteristics.

        These values are descriptive, not exact physical camera
        movements.
        """

        # Normalize the affine portion.
        A = H[:2, :2]

        # Approximate uniform scale.
        scale = np.sqrt(abs(np.linalg.det(A)))

        # Approximate rotation.
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

    def update(self, frame, mask=None):
        """
        Process a new frame.

        Args:
            frame: BGR or grayscale image.
            mask: Optional uint8 mask. 255 = eligible for feature
                  detection; 0 = ignore.

        Returns:
            Dictionary containing:
                homography: Previous frame -> current frame.
                cumulative_homography: Current frame -> initial frame.
                motion: Approximate motion characteristics.
                inliers: Number of RANSAC inliers.
                success: Whether motion estimation succeeded.
        """

        gray = self._to_gray(frame)

        # Initialize on first frame.
        if self.prev_gray is None:
            self.prev_gray = gray
            self.prev_pts = self._detect_features(gray, mask)

            return self._result(
                np.eye(3),
                False,
                0
            )

        # Detect new features periodically or when too few remain.
        if (
            self.prev_pts is None
            or len(self.prev_pts) < self.min_inliers * 2
            or self.frame_count % self.feature_interval == 0
        ):
            self.prev_pts = self._detect_features(
                self.prev_gray,
                mask
            )

        if self.prev_pts is None or len(self.prev_pts) < 8:
            return self._handle_failure(gray)

        # Track features between frames.
        curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(
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

        if curr_pts is None or status is None:
            return self._handle_failure(gray)

        valid = status.flatten() == 1

        old = self.prev_pts[valid].reshape(-1, 2)
        new = curr_pts[valid].reshape(-1, 2)

        if len(old) < 8:
            return self._handle_failure(gray)

        # Forward-backward consistency check.
        backward_pts, backward_status, _ = cv2.calcOpticalFlowPyrLK(
            gray,
            self.prev_gray,
            new.reshape(-1, 1, 2),
            None
        )

        if backward_pts is not None and backward_status is not None:
            backward_valid = backward_status.flatten() == 1

            old = old[backward_valid]
            new = new[backward_valid]
            backward_pts = backward_pts.reshape(-1, 2)[backward_valid]

            fb_error = np.linalg.norm(old - backward_pts, axis=1)

            good = fb_error < 1.5

            old = old[good]
            new = new[good]

        if len(old) < 8:
            return self._handle_failure(gray)

        # Estimate camera transformation.
        H, inliers = self._estimate_homography(old, new)

        if H is None:
            return self._handle_failure(gray, inliers)

        # Accumulate transformations.
        # H maps previous frame -> current frame.
        # cumulative_H maps current frame -> initial frame.
        self.cumulative_H = (
            self.cumulative_H @ np.linalg.inv(H)
        )

        self.last_H = H
        self.last_inliers = inliers
        self.last_success = True

        # Prepare for next frame.
        self.prev_gray = gray
        self.prev_pts = new.reshape(-1, 1, 2)
        self.frame_count += 1

        return self._result(H, True, inliers)

    def _handle_failure(self, gray, inliers=0):
        """
        Do not accumulate an invalid transformation.
        Reinitialize features for the next frame.
        """

        self.prev_gray = gray
        self.prev_pts = self._detect_features(gray)

        self.last_success = False
        self.last_inliers = inliers

        self.frame_count += 1

        return self._result(
            np.eye(3),
            False,
            inliers
        )

    def _result(self, H, success, inliers):
        motion = self._extract_motion(H)

        return {
            "homography": H,
            "cumulative_homography": self.cumulative_H.copy(),
            "motion": motion,
            "inliers": inliers,
            "success": success
        }

    def reset(self):
        """Reset all tracking and accumulated motion."""

        self.prev_gray = None
        self.prev_pts = None
        self.frame_count = 0

        self.cumulative_H = np.eye(3, dtype=np.float64)
        self.last_H = np.eye(3, dtype=np.float64)
        self.last_inliers = 0
        self.last_success = False
