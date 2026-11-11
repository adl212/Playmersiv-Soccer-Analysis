
import cv2
import numpy as np


class BallTracker:

    def __init__(self, camera_motion_estimator=None):

        self.kf = cv2.KalmanFilter(4, 2)

        self.kf.transitionMatrix = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], dtype=np.float32)

        self.kf.measurementMatrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ], dtype=np.float32)

        self.kf.processNoiseCov = (
            np.eye(4, dtype=np.float32) * 0.05
        )

        self.kf.measurementNoiseCov = (
            np.eye(2, dtype=np.float32) * 5
        )

        self.initialized = False
        self.missed_frames = 0

        # Optional camera motion estimator
        self.camera = camera_motion_estimator

        self.last_camera_result = None

        self.max_distance = 150
        self.max_missed_frames = 15

    def _transform_state(self, H):
        """
        Transform Kalman position and velocity from the
        previous frame's coordinate system into the current
        frame's coordinate system.

        Position uses the homography.
        Velocity uses its local Jacobian.
        """

        state = self.kf.statePost.flatten().astype(np.float64)

        x, y, vx, vy = state

        h = H.astype(np.float64)

        # Homogeneous transformation
        denominator = (
            h[2, 0] * x +
            h[2, 1] * y +
            h[2, 2]
        )

        if abs(denominator) < 1e-8:
            return False

        numerator_x = (
            h[0, 0] * x +
            h[0, 1] * y +
            h[0, 2]
        )

        numerator_y = (
            h[1, 0] * x +
            h[1, 1] * y +
            h[1, 2]
        )

        new_x = numerator_x / denominator
        new_y = numerator_y / denominator

        # Jacobian of the homography at the ball position
        J = np.array([
            [
                (h[0, 0] * denominator -
                 numerator_x * h[2, 0]) / denominator**2,

                (h[0, 1] * denominator -
                 numerator_x * h[2, 1]) / denominator**2
            ],
            [
                (h[1, 0] * denominator -
                 numerator_y * h[2, 0]) / denominator**2,

                (h[1, 1] * denominator -
                 numerator_y * h[2, 1]) / denominator**2
            ]
        ])

        new_velocity = J @ np.array([vx, vy])

        if not np.all(np.isfinite(new_velocity)):
            return False

        # Update state
        self.kf.statePost = np.array([
            [new_x],
            [new_y],
            [new_velocity[0]],
            [new_velocity[1]]
        ], dtype=np.float32)

        # Transform covariance as well
        T = np.zeros((4, 4), dtype=np.float64)

        T[:2, :2] = J
        T[2:, 2:] = J

        self.kf.errorCovPost = (
            T @ self.kf.errorCovPost @ T.T
        ).astype(np.float32)

        return True

    def update(self, detection, frame=None, mask=None):
        """
        Args:
            detection: (x, y) ball center, or None.
            frame: Current video frame (BGR).
            mask: Optional camera feature-detection mask.

        Returns:
            Estimated ball position (x, y), or None.
        """

        camera_success = False

        # 1. Estimate camera movement
        if self.camera is not None and frame is not None:

            self.last_camera_result = self.camera.update(
                frame,
                mask=mask
            )

            camera_success = (
                self.last_camera_result["success"]
            )

        # 2. Initialize tracker
        if not self.initialized:

            if detection is None:
                return None

            x, y = detection

            self.kf.statePost = np.array([
                [x], [y], [0], [0]
            ], dtype=np.float32)

            self.initialized = True
            self.missed_frames = 0

            return (float(x), float(y))

        # 3. Compensate for camera movement
        if camera_success:

            H = self.last_camera_result["homography"]

            transformed = self._transform_state(H)

            if not transformed:
                camera_success = False

        # If camera estimation fails, increase uncertainty.
        # This prevents overconfidence in the uncorrected prediction.
        if not camera_success:
            self.kf.errorCovPost *= 1.25

        # 4. Predict ball position
        prediction = self.kf.predict()

        predicted_position = prediction[:2].flatten()

        # 5. Correct using ball detection
        if detection is not None:

            measurement = np.array([
                [detection[0]],
                [detection[1]]
            ], dtype=np.float32)

            distance = np.linalg.norm(
                measurement.flatten() - predicted_position
            )

            if distance < self.max_distance:

                self.kf.correct(measurement)
                self.missed_frames = 0

            else:
                self.missed_frames += 1

                # Keep prediction as the current state
                self.kf.statePost = self.kf.statePre.copy()
                self.kf.errorCovPost = self.kf.errorCovPre.copy()

        else:
            self.missed_frames += 1

            # Important: advance state even without a detection
            self.kf.statePost = self.kf.statePre.copy()
            self.kf.errorCovPost = self.kf.errorCovPre.copy()

        # 6. Handle prolonged detection loss
        if self.missed_frames > self.max_missed_frames:

            self.initialized = False
            self.missed_frames = 0

            return None

        state = self.kf.statePost.flatten()

        return (
            float(state[0]),
            float(state[1])
        )

    def velocity(self):

        if not self.initialized:
            return np.zeros(2)

        return self.kf.statePost.flatten()[2:4]

    def reset(self):

        self.initialized = False
        self.missed_frames = 0

        self.kf.statePost = np.zeros((4, 1), dtype=np.float32)
        self.kf.statePre = np.zeros((4, 1), dtype=np.float32)

        self.kf.errorCovPost = np.eye(4, dtype=np.float32)
        self.kf.errorCovPre = np.eye(4, dtype=np.float32)
