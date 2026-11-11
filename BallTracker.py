
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

    
    def update(self, detection, camera_H=None):
        """
        Update ball tracking.

        detection: (x, y) or None
        camera_H: homography mapping previous-frame coordinates
                to current-frame coordinates.
        """

        if not self.initialized:
            if detection is None:
                return None

            x, y = detection

            self.kf.statePost = np.array(
                [[x], [y], [0], [0]],
                dtype=np.float32
            )

            self.initialized = True
            self.missed_frames = 0

            return (x, y)

        # --------------------------------------------------------
        # 1. Compensate tracker state for camera motion
        # --------------------------------------------------------

        if camera_H is not None:
            self._transform_state(camera_H)

        # --------------------------------------------------------
        # 2. Kalman prediction
        # --------------------------------------------------------

        prediction = self.kf.predict()

        # --------------------------------------------------------
        # 3. Correct using YOLO detection
        # --------------------------------------------------------

        if detection is not None:

            measurement = np.array(
                [[detection[0]], [detection[1]]],
                dtype=np.float32
            )

            distance = np.linalg.norm(
                measurement.flatten() - prediction[:2].flatten()
            )

            if distance < 150:
                self.kf.correct(measurement)
                self.missed_frames = 0
            else:
                # Reject implausible detection.
                self.kf.statePost = self.kf.statePre.copy()
                self.kf.errorCovPost = self.kf.errorCovPre.copy()
                self.missed_frames += 1

        else:
            # No detection: retain Kalman prediction.
            self.kf.statePost = self.kf.statePre.copy()
            self.kf.errorCovPost = self.kf.errorCovPre.copy()
            self.missed_frames += 1

        # --------------------------------------------------------
        # 4. Handle lost tracking
        # --------------------------------------------------------

        if self.missed_frames > 15:
            self.reset()
            return None

        state = self.kf.statePost.flatten()

        return (float(state[0]), float(state[1]))


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
