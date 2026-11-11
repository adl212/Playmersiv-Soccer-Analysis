
import cv2
import numpy as np

class BallTracker:

    def __init__(self):
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

    def update(self, detection):

        if not self.initialized:

            if detection is None:
                return None

            x, y = detection

            self.kf.statePost = np.array([
                [x], [y], [0], [0]
            ], dtype=np.float32)

            self.initialized = True

            return (x, y)

        prediction = self.kf.predict()

        if detection is not None:

            measurement = np.array([
                [detection[0]],
                [detection[1]]
            ], dtype=np.float32)

            distance = np.linalg.norm(
                measurement.flatten() -
                prediction[:2].flatten()
            )

            if distance < 150:
                self.kf.correct(measurement)
                self.missed_frames = 0
            else:
                self.missed_frames += 1

        else:
            self.missed_frames += 1

        if self.missed_frames > 15:
            self.initialized = False
            return None

        state = self.kf.statePost.flatten()

        return state[:2]

    def velocity(self):

        if not self.initialized:
            return np.zeros(2)

        return self.kf.statePost.flatten()[2:4]
