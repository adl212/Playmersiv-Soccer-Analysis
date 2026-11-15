
import cv2
import numpy as np


class PlayerTrack:

    def __init__(self, track_id, detection):

        self.id = track_id

        x, y = detection

        self.kf = cv2.KalmanFilter(4, 2)

        # State: [x, y, vx, vy]
        self.kf.transitionMatrix = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], dtype=np.float32)

        # Measurement: [x, y]
        self.kf.measurementMatrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ], dtype=np.float32)

        self.kf.processNoiseCov = (
            np.eye(4, dtype=np.float32) * 0.05
        )

        self.kf.measurementNoiseCov = (
            np.eye(2, dtype=np.float32) * 10
        )

        self.kf.errorCovPost = (
            np.eye(4, dtype=np.float32) * 10
        )

        self.kf.statePost = np.array([
            [x],
            [y],
            [0],
            [0]
        ], dtype=np.float32)

        self.missed = 0
        self.hits = 1

        self.detected = True

    def compensate_camera(self, H):
        """
        Transform the Kalman state from the previous
        frame into the current frame.

        H: previous-frame -> current-frame homography.
        """

        if H is None:
            return

        H = np.asarray(H, dtype=np.float64)

        state = self.kf.statePost.flatten()

        x, y, vx, vy = state

        # Homography denominator
        w = (
            H[2, 0] * x +
            H[2, 1] * y +
            H[2, 2]
        )

        if abs(w) < 1e-8:
            return

        # Transform position
        nx = (
            H[0, 0] * x +
            H[0, 1] * y +
            H[0, 2]
        ) / w

        ny = (
            H[1, 0] * x +
            H[1, 1] * y +
            H[1, 2]
        ) / w

        # Local Jacobian of the homography
        J = np.array([
            [
                (H[0, 0] - nx * H[2, 0]) / w,
                (H[0, 1] - nx * H[2, 1]) / w
            ],
            [
                (H[1, 0] - ny * H[2, 0]) / w,
                (H[1, 1] - ny * H[2, 1]) / w
            ]
        ])

        # Transform velocity
        velocity = J @ np.array([vx, vy])

        new_state = np.array([
            [nx],
            [ny],
            [velocity[0]],
            [velocity[1]]
        ], dtype=np.float32)

        self.kf.statePost = new_state

        # Transform covariance as well
        T = np.zeros((4, 4), dtype=np.float32)

        T[:2, :2] = J
        T[2:, 2:] = J

        P = self.kf.errorCovPost

        self.kf.errorCovPost = T @ P @ T.T

    def predict(self):

        prediction = self.kf.predict()

        # Commit prediction as current state until
        # a measurement corrects it.
        self.kf.statePost = self.kf.statePre.copy()
        self.kf.errorCovPost = self.kf.errorCovPre.copy()

        self.detected = False

        return prediction[:2].flatten()

    def correct(self, detection):

        measurement = np.array([
            [detection[0]],
            [detection[1]]
        ], dtype=np.float32)

        self.kf.correct(measurement)

        self.missed = 0
        self.hits += 1
        self.detected = True

    def position(self):

        state = self.kf.statePost.flatten()

        return float(state[0]), float(state[1])

    def velocity(self):

        state = self.kf.statePost.flatten()

        return float(state[2]), float(state[3])


class PlayerTracker:

    def __init__(
        self,
        max_distance=100,
        max_missed=15
    ):

        self.tracks = []

        self.next_id = 0

        self.max_distance = max_distance
        self.max_missed = max_missed

    def update(self, detections, camera_H=None):
        """
        detections:
            List of player bottom-center positions [(x, y), ...]

        camera_H:
            Previous-frame -> current-frame homography.

        Returns:
            List of player track dictionaries.
        """

        detections = [
            tuple(map(float, d))
            for d in detections
        ]

        # ------------------------------------------------
        # 1. Transform existing tracks for camera movement
        # ------------------------------------------------

        for track in self.tracks:

            track.compensate_camera(camera_H)

        # ------------------------------------------------
        # 2. Predict every existing player's position
        # ------------------------------------------------

        predictions = []

        for track in self.tracks:

            predictions.append(track.predict())

        # ------------------------------------------------
        # 3. Calculate detection-to-track distances
        # ------------------------------------------------

        candidates = []

        for track_idx, prediction in enumerate(predictions):

            for detection_idx, detection in enumerate(detections):

                distance = np.linalg.norm(
                    prediction - np.array(detection)
                )

                if distance <= self.max_distance:

                    candidates.append((
                        distance,
                        track_idx,
                        detection_idx
                    ))

        # ------------------------------------------------
        # 4. Greedy nearest-neighbor association
        # ------------------------------------------------

        candidates.sort(key=lambda x: x[0])

        matched_tracks = set()
        matched_detections = set()

        for distance, track_idx, detection_idx in candidates:

            if track_idx in matched_tracks:
                continue

            if detection_idx in matched_detections:
                continue

            track = self.tracks[track_idx]

            detection = detections[detection_idx]

            track.correct(detection)

            matched_tracks.add(track_idx)
            matched_detections.add(detection_idx)

        # ------------------------------------------------
        # 5. Handle missed detections
        # ------------------------------------------------

        for track_idx, track in enumerate(self.tracks):

            if track_idx not in matched_tracks:

                track.missed += 1

        # ------------------------------------------------
        # 6. Create new tracks for unmatched detections
        # ------------------------------------------------

        for detection_idx, detection in enumerate(detections):

            if detection_idx not in matched_detections:

                new_track = PlayerTrack(
                    self.next_id,
                    detection
                )

                self.tracks.append(new_track)

                self.next_id += 1

        # ------------------------------------------------
        # 7. Remove tracks that have been missing too long
        # ------------------------------------------------

        self.tracks = [
            track for track in self.tracks
            if track.missed <= self.max_missed
        ]

        # ------------------------------------------------
        # 8. Return current tracks
        # ------------------------------------------------

        results = []

        for track in self.tracks:

            x, y = track.position()

            vx, vy = track.velocity()

            results.append({
                "id": track.id,
                "position": (x, y),
                "velocity": (vx, vy),
                "detected": track.detected,
                "missed": track.missed
            })

        return results
