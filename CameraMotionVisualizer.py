
import cv2
import numpy as np


class CameraMotionVisualizer:

    def __init__(self, vector_spacing=40):
        self.vector_spacing = vector_spacing

    def draw_motion_vectors(self, frame, H):
        """
        Visualize how image points move under the estimated
        homography.
        """

        output = frame.copy()

        height, width = frame.shape[:2]
        spacing = self.vector_spacing

        points = []

        for y in range(spacing, height, spacing):
            for x in range(spacing, width, spacing):
                points.append([x, y])

        points = np.array(
            points,
            dtype=np.float32
        ).reshape(-1, 1, 2)

        transformed = cv2.perspectiveTransform(
            points,
            H
        )

        for original, moved in zip(
            points.reshape(-1, 2),
            transformed.reshape(-1, 2)
        ):

            x1, y1 = original.astype(int)
            x2, y2 = moved.astype(int)

            displacement = np.linalg.norm(
                moved - original
            )

            if displacement < 1:
                continue

            if (
                0 <= x2 < width and
                0 <= y2 < height
            ):
                cv2.arrowedLine(
                    output,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 255),
                    1,
                    tipLength=0.3
                )

        return output

    def draw_point(self, frame, point, color, label):
        if point is None:
            return

        x, y = np.round(point).astype(int)

        cv2.circle(
            frame,
            (x, y),
            8,
            color,
            2
        )

        cv2.putText(
            frame,
            label,
            (x + 12, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2
        )

    def draw_status(self, frame, motion, inliers, success):

        if success:
            lines = [
                f"Pan X: {motion['translation_x']:.1f}",
                f"Pan Y: {motion['translation_y']:.1f}",
                f"Zoom: {motion['zoom_percent']:+.2f}%",
                f"Rotation: {motion['rotation_degrees']:.2f}",
                f"Inliers: {inliers}"
            ]
        else:
            lines = [
                "Camera estimation failed"
            ]

        for i, line in enumerate(lines):

            cv2.putText(
                frame,
                line,
                (15, 30 + i * 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )

        return frame

    def visualize(
        self,
        frame,
        camera_result,
        detection=None,
        raw_prediction=None,
        compensated_prediction=None
    ):

        output = frame.copy()

        # Camera motion vectors
        if camera_result["success"]:

            output = self.draw_motion_vectors(
                output,
                camera_result["homography"]
            )

        # Actual YOLO detection
        self.draw_point(
            output,
            detection,
            (0, 255, 0),
            "Detection"
        )

        # Prediction without camera compensation
        self.draw_point(
            output,
            raw_prediction,
            (0, 0, 255),
            "Uncompensated"
        )

        # Prediction with camera compensation
        self.draw_point(
            output,
            compensated_prediction,
            (255, 0, 0),
            "Compensated"
        )

        self.draw_status(
            output,
            camera_result["motion"],
            camera_result["inliers"],
            camera_result["success"]
        )

        return output
