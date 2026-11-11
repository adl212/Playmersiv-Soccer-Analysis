
import cv2
import numpy as np


class ManualCalibration:

    def __init__(self, pitch_width=1050, pitch_height=680):

        self.pitch_width = pitch_width
        self.pitch_height = pitch_height

        self.image_points = []
        self.field_points = []

        self.current_frame = None
        self.window_name = "Manual Field Calibration"

    def _mouse_callback(self, event, x, y, flags, param):

        if event == cv2.EVENT_LBUTTONDOWN:

            self.image_points.append([x, y])

            print(
                f"Selected landmark {len(self.image_points)}: "
                f"({x}, {y})"
            )

    def _draw_points(self, frame):

        output = frame.copy()

        for i, point in enumerate(self.image_points):

            x, y = point

            cv2.circle(
                output,
                (x, y),
                7,
                (0, 255, 0),
                -1
            )

            cv2.putText(
                output,
                str(i + 1),
                (x + 10, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

        instructions = [
            "Click field landmarks in the video.",
            "ENTER: Finish selection",
            "U: Undo last point",
            "R: Reset selection",
            "ESC: Cancel"
        ]

        for i, text in enumerate(instructions):

            cv2.putText(
                output,
                text,
                (15, 30 + i * 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2
            )

        return output

    def select_landmarks(self, frame):

        self.current_frame = frame.copy()

        cv2.namedWindow(self.window_name)

        cv2.setMouseCallback(
            self.window_name,
            self._mouse_callback
        )

        while True:

            display = self._draw_points(
                self.current_frame
            )

            cv2.imshow(
                self.window_name,
                display
            )

            key = cv2.waitKey(20) & 0xFF

            # Finish
            if key == 13:

                if len(self.image_points) < 4:
                    print("Select at least 4 landmarks.")
                    continue

                break

            # Undo
            elif key == ord("u"):

                if self.image_points:
                    self.image_points.pop()

            # Reset
            elif key == ord("r"):

                self.image_points.clear()

            # Cancel
            elif key == 27:

                self.image_points.clear()
                cv2.destroyWindow(self.window_name)

                return None

        cv2.destroyWindow(self.window_name)

        return np.array(
            self.image_points,
            dtype=np.float32
        )

    def assign_field_points(self):
        """
        Assign corresponding normalized field coordinates
        to each selected image landmark.

        Coordinates are entered as percentages:
            X: 0-100 (left to right)
            Y: 0-100 (top to bottom)
        """

        print("\nAssign corresponding field coordinates.")

        for i in range(len(self.image_points)):

            print(f"\nLandmark {i + 1}")

            while True:

                try:

                    x = float(input("Field X (0-100): "))
                    y = float(input("Field Y (0-100): "))

                    if not (0 <= x <= 100 and 0 <= y <= 100):
                        print("Coordinates must be between 0 and 100.")
                        continue

                    self.field_points.append([
                        x / 100 * self.pitch_width,
                        y / 100 * self.pitch_height
                    ])

                    break

                except ValueError:
                    print("Enter valid numbers.")

        return np.array(
            self.field_points,
            dtype=np.float32
        )

    def calibrate(self, frame):

        image_points = self.select_landmarks(frame)

        if image_points is None:
            return None

        field_points = self.assign_field_points()

        H, mask = cv2.findHomography(
            image_points,
            field_points,
            cv2.RANSAC,
            5.0
        )

        if H is None:
            raise ValueError("Homography estimation failed.")

        return H, image_points, field_points
