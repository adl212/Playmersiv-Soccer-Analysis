
import cv2
import numpy as np


class FieldMapper:

    def __init__(self, pitch_width=1050, pitch_height=680):

        self.pitch_width = pitch_width
        self.pitch_height = pitch_height

        # Maps current video-frame coordinates to pitch coordinates.
        self.H_image_to_field = None

    def calibrate(self, image_points, field_points):
        """
        image_points:
            Known landmark coordinates in the video frame.

        field_points:
            Corresponding coordinates on the top-down pitch.

        At least 4 non-collinear correspondences are required.
        """

        image_points = np.asarray(
            image_points,
            dtype=np.float32
        )

        field_points = np.asarray(
            field_points,
            dtype=np.float32
        )

        if len(image_points) < 4:
            raise ValueError(
                "At least 4 correspondences are required."
            )

        H, mask = cv2.findHomography(
            image_points,
            field_points,
            cv2.RANSAC,
            5.0
        )

        if H is None:
            raise ValueError("Could not estimate field homography.")

        self.H_image_to_field = H

        return H

    def update_camera_motion(self, H_previous_to_current):
        """
        Update field mapping after camera movement.

        H_previous_to_current maps previous-frame pixels
        into current-frame pixels.

        The field mapping must therefore be transformed
        into the new frame's coordinate system.
        """

        if self.H_image_to_field is None:
            return False

        try:
            H_current_to_previous = np.linalg.inv(
                H_previous_to_current
            )

            self.H_image_to_field = (
                self.H_image_to_field @ H_current_to_previous
            )

            self.H_image_to_field /= self.H_image_to_field[2, 2]

            return True

        except np.linalg.LinAlgError:
            return False

    def image_to_field(self, point):
        """
        Convert image pixel coordinates to normalized
        pitch coordinates.

        Returns:
            (x, y), where both coordinates are between
            approximately 0 and 1.
        """

        if self.H_image_to_field is None:
            return None

        point = np.array(
            [[[point[0], point[1]]]],
            dtype=np.float32
        )

        transformed = cv2.perspectiveTransform(
            point,
            self.H_image_to_field
        )

        x, y = transformed[0, 0]

        return (
            float(x / self.pitch_width),
            float(y / self.pitch_height)
        )

    def field_to_pixel(self, normalized_point):
        """
        Convert normalized pitch coordinates into
        visualization pixel coordinates.
        """

        x, y = normalized_point

        return (
            int(x * self.pitch_width),
            int(y * self.pitch_height)
        )
