
import cv2
import numpy as np


class PitchVisualizer:

    def __init__(self, width=1000, height=700):

        self.width = width
        self.height = height

    def create_pitch(self):

        pitch = np.zeros(
            (self.height, self.width, 3),
            dtype=np.uint8
        )

        pitch[:] = (40, 120, 40)

        white = (255, 255, 255)

        # Outer boundary
        cv2.rectangle(
            pitch,
            (0, 0),
            (self.width - 1, self.height - 1),
            white,
            3
        )

        # Halfway line
        cv2.line(
            pitch,
            (self.width // 2, 0),
            (self.width // 2, self.height),
            white,
            2
        )

        # Center circle
        cv2.circle(
            pitch,
            (self.width // 2, self.height // 2),
            int(self.height * 0.1),
            white,
            2
        )

        # Center spot
        cv2.circle(
            pitch,
            (self.width // 2, self.height // 2),
            5,
            white,
            -1
        )

        # Penalty areas
        penalty_width = int(self.width * 0.18)
        penalty_height = int(self.height * 440 / self.height)

        top = (self.height - penalty_height) // 2
        bottom = top + penalty_height

        cv2.rectangle(
            pitch,
            (0, top),
            (penalty_width, bottom),
            white,
            2
        )

        cv2.rectangle(
            pitch,
            (self.width - penalty_width, top),
            (self.width, bottom),
            white,
            2
        )

        # Goal areas
        goal_width = int(self.width * 0.06)
        goal_height = int(self.height * 200 / self.height)

        top_goal = (self.height - goal_height) // 2
        bottom_goal = top_goal + goal_height

        cv2.rectangle(
            pitch,
            (0, top_goal),
            (goal_width, bottom_goal),
            white,
            2
        )

        cv2.rectangle(
            pitch,
            (self.width - goal_width, top_goal),
            (self.width, bottom_goal),
            white,
            2
        )

        return pitch

    def plot_ball(self, pitch, normalized_position):

        if normalized_position is None:
            return pitch

        x, y = normalized_position

        # Don't plot outside the pitch.
        if not (0 <= x <= 1 and 0 <= y <= 1):
            return pitch

        px = int(x * self.width)
        py = int(y * self.height)

        cv2.circle(
            pitch,
            (px, py),
            9,
            (0, 255, 255),
            -1
        )

        cv2.circle(
            pitch,
            (px, py),
            12,
            (255, 255, 255),
            2
        )

        return pitch

    def show(self, normalized_position):

        pitch = self.create_pitch()

        self.plot_ball(
            pitch,
            normalized_position
        )

        return pitch
