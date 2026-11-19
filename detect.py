
from ultralytics import YOLO


class BallDetector:

    def __init__(
        self,
        model_path="best.pt",
        confidence=0.25,
        device=None
    ):
        self.model = YOLO(model_path)
        self.confidence = confidence
        self.device = device

        # Find the ball class ID from model names.
        self.ball_class_id = None

        for class_id, name in self.model.names.items():
            if name.lower() == "ball":
                self.ball_class_id = class_id
                break

        if self.ball_class_id is None:
            raise ValueError(
                f"No 'ball' class found. "
                f"Available classes: {self.model.names}"
            )

    
    def get_detections(self, frame, player_class_id):
        """
        Run YOLO once and extract:
        - Highest-confidence ball center
        - Bottom-center of every detected player bounding box

        Returns:
        ball_detection: (x, y) or None
        player_positions: [(x, y), ...]
        """

        results = self.model.predict(
            source=frame,
            conf=self.confidence,
            verbose=False
        )

        result = results[0]

        ball_detection = None
        best_ball_confidence = 0

        player_positions = []

        # Find ball class ID from model names
        names = self.model.names

        if isinstance(names, dict):
            ball_class_id = next(
                (class_id for class_id, name in names.items()
                if name.lower() == "ball"),
                None
            )
        else:
            ball_class_id = next(
                (class_id for class_id, name in enumerate(names)
                if name.lower() == "ball"),
                None
            )

        if ball_class_id is None:
            raise ValueError("Could not find a 'ball' class in model.names")

        for box in result.boxes:

            class_id = int(box.cls[0].item())
            confidence = float(box.conf[0].item())

            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()

            # -----------------------------------------
            # BALL DETECTION
            # -----------------------------------------

            if class_id == ball_class_id:

                if confidence > best_ball_confidence:

                    center_x = (x1 + x2) / 2
                    center_y = (y1 + y2) / 2

                    ball_detection = (
                        float(center_x),
                        float(center_y)
                    )

                    best_ball_confidence = confidence

            # -----------------------------------------
            # PLAYER DETECTION
            # -----------------------------------------

            elif class_id == player_class_id and confidence > 0.5:

                bottom_center_x = (x1 + x2) / 2
                bottom_center_y = y2

                player_positions.append((
                    float(bottom_center_x),
                    float(bottom_center_y)
                ))

        return ball_detection, player_positions
