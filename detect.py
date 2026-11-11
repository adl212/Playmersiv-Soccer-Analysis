
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

    def get_ball_detection(self, frame):

        results = self.model.predict(
            frame,
            conf=self.confidence,
            classes=[self.ball_class_id],
            device=self.device,
            verbose=False
        )

        best_detection = None
        best_confidence = 0

        for result in results:

            if result.boxes is None:
                continue

            for box in result.boxes:

                confidence = float(box.conf[0])

                if confidence > best_confidence:

                    x1, y1, x2, y2 = (
                        box.xyxy[0].cpu().numpy()
                    )

                    center_x = (x1 + x2) / 2
                    center_y = (y1 + y2) / 2

                    best_detection = (
                        float(center_x),
                        float(center_y)
                    )

                    best_confidence = confidence

        return best_detection
