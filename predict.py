
from ultralytics import YOLO
import cv2
import numpy as np
from BallTracker import BallTracker

model = YOLO('soccer_yolov8n_best.pt')

# Change these to match your model's class IDs.
BALL_CLASS = 1
PLAYER_CLASS = 0

tracker = BallTracker()

cap = cv2.VideoCapture("/kaggle/input/datasets/adl212/playmersiv-ai-camera-footage/18eBfrhGS3BurZIDah6IcXVfDnbUaE5Jk.mp4")

# 1. Define video properties
output_filename = 'output.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Codec for .mp4 files
fps = cap.get(5)                               # Frames per second
frame_size = (int(cap.get(3)), int(cap.get(4)))                  # Width, Height (MUST match your frame dimensions)
print(frame_size)
# 2. Initialize the VideoWriter object
out = cv2.VideoWriter(output_filename, fourcc, fps, frame_size)

trajectory = []
frames_looped = 20000
cap.set(cv2.CAP_PROP_POS_FRAMES, frames_looped)
while cap.isOpened() and frames_looped < 23000:
    frames_looped += 1
    if frames_looped % 100 == 0:
        print(frames_looped)
    ret, frame = cap.read()

    if not ret:
        break

    results = model.predict(
        frame,
        imgsz=640,
        verbose=False,
        device=[0,1]
    )

    ball_detection = None

    for box in results[0].boxes:

        class_id = int(box.cls[0])
        confidence = float(box.conf[0])

        x1, y1, x2, y2 = (
            box.xyxy[0].cpu().numpy().astype(int)
        )

        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2

        if class_id == BALL_CLASS:

            if ball_detection is None:
                ball_detection = (
                    center_x,
                    center_y
                )

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

        elif class_id == PLAYER_CLASS:

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (255, 0, 0),
                2
            )

    # Update ball tracker.
    ball = tracker.update(ball_detection)

    if ball is not None:

        x, y = map(int, ball)

        trajectory.append((x, y))

        # Draw trajectory.
        for i in range(1, len(trajectory)):

            cv2.line(
                frame,
                trajectory[i - 1],
                trajectory[i],
                (0, 255, 255),
                2
            )

        # Draw filtered position.
        cv2.circle(
            frame,
            (x, y),
            8,
            (0, 0, 255),
            -1
        )

        velocity = tracker.velocity()

        cv2.putText(
            frame,
            f"Velocity: {np.linalg.norm(velocity):.1f}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )
    else:
        trajectory = []
    out.write(frame)

out.release()