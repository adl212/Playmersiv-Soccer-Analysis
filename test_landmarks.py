
import cv2
import numpy as np

from LandmarkDetector import LandmarkDetector, FIELD_POINTS_PERCENT


# ============================================================
# CONFIGURATION
# ============================================================

VIDEO_PATH = "18eBfrhGS3BurZIDah6IcXVfDnbUaE5Jk.mp4"
MODEL_PATH = "best_v7.pt"

# Frame to inspect. Use the same frame number as your main.py
FRAME_INDEX = 100000

PITCH_WIDTH = 1050
PITCH_HEIGHT = 680

# Visualization dimensions
PITCH_DISPLAY_WIDTH = 1000
PITCH_DISPLAY_HEIGHT = 647

# Reprojection error threshold, in pitch pixels.
ERROR_THRESHOLD = 20.0

# Save visualizations to disk
SAVE_IMAGES = True


# ============================================================
# INITIALIZATION
# ============================================================

landmark_detector = LandmarkDetector(
    model_path=MODEL_PATH,
    pitch_width=PITCH_WIDTH,
    pitch_height=PITCH_HEIGHT,

    inference_confidence=0.15,
    center_keypoint_confidence=0.50,
    penalty_keypoint_confidence=0.70,
    penalty_box_detection_confidence=0.50,

    ransac_threshold=10.0,
    min_inliers=4,
    max_mean_error=20.0,
    min_pitch_coverage=0.01
)

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    raise RuntimeError(f"Could not open video: {VIDEO_PATH}")

cap.set(cv2.CAP_PROP_POS_FRAMES, FRAME_INDEX)

success, frame = cap.read()

if not success:
    raise RuntimeError(
        f"Could not read frame {FRAME_INDEX}. "
        "Check FRAME_INDEX and VIDEO_PATH."
    )

frame_height, frame_width = frame.shape[:2]


# ============================================================
# VISUALIZATION COLORS
# ============================================================

COLORS = {
    0: (0, 255, 255),     # Center circle: yellow
    1: (255, 100, 0),     # Left box: blue/orange in BGR
    2: (0, 140, 255),     # Right box: orange
}

GREEN = (0, 255, 0)
RED = (0, 0, 255)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def percent_to_pitch_pixel(point):
    x, y = point

    return (
        int(x / 100 * PITCH_DISPLAY_WIDTH),
        int(y / 100 * PITCH_DISPLAY_HEIGHT)
    )


def draw_pitch():

    pitch = np.zeros(
        (PITCH_DISPLAY_HEIGHT, PITCH_DISPLAY_WIDTH, 3),
        dtype=np.uint8
    )

    pitch[:] = (35, 105, 35)

    def p(x, y):
        return percent_to_pitch_pixel((x, y))

    # Outer boundary
    cv2.rectangle(
        pitch,
        p(0, 0),
        p(100, 100),
        WHITE,
        2
    )

    # Halfway line
    cv2.line(pitch, p(50, 0), p(50, 100), WHITE, 2)

    # Center circle
    cv2.circle(
        pitch,
        p(50, 50),
        int(8.714 / 100 * PITCH_DISPLAY_WIDTH),
        WHITE,
        2
    )

    # Center spot
    cv2.circle(pitch, p(50, 50), 4, WHITE, -1)

    # Penalty boxes
    cv2.rectangle(
        pitch,
        p(0, 20.353),
        p(15.714, 79.647),
        WHITE,
        2
    )

    cv2.rectangle(
        pitch,
        p(84.286, 20.353),
        p(100, 79.647),
        WHITE,
        2
    )

    # Six-yard boxes
    cv2.rectangle(
        pitch,
        p(0, 36.765),
        p(5.714, 63.235),
        WHITE,
        2
    )

    cv2.rectangle(
        pitch,
        p(94.286, 36.765),
        p(100, 63.235),
        WHITE,
        2
    )

    return pitch


def draw_detected_landmarks(image, landmarks):

    output = image.copy()

    for landmark in landmarks:

        name = landmark["name"]
        x, y = landmark["image_point"]
        score = landmark["score"]
        class_id = landmark["class_id"]

        point = (int(x), int(y))
        color = COLORS.get(class_id, WHITE)

        cv2.circle(output, point, 7, color, -1)
        cv2.circle(output, point, 10, WHITE, 2)

        label = f"{name} ({score:.2f})"

        label_y = max(20, point[1] - 12)

        cv2.putText(
            output,
            label,
            (point[0] + 8, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            2
        )

    return output


def draw_pitch_landmarks(pitch, landmarks, H):

    output = pitch.copy()

    if H is None:
        return output

    for landmark in landmarks:

        name = landmark["name"]
        image_point = np.array(
            landmark["image_point"],
            dtype=np.float32
        ).reshape(1, 1, 2)

        class_id = landmark["class_id"]
        color = COLORS.get(class_id, WHITE)

        # Transform image coordinate into canonical pitch coordinate.
        projected = cv2.perspectiveTransform(
            image_point,
            H
        )[0][0]

        x, y = projected

        normalized = (
            x / PITCH_WIDTH * 100,
            y / PITCH_HEIGHT * 100
        )

        if not (
            0 <= normalized[0] <= 100
            and 0 <= normalized[1] <= 100
        ):
            continue

        px, py = percent_to_pitch_pixel(normalized)

        # Actual transformed prediction
        cv2.circle(output, (px, py), 8, color, -1)
        cv2.circle(output, (px, py), 11, WHITE, 2)

        cv2.putText(
            output,
            name,
            (px + 10, py - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            WHITE,
            1
        )

    # Draw expected canonical landmark positions.
    for name, expected in FIELD_POINTS_PERCENT.items():

        px, py = percent_to_pitch_pixel(expected)

        cv2.drawMarker(
            output,
            (px, py),
            GREEN,
            markerType=cv2.MARKER_CROSS,
            markerSize=18,
            thickness=2
        )

    return output


def draw_clipped_projected_line(image, points, H_inverse):

    """
    Projects a canonical pitch line into the camera image.
    Clips individual segments to the image boundaries.
    """

    points = np.array(points, dtype=np.float32).reshape(-1, 1, 2)

    projected = cv2.perspectiveTransform(
        points,
        H_inverse
    ).reshape(-1, 2)

    for i in range(len(projected) - 1):

        p1 = projected[i]
        p2 = projected[i + 1]

        if not np.all(np.isfinite(p1)) or not np.all(np.isfinite(p2)):
            continue

        if max(abs(p1[0]), abs(p1[1]), abs(p2[0]), abs(p2[1])) > 1e7:
            continue

        pt1 = tuple(np.round(p1).astype(int))
        pt2 = tuple(np.round(p2).astype(int))

        success, clipped1, clipped2 = cv2.clipLine(
            (0, 0, frame_width, frame_height),
            pt1,
            pt2
        )

        if success:
            cv2.line(
                image,
                clipped1,
                clipped2,
                (0, 255, 0),
                2
            )


def draw_projected_pitch(image, H):

    """
    Draws canonical pitch markings projected back into the video.
    This is independent of the model's visualization.
    """

    output = image.copy()

    if H is None:
        return output

    try:
        H_inverse = np.linalg.inv(H)
    except np.linalg.LinAlgError:
        print("Homography is singular.")
        return output

    def horizontal_line(y):
        return [
            (x, y)
            for x in np.linspace(0, PITCH_WIDTH, 100)
        ]

    def vertical_line(x):
        return [
            (x, y)
            for y in np.linspace(0, PITCH_HEIGHT, 100)
        ]

    # Outer boundary
    draw_clipped_projected_line(
        output, horizontal_line(0), H_inverse
    )
    draw_clipped_projected_line(
        output, horizontal_line(PITCH_HEIGHT), H_inverse
    )
    draw_clipped_projected_line(
        output, vertical_line(0), H_inverse
    )
    draw_clipped_projected_line(
        output, vertical_line(PITCH_WIDTH), H_inverse
    )

    # Halfway line
    draw_clipped_projected_line(
        output,
        vertical_line(PITCH_WIDTH / 2),
        H_inverse
    )

    # Center circle
    circle_points = [
        (
            PITCH_WIDTH / 2
            + PITCH_WIDTH * 0.08714 * np.cos(t),
            PITCH_HEIGHT / 2
            + PITCH_HEIGHT * 0.13456 * np.sin(t)
        )
        for t in np.linspace(0, 2 * np.pi, 100)
    ]

    draw_clipped_projected_line(
        output,
        circle_points,
        H_inverse
    )

    # Penalty box boundaries
    for x in [0, PITCH_WIDTH * 0.15714,
              PITCH_WIDTH * 0.84286, PITCH_WIDTH]:

        draw_clipped_projected_line(
            output,
            vertical_line(x),
            H_inverse
        )

    for y in [
        PITCH_HEIGHT * 0.20353,
        PITCH_HEIGHT * 0.79647
    ]:

        draw_clipped_projected_line(
            output,
            horizontal_line(y),
            H_inverse
        )

    return output


# ============================================================
# RUN LANDMARK DETECTION
# ============================================================

print(f"\nAnalyzing frame {FRAME_INDEX}...")

result, H, inlier_count, mean_error, detected_landmarks = (
    landmark_detector.calibrate(frame)
)

print("\n========== IMAGE TO CANONICAL COORDINATES ==========")

for landmark in detected_landmarks:

    name = landmark["name"]
    image_x, image_y = landmark["image_point"]

    expected_x, expected_y = FIELD_POINTS_PERCENT[name]

    print(
        f"{name:30s} "
        f"IMAGE=({image_x:7.1f}, {image_y:7.1f}) "
        f"FIELD=({expected_x:6.2f}, {expected_y:6.2f})"
    )

print("\n========== DETECTION SUMMARY ==========")

print("Total detected landmarks:", len(detected_landmarks))

class_counts = {
    0: 0,
    1: 0,
    2: 0
}

for landmark in detected_landmarks:
    class_counts[landmark["class_id"]] += 1

print("Center circle:", class_counts[0])
print("Left penalty box:", class_counts[1])
print("Right penalty box:", class_counts[2])

print("Homography available:", H is not None)
print("Inlier count:", inlier_count)
print("Mean reprojection error:", mean_error)


# ============================================================
# REPROJECTION ERROR DIAGNOSTICS
# ============================================================

if H is not None:

    image_points = np.array(
        [item["image_point"] for item in detected_landmarks],
        dtype=np.float32
    )

    expected_points = np.array(
        [
            (
                FIELD_POINTS_PERCENT[item["name"]][0]
                * PITCH_WIDTH / 100,
                FIELD_POINTS_PERCENT[item["name"]][1]
                * PITCH_HEIGHT / 100
            )
            for item in detected_landmarks
        ],
        dtype=np.float32
    )

    projected_points = cv2.perspectiveTransform(
        image_points.reshape(-1, 1, 2),
        H
    ).reshape(-1, 2)

    errors = np.linalg.norm(
        projected_points - expected_points,
        axis=1
    )

    print("\n========== INDIVIDUAL REPROJECTION ERRORS ==========")

    for landmark, error in zip(detected_landmarks, errors):

        status = (
            "GOOD"
            if error <= ERROR_THRESHOLD
            else "HIGH ERROR"
        )

        print(
            f'{landmark["name"]:30s} '
            f'error={error:8.2f} px | {status}'
        )

    print("\nMean error:", np.mean(errors))
    print("Median error:", np.median(errors))
    print("Maximum error:", np.max(errors))

    print("\nEstimated homography:")
    print(H)

else:
    print("\nNo valid homography was estimated.")


# ============================================================
# GENERATE VISUALIZATIONS
# ============================================================

# 1. Original frame with model predictions.
image_predictions = draw_detected_landmarks(
    frame,
    detected_landmarks
)

# 2. Normalized 2D pitch with transformed predictions.
pitch = draw_pitch()

pitch_predictions = draw_pitch_landmarks(
    pitch,
    detected_landmarks,
    H
)

# 3. Original frame with projected pitch markings.
if H is not None:

    pitch_overlay = draw_projected_pitch(
        frame,
        H
    )

else:
    pitch_overlay = frame.copy()


# ============================================================
# DISPLAY RESULTS
# ============================================================

cv2.imshow(
    "1 - Original Video: Detected Landmarks",
    image_predictions
)

cv2.imshow(
    "2 - 2D Pitch: Transformed Landmarks",
    pitch_predictions
)

cv2.imshow(
    "3 - Original Video: Projected Pitch",
    pitch_overlay
)

print("\nVisualization windows opened.")
print("Press Q or ESC to exit.")

if SAVE_IMAGES:

    cv2.imwrite(
        "test_landmarks_original.png",
        image_predictions
    )

    cv2.imwrite(
        "test_landmarks_2d_pitch.png",
        pitch_predictions
    )

    cv2.imwrite(
        "test_landmarks_pitch_overlay.png",
        pitch_overlay
    )

    print("Saved visualization images.")

while True:

    key = cv2.waitKey(100) & 0xFF

    if key == ord("q") or key == 27:
        break

cap.release()
cv2.destroyAllWindows()
