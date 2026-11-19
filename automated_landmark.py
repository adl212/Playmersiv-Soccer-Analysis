import ultralytics
import cv2
import supervision as sv
import numpy as np

model = ultralytics.YOLO("best_v7.pt")

CLASS_KEYPOINT_SCHEMA = {
    0: (0, [
        "center-circle-top", "center-circle-right", "center-circle-bottom", "center-circle-left", "center"
    ]),
    1: (5, [
        "big-left-bottom-left", "big-left-bottom-right", "big-left-top-right", "big-left-top-left", "left-semicircle-left", 
        "left-semicircle-right", "left-pk"
    ]),
    2: (12, [
        "big-right-top-left", "big-right-top-right", "big-right-bottom-right", "big-right-bottom-left", "right-semicircle-left", 
        "right-semicircle-right", "right-pk"
    ]),
}

keypoint_map = CLASS_KEYPOINT_SCHEMA[0][1] + CLASS_KEYPOINT_SCHEMA[1][1] + CLASS_KEYPOINT_SCHEMA[2][1]
def create_viz(result, file_name, KEYPOINT_CONF=0.5, ):
  labels = []
  class_ids = result.boxes.cls.int().tolist()
  keypoints_data = result.keypoints.data
  for cls_num, cls_id in enumerate(class_ids):
      schema = CLASS_KEYPOINT_SCHEMA[cls_id]
      start = schema[0]
      det = keypoints_data[cls_num]
      for i, (x, y, conf) in enumerate(det):
          if conf < KEYPOINT_CONF:
              continue
              
          if i >= start and i < (start + len(schema[1])):
              labels.append(schema[1][i-start] + " CONF: " + str(round(float(conf), 2)))
          else:
              labels.append("INC")

  key_points = sv.KeyPoints.from_ultralytics(result)
  key_points.xy = np.array([key_points.xy[key_points.confidence > KEYPOINT_CONF]])
  vertex_annotator = sv.VertexAnnotator(radius=6, color=sv.Color.BLUE)
  edge_annotator = sv.EdgeAnnotator(thickness=2)
  label_annotator = sv.VertexLabelAnnotator(text_padding=1)


  annotated_frame = cv2.imread(file_name)
  annotated_frame = label_annotator.annotate(
      scene=annotated_frame, key_points=key_points, labels=labels)
  annotated_frame = edge_annotator.annotate(
      scene=annotated_frame,
      key_points=key_points)
  annotated_frame = vertex_annotator.annotate(
      scene=annotated_frame,
      key_points=key_points)
  cv2.imshow("Visualize Keypoint Model", annotated_frame)
# 2. Open the video file
video_path = "18eBfrhGS3BurZIDah6IcXVfDnbUaE5Jk.mp4"
cap = cv2.VideoCapture(video_path)

# 3. Specify the frame you want to start on (0-indexed)
start_frame = 95000  # e.g., jump straight to frame 150
cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

# 4. Read and process frames starting from the target point
while cap.isOpened():
  success, frame = cap.read()
  if not success:
    break

  # Run YOLO inference on the current frame
  results = model(frame, save=True)
  file_name = (results[0].save_dir + "/" + results[0].path)
  # Optional: visualize results
  create_viz(results[0], file_name)

  # Press 'q' to exit early
  if cv2.waitKey(1) & 0xFF == ord("q"):
    break

cap.release()
cv2.destroyAllWindows()