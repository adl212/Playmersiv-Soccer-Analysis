            det_points = points[detection_index]

            for local_index, name in enumerate(names):
                keypoint_index = start + local_index
                if keypoint_index >= len(det_points):
                    continue

                x, y = float(det_points[keypoint_index][0]), float(det_points[keypoint_index][1])
                kp_conf = (
                    float(det_points[keypoint_index][2])
                    if det_points.shape[-1] >= 3
                    else float(box_conf[detection_index])
                )
                score = min(kp_conf, float(box_conf[detection_index]))

                if (
                    score < self.keypoint_confidence
                    or not np.isfinite(x)
                    or not np.isfinite(y)
                ):
                    continue

                # If duplicate detections predict the same semantic point,
                # retain the highest-confidence observation.
                previous = observations.get(name)
                if previous is None or score > previous["score"]:
                    observations[name] = {
                        "name": name,
                        "image_point": np.array([x, y], dtype=np.float32),
                        "score": score,
                        "class_id": int(class_id),
                    }

        return observations

    def correspondences(self, observations):
        image_points, field_points, names = [], [], []
        for name, observation in observations.items():
            if name not in FIELD_POINTS_PERCENT:
                continue
            x_percent, y_percent = FIELD_POINTS_PERCENT[name]
            image_points.append(observation["image_point"])
            field_points.append([
                x_percent / 100.0 * self.pitch_width,
                y_percent / 100.0 * self.pitch_height,
            ])
            names.append(name)

        return (
            np.asarray(image_points, dtype=np.float32).reshape(-1, 2),
            np.asarray(field_points, dtype=np.float32).reshape(-1, 2),
            names,
        )

    def estimate_homography(self, image_points, field_points):
        """Estimate H: image pixels -> canonical pitch pixels."""
        image_points = np.asarray(image_points, dtype=np.float32).reshape(-1, 2)
        field_points = np.asarray(field_points, dtype=np.float32).reshape(-1, 2)

        if len(image_points) < self.min_inliers:
            return None, 0, None, None, 0.0

        H, inlier_mask = cv2.findHomography(
            image_points,
            field_points,
            cv2.RANSAC,
            self.ransac_threshold,
        )
        if H is None or inlier_mask is None or not np.all(np.isfinite(H)):
            return None, 0, None, None, 0.0

        if abs(H[2, 2]) < 1e-10:
            return None, 0, None, None, 0.0

        H = H / H[2, 2]
        inliers = inlier_mask.ravel().astype(bool)
        inlier_count = int(inliers.sum())
        if inlier_count < self.min_inliers:
            return None, inlier_count, None, inliers, 0.0

        projected = cv2.perspectiveTransform(
            image_points.reshape(-1, 1, 2), H
        ).reshape(-1, 2)
        errors = np.linalg.norm(projected - field_points, axis=1)
        mean_error = float(np.mean(errors[inliers]))

        # Reject degenerate sets clustered in a tiny part of the pitch.
        hull = cv2.convexHull(field_points[inliers].astype(np.float32))
        hull_area = float(cv2.contourArea(hull))
        coverage = hull_area / float(self.pitch_width * self.pitch_height)

        if mean_error > self.max_mean_error or coverage < self.min_pitch_coverage:
            return None, inlier_count, mean_error, inliers, coverage

        return H, inlier_count, mean_error, inliers, coverage

    def calibrate(self, frame, observations=None):
        """Detect (unless supplied), then estimate a field homography."""
        if observations is None:
            observations = self.detect(frame)

        image_points, field_points, names = self.correspondences(observations)
        H, inliers, mean_error, inlier_mask, coverage = self.estimate_homography(
            image_points, field_points
        )

        return {
            "observations": observations,
            "image_points": image_points,
            "field_points": field_points,
            "names": names,
            "homography": H,
            "inliers": inliers,
            "mean_error": mean_error,
            "inlier_mask": inlier_mask,
            "coverage": coverage,
            "success": H is not None,
        }

    @staticmethod
    def draw(frame, observations):
        output = frame.copy()
        for name, item in observations.items():
            x, y = np.round(item["image_point"]).astype(int)
            cv2.circle(output, (x, y), 5, (0, 255, 255), -1)
            cv2.putText(
                output, name, (x + 5, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1