import numpy as np

from CameraMotion import CameraMotionEstimator


def _estimator_with_frame():
    estimator = CameraMotionEstimator()
    estimator.prev_gray = np.zeros((1080, 1920), dtype=np.uint8)
    return estimator


def _spread_points():
    return np.array(
        [
            [100, 100], [800, 100], [1800, 100],
            [100, 500], [800, 500], [1800, 500],
            [100, 1000], [800, 1000], [1800, 1000],
        ],
        dtype=np.float32,
    )


def test_camera_estimate_requires_at_least_six_inliers():
    estimator = _estimator_with_frame()
    old_points = _spread_points()[:4]
    new_points = old_points + np.array([20, 0], dtype=np.float32)

    homography, inliers, _ = estimator._estimate(old_points, new_points)

    assert homography is None
    assert inliers == 0


def test_camera_estimate_accepts_small_frame_motion():
    estimator = _estimator_with_frame()
    old_points = _spread_points()
    new_points = old_points + np.array([20, -5], dtype=np.float32)

    homography, inliers, error = estimator._estimate(old_points, new_points)

    assert homography is not None
    assert inliers == len(old_points)
    assert error < 1e-3


def test_camera_estimate_rejects_implausible_frame_motion():
    estimator = _estimator_with_frame()
    old_points = _spread_points()
    new_points = old_points + np.array([1000, 0], dtype=np.float32)

    homography, inliers, _ = estimator._estimate(old_points, new_points)

    assert homography is None
    assert inliers == len(old_points)