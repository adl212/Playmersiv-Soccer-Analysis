from GoalDetector import GoalDetector


def test_confirms_goal_after_goal_area_evidence_and_center_restart():
    detector = GoalDetector(fps=30)

    # Observed ball moves quickly toward the left goal, inside the goal mouth.
    observations = [
        (1, (0.25, 0.50)),
        (2, (0.18, 0.50)),
        (3, (0.10, 0.50)),
        (4, (0.02, 0.50)),
    ]
    result = None
    for frame, point in observations:
        result = detector.update(frame, point)

    assert result["state"] == detector.GOAL_CANDIDATE
    assert result["goal_area_evidence"] is True

    # Ball is temporarily not observed after reaching the goal area.
    detector.update(5, None, ball_missed_frames=2)

    # A restart is observed at the center spot: stationary, then moved.
    detector.update(10, (0.50, 0.50))
    detector.update(11, (0.50, 0.50))
    detector.update(12, (0.50, 0.50))
    result = detector.update(13, (0.60, 0.50))

    assert result["goal"] is True
    assert result["goal_side"] == "left"
    assert result["state"] == detector.CONFIRMED
    assert result["event_frame"] == 13


def test_does_not_confirm_without_goal_area_evidence():
    detector = GoalDetector(fps=30)

    # A fast pass toward the middle of the pitch is not a goal attempt.
    detector.update(1, (0.35, 0.50))
    detector.update(2, (0.42, 0.50))
    result = detector.update(3, (0.50, 0.50))

    detector.update(10, (0.50, 0.50))
    detector.update(11, (0.50, 0.50))
    result = detector.update(12, (0.55, 0.50))

    assert result["goal"] is False


def test_kalman_only_missing_frames_do_not_create_ball_history():
    detector = GoalDetector(fps=30)

    detector.update(1, (0.20, 0.50))
    detector.update(2, None, ball_missed_frames=1)
    detector.update(3, None, ball_missed_frames=2)

    assert len(detector.ball_history) == 1
    assert detector.state == detector.IDLE
