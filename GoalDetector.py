from collections import deque
import math


class GoalDetector:
    """
    Lightweight, explainable goal-event detector.

    It does not claim referee-level goal-line certainty. It looks for:
      1. An observed ball trajectory projected toward a goal mouth.
      2. Evidence that the ball reached/disappeared near that goal.
      3. A subsequent center-spot restart (kickoff).

    All pitch positions are normalized (x, y) in [0, 1].
    """

    IDLE = "IDLE"
    SHOT_CANDIDATE = "SHOT_CANDIDATE"
    GOAL_CANDIDATE = "GOAL_CANDIDATE"
    CONFIRMED = "GOAL_CONFIRMED"

    def __init__(
        self,
        fps=30.0,
        goal_line_x_margin=0.035,
        goal_mouth_y=(0.435, 0.565),
        shot_speed_threshold=0.30,
        max_projection_seconds=1.8,
        candidate_timeout_seconds=35.0,
        kickoff_center_radius=0.065,
        kickoff_move_threshold=0.025,
        kickoff_stationary_frames=3,
        kickoff_max_frames=45,
    ):
        self.fps = max(1.0, float(fps))
        self.goal_line_x_margin = float(goal_line_x_margin)
        self.goal_mouth_y = tuple(goal_mouth_y)
        self.shot_speed_threshold = float(shot_speed_threshold)
        self.max_projection_frames = int(max_projection_seconds * self.fps)
        self.candidate_timeout_frames = int(candidate_timeout_seconds * self.fps)
        self.kickoff_center_radius = float(kickoff_center_radius)
        self.kickoff_move_threshold = float(kickoff_move_threshold)
        self.kickoff_stationary_frames = int(kickoff_stationary_frames)
        self.kickoff_max_frames = int(kickoff_max_frames)

        self.ball_history = deque(maxlen=8)
        self.state = self.IDLE
        self.candidate_side = None
        self.candidate_frame = None
        self.goal_area_evidence = False
        self.candidate_reason = None

        self.center_observations = deque(maxlen=8)
        self.kickoff_anchor = None
        self.kickoff_stationary_count = 0
        self.kickoff_armed_frame = None

        self.last_event = None
        self.last_result = self._result(False, None, 0.0, "waiting-for-shot")

    @staticmethod
    def _distance(a, b):
        return math.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1]))

    def _result(self, goal, side, confidence, reason):
        return {
            "goal": bool(goal),
            "state": self.state,
            "confidence": float(max(0.0, min(1.0, confidence))),
            "goal_side": side,
            "event_frame": self.last_event,
            "reason": reason,
            "goal_area_evidence": bool(self.goal_area_evidence),
        }

    def _trajectory_candidate(self):
        """Estimate short-horizon velocity from observed pitch-plane points."""
        if len(self.ball_history) < 3:
            return None

        first_frame, first = self.ball_history[0]
        last_frame, last = self.ball_history[-1]
        elapsed = last_frame - first_frame
        if elapsed < 2:
            return None

        vx = (last[0] - first[0]) * self.fps / elapsed
        vy = (last[1] - first[1]) * self.fps / elapsed
        speed = math.hypot(vx, vy)
        if speed < self.shot_speed_threshold:
            return None

        candidates = []
        for side, goal_x, direction in (
            ("left", 0.0, -1.0),
            ("right", 1.0, 1.0),
        ):
            if vx * direction <= 0.0:
                continue
            seconds_to_line = (goal_x - last[0]) / vx
            if seconds_to_line < 0.0:
                continue
            frames_to_line = seconds_to_line * self.fps
            if frames_to_line > self.max_projection_frames:
                continue
            projected_y = last[1] + vy * seconds_to_line
            if self.goal_mouth_y[0] <= projected_y <= self.goal_mouth_y[1]:
                # Prefer a close, fast trajectory rather than distant motion.
                proximity = 1.0 - min(1.0, abs(goal_x - last[0]) / 0.5)
                confidence = min(0.85, 0.45 + 0.20 * proximity + 0.10 * min(1.0, speed))
                candidates.append((confidence, side, projected_y))

        if not candidates:
            return None
        confidence, side, projected_y = max(candidates, key=lambda item: item[0])
        return {
            "side": side,
            "confidence": confidence,
            "projected_y": projected_y,
            "speed": speed,
        }

    def _update_kickoff_detector(self, frame_number, observed_ball):
        """
        Detect a restart as: ball observed near center, remains there briefly,
        then moves away. Player distribution is a supporting (not mandatory)
        cue because camera crops can hide one side of the pitch.
        """
        if observed_ball is None:
            return False

        center = (0.5, 0.5)
        near_center = self._distance(observed_ball, center) <= self.kickoff_center_radius

        if near_center:
            self.center_observations.append((frame_number, observed_ball))
            if self.kickoff_anchor is None:
                self.kickoff_anchor = observed_ball
                self.kickoff_stationary_count = 1
                self.kickoff_armed_frame = frame_number
            else:
                if self._distance(observed_ball, self.kickoff_anchor) <= 0.018:
                    self.kickoff_stationary_count += 1
                else:
                    self.kickoff_anchor = observed_ball
                    self.kickoff_stationary_count = 1
                    self.kickoff_armed_frame = frame_number

            if self.kickoff_stationary_count >= self.kickoff_stationary_frames:
                self.kickoff_armed_frame = (
                    self.kickoff_armed_frame
                    if self.kickoff_armed_frame is not None
                    else frame_number
                )
            return False

        # A ball that has moved away from a previously stable center position
        # is the restart signal.
        if (
            self.kickoff_anchor is not None
            and self.kickoff_stationary_count >= self.kickoff_stationary_frames
            and self.kickoff_armed_frame is not None
            and frame_number - self.kickoff_armed_frame <= self.kickoff_max_frames
            and self._distance(observed_ball, self.kickoff_anchor) >= self.kickoff_move_threshold
        ):
            # Player distribution is intentionally not required: a broadcast
            # crop can hide one half of the pitch. The center restart is the
            # primary confirmation cue.
            self._reset_kickoff()
            return True

        if (
            self.kickoff_armed_frame is not None
            and frame_number - self.kickoff_armed_frame > self.kickoff_max_frames
        ):
            self._reset_kickoff()
        return False

    def _reset_kickoff(self):
        self.kickoff_anchor = None
        self.kickoff_stationary_count = 0
        self.kickoff_armed_frame = None
        self.center_observations.clear()

    def _reset_candidate(self):
        self.state = self.IDLE
        self.candidate_side = None
        self.candidate_frame = None
        self.goal_area_evidence = False
        self.candidate_reason = None
        self._reset_kickoff()

    def update(
        self,
        frame_number,
        observed_ball_field_position,
        ball_missed_frames=0,
    ):
        """
        Args:
            frame_number: monotonically increasing video frame index.
            observed_ball_field_position: normalized (x, y) from the current
                YOLO ball detection mapped to the pitch, or None. Do not pass
                a Kalman-only prediction here as if it were an observation.
            player_field_positions: iterable of normalized (x, y) positions
                for currently detected players only.
            ball_missed_frames: BallTracker.missed_frames, for disappearance
                evidence after a candidate trajectory.
        """
        observed = None
        if observed_ball_field_position is not None:
            x, y = map(float, observed_ball_field_position)
            if math.isfinite(x) and math.isfinite(y):
                observed = (x, y)
                self.ball_history.append((int(frame_number), observed))

        # A candidate expires rather than being linked to a distant restart.
        if (
            self.candidate_frame is not None
            and frame_number - self.candidate_frame > self.candidate_timeout_frames
        ):
            self._reset_candidate()

        trajectory = self._trajectory_candidate() if observed is not None else None

        if self.state == self.IDLE and trajectory is not None:
            self.state = self.SHOT_CANDIDATE
            self.candidate_side = trajectory["side"]
            self.candidate_frame = int(frame_number)
            self.candidate_reason = "trajectory-toward-goal"

        if self.candidate_side is not None and observed is not None:
            near_goal_line = (
                observed[0] <= self.goal_line_x_margin
                if self.candidate_side == "left"
                else observed[0] >= 1.0 - self.goal_line_x_margin
            )
            in_goal_mouth = self.goal_mouth_y[0] <= observed[1] <= self.goal_mouth_y[1]
            if near_goal_line and in_goal_mouth:
                self.goal_area_evidence = True
                self.state = self.GOAL_CANDIDATE
                self.candidate_reason = "observed-near-goal-mouth"

        # If the ball disappears shortly after a projected shot, retain it as
        # weaker goal-area evidence. This accommodates occlusion, not proof.
        if (
            self.candidate_side is not None
            and not self.goal_area_evidence
            and ball_missed_frames >= 2
            and frame_number - self.candidate_frame <= int(3.0 * self.fps)
        ):
            latest = self.ball_history[-1][1] if self.ball_history else None
            if latest is not None:
                close_to_line = (
                    latest[0] <= 0.12
                    if self.candidate_side == "left"
                    else latest[0] >= 0.88
                )
                if close_to_line and self.goal_mouth_y[0] <= latest[1] <= self.goal_mouth_y[1]:
                    self.goal_area_evidence = True
                    self.state = self.GOAL_CANDIDATE
                    self.candidate_reason = "ball-lost-near-projected-goal"

        kickoff = self._update_kickoff_detector(frame_number, observed)

        if (
            kickoff
            and self.candidate_side is not None
            and self.goal_area_evidence
            and self.state == self.GOAL_CANDIDATE
        ):
            self.state = self.CONFIRMED
            self.last_event = int(frame_number)
            result = self._result(
                True,
                self.candidate_side,
                0.95 if self.candidate_reason == "observed-near-goal-mouth" else 0.78,
                "goal-evidence-followed-by-kickoff",
            )
            self.last_result = result
            self.ball_history.clear()
            self._reset_candidate()
            # Preserve the confirmed state in this returned event frame.
            result["state"] = self.CONFIRMED
            return result

        if self.candidate_side is None:
            state = self.IDLE
            reason = "waiting-for-shot"
            confidence = 0.0
        elif self.goal_area_evidence:
            state = self.GOAL_CANDIDATE
            reason = self.candidate_reason or "goal-area-evidence"
            confidence = 0.65 if "observed" in reason else 0.45
        else:
            state = self.SHOT_CANDIDATE
            reason = self.candidate_reason or "trajectory-toward-goal"
            confidence = 0.35

        self.state = state
        result = self._result(False, self.candidate_side, confidence, reason)
        self.last_result = result
        return result

    def reset(self):
        self.ball_history.clear()
        self._reset_candidate()
        self.last_event = None
        self.last_result = self._result(False, None, 0.0, "reset")
