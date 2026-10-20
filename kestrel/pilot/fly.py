"""The camera-only pilot: detector, gate pose, gate memory and the state machine (SPEC 8.3).

It knows its own state and sees camera frames. It never learns where gates are except through
its detector, and it does not know how many gates a course has.
"""

from __future__ import annotations

import math

import numpy as np
import pybullet as p

from kestrel.perception.detect import Detection, GateDetector
from kestrel.perception.pose import estimate_gate
from kestrel.perception.tracker import GateTracker
from kestrel.pilot.controller import (
    LowLevelController,
    OwnState,
    ReferenceTrajectory,
    turn_toward,
)
from kestrel.pilot.flight_log import PilotState
from kestrel.pilot.params import PilotParams
from kestrel.sim.camera import Intrinsics, camera_view_matrix

SEARCH_HEIGHT_M = 1.5
SEARCH_CLIMB_SPEED_M_S = 1.0
# The reference counts as at the approach point within this distance.
ALIGN_RADIUS_M = 0.15
EXIT_RADIUS_M = 0.4
# A gate is a candidate target only while the drone is this far in front of its plane.
FRONT_MARGIN_M = 0.3


def yaw_of(quat: np.ndarray) -> float:
    return float(p.getEulerFromQuaternion(list(quat))[2])


class CameraPilot:
    needs_frames = True

    def __init__(
        self, params: PilotParams, detector: GateDetector, intrinsics: Intrinsics, own: OwnState
    ) -> None:
        self.params = params
        self.detector = detector
        self.intrinsics = intrinsics
        self.tracker = GateTracker(params.ema_alpha, params.forget_s)
        self.reference = ReferenceTrajectory(own.pos)
        self.low_level = LowLevelController()
        self.yaw = yaw_of(own.quat)
        self.state = PilotState.SEARCH
        self.target_id: int | None = None
        self.aligned = False
        self.commit_center = np.zeros(3)
        self.commit_forward = np.zeros(3)
        self.search_anchor = np.array([own.pos[0], own.pos[1], SEARCH_HEIGHT_M])
        self.goal = self.search_anchor.copy()

    @property
    def believed_gate(self) -> np.ndarray | None:
        """Where the pilot thinks the gate it is flying at is."""
        if self.state in (PilotState.COMMIT, PilotState.EXIT):
            return self.commit_center
        target = self.tracker.get(self.target_id)
        return None if target is None else target.center

    def perceive(self, time_s: float, own: OwnState, frame: np.ndarray) -> list[Detection]:
        """Detect gates in a camera frame and fold their poses into the gate memory."""
        view = camera_view_matrix(own.pos, own.quat)
        detections = self.detector.detect(frame, self.params.conf_thresh)
        observations = []
        for detection in detections:
            observation = estimate_gate(detection, self.intrinsics, view, self.params.max_reproj_px)
            if observation is not None:
                observations.append(observation)
        self.tracker.update(observations, time_s, keep_id=self.target_id)
        return detections

    def _pick_target(self, pos: np.ndarray) -> None:
        candidates = [
            h
            for h in self.tracker.hypotheses
            if h.confirmed
            and not h.passed
            and float(np.dot(pos - h.center, h.forward)) < -FRONT_MARGIN_M
        ]
        if not candidates:
            if self.state is not PilotState.SEARCH:
                self.state = PilotState.SEARCH
                self.search_anchor = np.array([*self.reference.pos[:2], SEARCH_HEIGHT_M])
            return
        nearest = min(candidates, key=lambda h: float(np.linalg.norm(h.center - pos)))
        self.target_id = nearest.id
        self.aligned = False
        self.state = PilotState.APPROACH

    def _commit(self, center: np.ndarray, forward: np.ndarray) -> None:
        self.state = PilotState.COMMIT
        self.commit_center = center.copy()
        self.commit_forward = forward.copy()

    def _update_state(self, pos: np.ndarray) -> None:
        if self.state is PilotState.SEARCH:
            self._pick_target(pos)
        if self.state is PilotState.APPROACH:
            target = self.tracker.get(self.target_id)
            if target is None:
                self._pick_target(pos)
            else:
                crossed = float(np.dot(pos - target.center, target.forward)) > 0.0
                close = np.linalg.norm(pos - target.center) <= self.params.commit_dist
                if crossed or (self.aligned and close):
                    self._commit(target.center, target.forward)
        past = float(np.dot(pos - self.commit_center, self.commit_forward))
        if self.state is PilotState.COMMIT and past > 0.0:
            target = self.tracker.get(self.target_id)
            if target is not None:
                target.passed = True
            self.state = PilotState.EXIT
        if self.state is PilotState.EXIT:
            exit_point = self.commit_center + self.commit_forward * self.params.exit_dist
            beyond = past >= self.params.exit_dist
            if beyond or self.reference.distance_to(exit_point) < EXIT_RADIUS_M:
                self.target_id = None
                self._pick_target(pos)

    def control(self, own: OwnState, dt: float) -> np.ndarray:
        """Advance the state machine and return motor RPMs for one control step."""
        self._update_state(own.pos)
        params = self.params
        if self.state is PilotState.SEARCH:
            self.goal = self.search_anchor
            self.reference.advance(self.goal, SEARCH_CLIMB_SPEED_M_S, True, dt)
            self.yaw += params.search_yaw_rate * dt
        elif self.state is PilotState.APPROACH:
            target = self.tracker.get(self.target_id)
            assert target is not None
            approach_point = target.center - target.forward * params.approach_offset
            if not self.aligned and self.reference.distance_to(approach_point) < ALIGN_RADIUS_M:
                self.aligned = True
            self.goal = target.center if self.aligned else approach_point
            self.reference.advance(self.goal, params.speed_cap, not self.aligned, dt)
            to_gate = target.center - own.pos
            self.yaw = turn_toward(self.yaw, math.atan2(to_gate[1], to_gate[0]), dt)
        else:
            self.goal = self.commit_center + self.commit_forward * params.exit_dist
            self.reference.advance(self.goal, params.speed_cap, False, dt)
            heading = math.atan2(self.commit_forward[1], self.commit_forward[0])
            self.yaw = turn_toward(self.yaw, heading, dt)
        return self.low_level.rpm(own, self.reference, self.yaw, dt)
