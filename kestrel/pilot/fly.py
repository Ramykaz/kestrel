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
from kestrel.perception.tracker import MATCH_RADIUS_M, GateTracker
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
# A trusted approach point stops moving once the drone is within approach_offset plus this.
# Close up, a few degrees of noise in the gate's direction swing the point by decimetres, and
# a drone already near the gate would chase it backwards.
APPROACH_FREEZE_MARGIN_M = 1.0
# A gate is a candidate target only while the drone is this far in front of its plane.
FRONT_MARGIN_M = 0.3
# Target choice cost: metres of distance plus this many metres per radian of turn. Courses put
# the next gate roughly ahead of the last one's exit, but the pilot is not told the order.
TURN_COST_M_PER_RAD = 4.0


def level(direction: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    """Horizontal unit vector along a direction. Approach points sit level with the gate
    center: the drone often starts far below a gate, and a few degrees of tilt noise in a
    close-up gate direction would otherwise move the point up or down by decimetres."""
    for candidate in (direction, fallback):
        horizontal = np.array([candidate[0], candidate[1], 0.0])
        norm = float(np.linalg.norm(horizontal))
        if norm > 1e-3:
            return horizontal / norm
    raise ValueError("neither direction has a horizontal component")


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
        self.frozen_approach_point: np.ndarray | None = None
        self.commit_center = np.zeros(3)
        self.commit_forward = np.zeros(3)
        # Gates flown through, oldest first. Tracker estimates expire when out of view, so a
        # passed gate seen again later would otherwise look new and get flown twice.
        self.passed_centers: list[np.ndarray] = []
        self.search_s = 0.0
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

    def _pass_rank(self, center: np.ndarray) -> int | None:
        """Position of a gate in the passed list (0 is the oldest), or None if never passed."""
        for rank, passed in enumerate(self.passed_centers):
            if np.linalg.norm(center - passed) <= MATCH_RADIUS_M:
                return rank
        return None

    def _choice_cost(self, center: np.ndarray, pos: np.ndarray) -> float:
        offset = center - pos
        turn = abs(math.remainder(math.atan2(offset[1], offset[0]) - self.yaw, math.tau))
        return float(np.linalg.norm(offset)) + TURN_COST_M_PER_RAD * turn

    def _pick_target(self, pos: np.ndarray) -> None:
        in_front = [
            h
            for h in self.tracker.hypotheses
            if h.confirmed and float(np.dot(pos - h.center, h.forward)) < -FRONT_MARGIN_M
        ]
        fresh = [h for h in in_front if not h.passed and self._pass_rank(h.center) is None]
        full_turn_s = math.tau / self.params.search_yaw_rate
        if fresh:
            choice = min(fresh, key=lambda h: self._choice_cost(h.center, pos))
        elif in_front and self.search_s >= full_turn_s:
            # A full search turn found nothing new. Gates only count in course order, so a
            # gate flown out of order earlier may still be needed: fly the oldest pass again.
            choice = min(in_front, key=lambda h: self._pass_rank(h.center) or 0)
        else:
            if self.state is not PilotState.SEARCH:
                self.state = PilotState.SEARCH
                self.search_s = 0.0
                self.search_anchor = np.array([*self.reference.pos[:2], SEARCH_HEIGHT_M])
            return
        self.target_id = choice.id
        self.aligned = False
        self.frozen_approach_point = None
        self.search_s = 0.0
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
            rank = self._pass_rank(self.commit_center)
            if rank is not None:
                del self.passed_centers[rank]
            self.passed_centers.append(self.commit_center.copy())
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
            self.search_s += dt
        elif self.state is PilotState.APPROACH:
            target = self.tracker.get(self.target_id)
            assert target is not None
            to_gate = target.center - own.pos
            distance = float(np.linalg.norm(to_gate))
            approach_point = self.frozen_approach_point
            if approach_point is None and target.forward_trusted:
                direction = level(target.forward, fallback=to_gate)
                approach_point = target.center - direction * params.approach_offset
                if distance < params.approach_offset + APPROACH_FREEZE_MARGIN_M:
                    self.frozen_approach_point = approach_point
            elif approach_point is None:
                # Until the gate's direction is trusted, approach along the line of sight and
                # fly straight at the gate once inside the approach distance.
                direction = level(to_gate, fallback=target.forward)
                approach_point = target.center - direction * params.approach_offset
                if distance <= params.approach_offset:
                    self.aligned = True
            if not self.aligned and self.reference.distance_to(approach_point) < ALIGN_RADIUS_M:
                self.aligned = True
            self.goal = target.center if self.aligned else approach_point
            self.reference.advance(self.goal, params.speed_cap, not self.aligned, dt)
            self.yaw = turn_toward(self.yaw, math.atan2(to_gate[1], to_gate[0]), dt)
        else:
            self.goal = self.commit_center + self.commit_forward * params.exit_dist
            self.reference.advance(self.goal, params.speed_cap, False, dt)
            heading = math.atan2(self.commit_forward[1], self.commit_forward[0])
            self.yaw = turn_toward(self.yaw, heading, dt)
        return self.low_level.rpm(own, self.reference, self.yaw, dt)
