"""Ground-truth autopilot for development fixtures and simulator tests.

It steers by the true gate positions, optionally perturbed, so its flights are not results of
this project and are never exported as such. It exercises the physics, gate crossing, crash
detection, flight logs and the web player before a learned pilot exists.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from kestrel.data.labels import compute_gate_labels
from kestrel.perception.detect import Detection
from kestrel.pilot.controller import LowLevelController, OwnState, ReferenceTrajectory
from kestrel.pilot.flight_log import FlightLog, PilotState
from kestrel.sim.camera import OnboardCamera
from kestrel.sim.course import Course
from kestrel.sim.flight import fly_course
from kestrel.sim.oracle import Oracle
from kestrel.sim.world import World

TAKEOFF_HEIGHT_M = 1.5
SWITCH_RADIUS_M = {"takeoff": 0.3, "approach": 0.15, "center": 0.15, "exit": 0.4}
MIN_HEADING_SPEED_M_S = 0.2


@dataclass(frozen=True)
class ReferencePilotParams:
    speed_m_s: float = 2.0
    gate_position_noise_m: float = 0.0
    """Sigma of a fixed per-gate error added to each true gate center."""
    approach_offset_m: float = 1.5
    exit_offset_m: float = 0.8
    commit_dist_m: float = 1.2


@dataclass(frozen=True)
class _Waypoint:
    point: np.ndarray
    gate_index: int | None
    kind: str


def _waypoints(
    course: Course, params: ReferencePilotParams, rng: np.random.Generator
) -> list[_Waypoint]:
    start = np.array(course.start_pos)
    points = [_Waypoint(np.array([start[0], start[1], TAKEOFF_HEIGHT_M]), None, "takeoff")]
    for index, gate in enumerate(course.gates):
        believed = np.array(gate.center) + rng.normal(0.0, params.gate_position_noise_m, 3)
        forward = gate.forward
        points += [
            _Waypoint(believed - forward * params.approach_offset_m, index, "approach"),
            _Waypoint(believed, index, "center"),
            _Waypoint(believed + forward * params.exit_offset_m, index, "exit"),
        ]
    return points


class ReferencePilot:
    """Flies through waypoints placed around the true gates. Its logged detections are
    ground-truth projections with confidence 1.0, since there is no detector in this loop."""

    needs_frames = False

    def __init__(
        self,
        params: ReferencePilotParams,
        world: World,
        oracle: Oracle,
        camera: OnboardCamera,
        own: OwnState,
        rng: np.random.Generator,
    ) -> None:
        self.params = params
        self.world = world
        self.oracle = oracle
        self.camera = camera
        self.waypoints = _waypoints(oracle.course, params, rng)
        self.index = 0
        self.reference = ReferenceTrajectory(own.pos)
        self.low_level = LowLevelController()
        self.yaw = oracle.course.start_yaw
        self.state = PilotState.SEARCH
        self.goal = self.waypoints[0].point

    @property
    def believed_gate(self) -> np.ndarray | None:
        gate_index = self.waypoints[self.index].gate_index
        return None if gate_index is None else self.waypoints[3 * gate_index + 2].point

    def perceive(self, time_s: float, own: OwnState, frame: np.ndarray | None) -> list[Detection]:
        view = self.camera.view_matrix(own.pos, own.quat)
        labels = compute_gate_labels(
            self.oracle, self.world.client, own.pos, view, self.camera.intrinsics
        )
        return [
            Detection(label.class_id, 1.0, label.box_xyxy, label.keypoints, label.visibility)
            for label in labels
        ]

    def control(self, own: OwnState, dt: float) -> np.ndarray:
        waypoint = self.waypoints[self.index]
        reached = self.reference.distance_to(waypoint.point) < SWITCH_RADIUS_M[waypoint.kind]
        if reached and self.index < len(self.waypoints) - 1:
            self.index += 1
            waypoint = self.waypoints[self.index]
        # Arrive slowly enough to turn onto the gate normal before the gate.
        arrive = waypoint.kind in ("takeoff", "approach")
        self.reference.advance(waypoint.point, self.params.speed_m_s, arrive, dt)
        if np.linalg.norm(self.reference.vel[:2]) > MIN_HEADING_SPEED_M_S:
            self.yaw = math.atan2(self.reference.vel[1], self.reference.vel[0])
        self.goal = waypoint.point
        distance = float(np.linalg.norm(waypoint.point - own.pos))
        if waypoint.kind == "takeoff":
            self.state = PilotState.SEARCH
        elif waypoint.kind == "exit":
            self.state = PilotState.EXIT
        elif waypoint.kind == "center" and distance <= self.params.commit_dist_m:
            self.state = PilotState.COMMIT
        else:
            self.state = PilotState.APPROACH
        return self.low_level.rpm(own, self.reference, self.yaw, dt)


def fly_reference(
    course: Course,
    params: ReferencePilotParams,
    seed: int = 0,
    keep_crash_frames: bool = False,
) -> FlightLog:
    """Fly one course with the ground-truth autopilot and return its log."""
    rng = np.random.default_rng(seed)

    def make_pilot(world: World, oracle: Oracle, camera: OnboardCamera, own: OwnState):
        return ReferencePilot(params, world, oracle, camera, own, rng)

    return fly_course(course, make_pilot, seed=seed, keep_crash_frames=keep_crash_frames)
