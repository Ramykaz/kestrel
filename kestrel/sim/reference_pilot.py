"""Ground-truth autopilot for development fixtures and simulator tests.

It steers by the true gate positions, optionally perturbed, so its flights are not results of
this project and are never exported as such. It exercises the physics, gate crossing, crash
detection, flight logs and the web player before a learned pilot exists.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass

import numpy as np
import pybullet as p
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.utils.enums import DroneModel

from kestrel.data.labels import compute_gate_labels
from kestrel.pilot.flight_log import Detection, FlightEnd, FlightLog, PilotState
from kestrel.sim.camera import OnboardCamera
from kestrel.sim.course import Course
from kestrel.sim.crossing import CrossingKind, detect_crossing
from kestrel.sim.gates import GATE_POST_LINK_INDEX
from kestrel.sim.oracle import Oracle
from kestrel.sim.world import CAMERA_HZ, CONTROL_HZ, PHYSICS_HZ, World

TIMEOUT_S = 90.0
TAKEOFF_HEIGHT_M = 1.5
MAX_ACCEL_M_S2 = 2.5
ALIGN_SPEED_M_S = 0.6
SWITCH_RADIUS_M = {"takeoff": 0.3, "approach": 0.15, "center": 0.15, "exit": 0.4}
# Contacts with the ground before this time are the drone resting on it before takeoff.
GROUND_CONTACT_GRACE_S = 1.0
CRASH_FRAME_SECONDS = 2.0
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


def _waypoints(course: Course, params: ReferencePilotParams, rng: np.random.Generator):
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


def _state_for(waypoint: _Waypoint, distance: float, params: ReferencePilotParams) -> PilotState:
    if waypoint.kind == "takeoff":
        return PilotState.SEARCH
    if waypoint.kind == "exit":
        return PilotState.EXIT
    if waypoint.kind == "center" and distance <= params.commit_dist_m:
        return PilotState.COMMIT
    return PilotState.APPROACH


def fly_reference(
    course: Course,
    params: ReferencePilotParams,
    seed: int = 0,
    keep_crash_frames: bool = False,
) -> FlightLog:
    """Fly one course with the ground-truth autopilot and return its log.

    Logged detections are ground-truth projections with confidence 1.0, since there is no
    detector in this loop.
    """
    rng = np.random.default_rng(seed)
    world = World()
    try:
        oracle = Oracle(course, world.load_course(course))
        camera = OnboardCamera(world.client, world.renderer)
        controller = DSLPIDControl(drone_model=DroneModel.CF2X)
        waypoints = _waypoints(course, params, rng)
        world.reset_drone(np.array(course.start_pos), course.start_yaw)
        log = FlightLog(course.seed, course.preset, sample_hz=CAMERA_HZ)
        recent_frames: deque[np.ndarray] = deque(maxlen=int(CRASH_FRAME_SECONDS * CAMERA_HZ))

        physics_per_control = PHYSICS_HZ // CONTROL_HZ
        control_dt = 1.0 / CONTROL_HZ
        waypoint_index = 0
        next_gate = 0
        yaw = course.start_yaw
        rpm = np.zeros(4)
        next_sample_time = 0.0
        prev_pos = world.drone_state().pos
        # The drone tracks a reference point whose acceleration is limited, because a large
        # position error asks DSLPIDControl for more tilt than the CF2X can hold altitude at.
        reference_pos = prev_pos.copy()
        reference_vel = np.zeros(3)

        for step in range(int(TIMEOUT_S * PHYSICS_HZ)):
            drone = world.drone_state()
            if step % physics_per_control == 0:
                waypoint = waypoints[waypoint_index]
                distance = float(np.linalg.norm(waypoint.point - reference_pos))
                if (
                    distance < SWITCH_RADIUS_M[waypoint.kind]
                    and waypoint_index < len(waypoints) - 1
                ):
                    waypoint_index += 1
                    waypoint = waypoints[waypoint_index]
                    distance = float(np.linalg.norm(waypoint.point - reference_pos))
                direction = (waypoint.point - reference_pos) / max(distance, 1e-6)
                speed = params.speed_m_s
                if waypoint.kind in ("takeoff", "approach"):
                    # Arrive slowly enough to turn onto the gate normal before the gate.
                    speed = min(speed, ALIGN_SPEED_M_S + math.sqrt(2 * MAX_ACCEL_M_S2 * distance))
                velocity_error = direction * speed - reference_vel
                max_change = MAX_ACCEL_M_S2 * control_dt
                change_norm = float(np.linalg.norm(velocity_error))
                if change_norm > max_change:
                    velocity_error *= max_change / change_norm
                reference_vel += velocity_error
                reference_pos += reference_vel * control_dt
                if np.linalg.norm(reference_vel[:2]) > MIN_HEADING_SPEED_M_S:
                    yaw = math.atan2(reference_vel[1], reference_vel[0])
                rpm, _, _ = controller.computeControl(
                    control_timestep=control_dt,
                    cur_pos=drone.pos,
                    cur_quat=drone.quat,
                    cur_vel=drone.vel,
                    cur_ang_vel=drone.ang_vel,
                    target_pos=reference_pos,
                    target_rpy=np.array([0.0, 0.0, yaw]),
                    target_vel=reference_vel,
                )
            waypoint = waypoints[waypoint_index]
            state = _state_for(waypoint, float(np.linalg.norm(waypoint.point - drone.pos)), params)

            if world.time_s >= next_sample_time:
                view = camera.view_matrix(drone.pos, drone.quat)
                labels = compute_gate_labels(
                    oracle, world.client, drone.pos, view, camera.intrinsics
                )
                detections = [
                    Detection(
                        label.class_id, 1.0, label.box_xyxy, label.keypoints, label.visibility
                    )
                    for label in labels
                ]
                log.record(world.time_s, drone.pos, drone.quat, waypoint.point, state, detections)
                if keep_crash_frames:
                    recent_frames.append(camera.render(drone.pos, drone.quat).rgb)
                next_sample_time += 1.0 / CAMERA_HZ

            world.step(rpm)
            cur_pos = world.drone_state().pos

            if next_gate < oracle.gate_count:
                crossing = detect_crossing(prev_pos, cur_pos, oracle.gate_pose(next_gate))
                if crossing is not None and crossing.kind is CrossingKind.PASSED:
                    log.gate_pass_times.append(world.time_s)
                    next_gate += 1
                    if next_gate == oracle.gate_count:
                        log.end, log.end_time = FlightEnd.FINISHED, world.time_s
                        break
            prev_pos = cur_pos

            hit = _crash_contact(world, oracle)
            if hit is not None:
                log.end, log.end_time, log.crash_body_kind = FlightEnd.CRASH, world.time_s, hit
                if keep_crash_frames:
                    log.crash_frames = list(recent_frames)
                break

        if log.end is None:
            log.end, log.end_time = FlightEnd.TIMEOUT, world.time_s
        return log
    finally:
        world.close()


def _crash_contact(world: World, oracle: Oracle) -> str | None:
    for contact in p.getContactPoints(bodyA=world.drone_id, physicsClientId=world.client):
        body, link = contact[2], contact[4]
        if body == world.ground_id:
            if world.time_s > GROUND_CONTACT_GRACE_S:
                return "ground"
            continue
        if oracle.gate_index_of_body(body) is not None:
            return "gate_post" if link == GATE_POST_LINK_INDEX else "gate_frame"
        return "object"
    return None
