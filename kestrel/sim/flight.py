"""One flight on one course: physics, camera, pilot, gate passes and crashes.

The simulator side knows the course; the pilot only gets its own state and camera frames.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

import numpy as np
import pybullet as p

from kestrel.data.postfx import CameraEffects, degrade, motion_blur_vector
from kestrel.perception.detect import Detection
from kestrel.pilot.controller import OwnState
from kestrel.pilot.flight_log import FlightEnd, FlightLog, PilotState
from kestrel.sim.camera import OnboardCamera
from kestrel.sim.course import Course
from kestrel.sim.crossing import CrossingKind, detect_crossing
from kestrel.sim.gates import GATE_POST_LINK_INDEX
from kestrel.sim.oracle import Oracle
from kestrel.sim.scene import SceneStyle, apply_style
from kestrel.sim.world import CAMERA_HZ, CONTROL_HZ, PHYSICS_HZ, World

TIMEOUT_S = 90.0
CRASH_FRAME_SECONDS = 2.0
# Contacts with the ground before this time are the drone resting on it before takeoff.
GROUND_CONTACT_GRACE_S = 1.0
PHYSICS_PER_CONTROL = PHYSICS_HZ // CONTROL_HZ
PHYSICS_PER_FRAME = PHYSICS_HZ // CAMERA_HZ


class Pilot(Protocol):
    needs_frames: bool
    state: PilotState
    goal: np.ndarray

    @property
    def believed_gate(self) -> np.ndarray | None: ...

    def perceive(
        self, time_s: float, own: OwnState, frame: np.ndarray | None
    ) -> list[Detection]: ...

    def control(self, own: OwnState, dt: float) -> np.ndarray: ...


PilotFactory = Callable[[World, Oracle, OnboardCamera, OwnState], Pilot]


@dataclass(frozen=True)
class CameraLook:
    """How the onboard camera sees a course: scene style plus fixed camera effects. Motion
    blur comes from the drone's actual motion."""

    style: SceneStyle
    exposure_ev: float
    noise_sigma: float
    jpeg_quality: int
    max_blur_px: float


def own_state(world: World) -> OwnState:
    drone = world.drone_state()
    return OwnState(drone.pos, drone.quat, drone.vel, drone.ang_vel)


def _camera_effects(
    look: CameraLook, own: OwnState, view: tuple[float, ...], camera: OnboardCamera
) -> CameraEffects:
    world_to_cv = np.diag([1.0, -1.0, -1.0]) @ np.array(view).reshape(4, 4, order="F")[:3, :3]
    blur = motion_blur_vector(
        world_to_cv @ own.vel, world_to_cv @ own.ang_vel, camera.intrinsics, look.max_blur_px
    )
    return CameraEffects(look.exposure_ev, blur, look.noise_sigma, look.jpeg_quality)


def _crash_contact(world: World, oracle: Oracle) -> tuple[str, int | None] | None:
    for contact in p.getContactPoints(bodyA=world.drone_id, physicsClientId=world.client):
        body, link = contact[2], contact[4]
        if body == world.ground_id:
            if world.time_s > GROUND_CONTACT_GRACE_S:
                return "ground", None
            continue
        gate_index = oracle.gate_index_of_body(body)
        if gate_index is not None:
            return ("gate_post" if link == GATE_POST_LINK_INDEX else "gate_frame"), gate_index
        return "object", None
    return None


def fly_course(
    course: Course,
    make_pilot: PilotFactory,
    look: CameraLook | None = None,
    seed: int = 0,
    keep_crash_frames: bool = False,
    use_egl: bool = False,
) -> FlightLog:
    """Fly one course until the last gate is passed, the drone hits something, or 90 s pass."""
    rng = np.random.default_rng(seed)
    world = World(use_egl=use_egl)
    try:
        oracle = Oracle(course, world.load_course(course))
        if look is not None:
            apply_style(world.client, world.ground_id, oracle.gate_body_ids, look.style)
        camera = OnboardCamera(world.client, world.renderer)
        world.reset_drone(np.array(course.start_pos), course.start_yaw)
        pilot = make_pilot(world, oracle, camera, own_state(world))
        log = FlightLog(course.seed, course.preset, sample_hz=CAMERA_HZ)
        recent_frames: deque[np.ndarray] = deque(maxlen=int(CRASH_FRAME_SECONDS * CAMERA_HZ))
        render = pilot.needs_frames or keep_crash_frames
        next_gate = 0
        rpm = np.zeros(4)
        prev_pos = world.drone_state().pos

        for step in range(int(TIMEOUT_S * PHYSICS_HZ)):
            own = own_state(world)
            if step % PHYSICS_PER_FRAME == 0:
                frame = None
                if render:
                    rendered = camera.render(
                        own.pos, own.quat, style=None if look is None else look.style
                    )
                    frame = rendered.rgb
                    if look is not None:
                        effects = _camera_effects(look, own, rendered.view_matrix, camera)
                        frame = degrade(frame, effects, rng)
                    if keep_crash_frames:
                        recent_frames.append(frame)
                detections = pilot.perceive(world.time_s, own, frame)
                log.record(
                    world.time_s,
                    own.pos,
                    own.quat,
                    pilot.goal,
                    pilot.state,
                    detections,
                    pilot.believed_gate,
                )
            if step % PHYSICS_PER_CONTROL == 0:
                rpm = pilot.control(own, 1.0 / CONTROL_HZ)

            world.step(rpm)
            cur_pos = world.drone_state().pos
            crossing = detect_crossing(prev_pos, cur_pos, oracle.gate_pose(next_gate))
            prev_pos = cur_pos
            if crossing is not None and crossing.kind is CrossingKind.PASSED:
                log.gate_pass_times.append(world.time_s)
                next_gate += 1
                if next_gate == oracle.gate_count:
                    log.end, log.end_time = FlightEnd.FINISHED, world.time_s
                    break

            hit = _crash_contact(world, oracle)
            if hit is not None:
                log.end, log.end_time = FlightEnd.CRASH, world.time_s
                log.crash_body_kind, log.crash_gate_index = hit
                if keep_crash_frames:
                    log.crash_frames = list(recent_frames)
                break

        if log.end is None:
            log.end, log.end_time = FlightEnd.TIMEOUT, world.time_s
        return log
    finally:
        world.close()
