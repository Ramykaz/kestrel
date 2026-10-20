"""Synthetic frame and label generation."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pybullet as p

from kestrel.data.labels import GateLabel, compute_gate_labels
from kestrel.sim.camera import OnboardCamera
from kestrel.sim.course import Course
from kestrel.sim.oracle import Oracle
from kestrel.sim.world import World

SCATTER_DISTANCE_M = (0.5, 10.0)
SCATTER_BEARING_DEG = 60.0
SCATTER_HEIGHT_JITTER_M = 1.0
SCATTER_HEIGHT_RANGE_M = (0.3, 5.5)
SCATTER_YAW_SIGMA_DEG = 20.0
SCATTER_PITCH_SIGMA_DEG = 10.0
SCATTER_ROLL_DEG = 25.0
# Share of scatter frames taken from behind the chosen gate, so gate_back has examples.
BACK_VIEW_SHARE = 0.15


@dataclass(frozen=True)
class LabeledFrame:
    rgb: np.ndarray
    labels: list[GateLabel]
    camera_pos: np.ndarray
    camera_quat: np.ndarray


def sample_scatter_pose(rng: np.random.Generator, oracle: Oracle) -> tuple[np.ndarray, np.ndarray]:
    """A drone pose near a random gate, roughly looking at it the way a racing drone would."""
    gate = oracle.gate_pose(int(rng.integers(oracle.gate_count)))
    center = np.array(gate.center)
    from_behind = rng.random() < BACK_VIEW_SHARE
    side_yaw = gate.yaw if from_behind else gate.yaw + math.pi
    bearing = side_yaw + math.radians(rng.uniform(-SCATTER_BEARING_DEG, SCATTER_BEARING_DEG))
    distance = rng.uniform(*SCATTER_DISTANCE_M)
    pos = center + distance * np.array([math.cos(bearing), math.sin(bearing), 0.0])
    pos[2] = np.clip(
        center[2] + rng.uniform(-SCATTER_HEIGHT_JITTER_M, SCATTER_HEIGHT_JITTER_M),
        *SCATTER_HEIGHT_RANGE_M,
    )
    to_gate = center - pos
    yaw = math.atan2(to_gate[1], to_gate[0]) + math.radians(rng.normal(0, SCATTER_YAW_SIGMA_DEG))
    elevation = math.atan2(to_gate[2], np.linalg.norm(to_gate[:2]))
    # Positive pitch tips the nose down; the camera itself is pitched up 15 degrees.
    pitch = math.radians(15.0) - elevation + math.radians(rng.normal(0, SCATTER_PITCH_SIGMA_DEG))
    roll = math.radians(rng.uniform(-SCATTER_ROLL_DEG, SCATTER_ROLL_DEG))
    return pos, np.array(p.getQuaternionFromEuler([roll, pitch, yaw]))


def render_scatter_frames(
    course: Course, count: int, rng: np.random.Generator, use_egl: bool = False
) -> list[LabeledFrame]:
    """Load the course into a fresh world and render count labeled scatter frames."""
    world = World(use_egl=use_egl)
    try:
        oracle = Oracle(course, world.load_course(course))
        camera = OnboardCamera(world.client, world.renderer)
        frames = []
        for _ in range(count):
            pos, quat = sample_scatter_pose(rng, oracle)
            # The drone sits at the camera so its collision shape cannot block label rays.
            world.reset_drone(pos)
            frame = camera.render(pos, quat)
            labels = compute_gate_labels(
                oracle, world.client, pos, frame.view_matrix, camera.intrinsics
            )
            frames.append(LabeledFrame(frame.rgb, labels, pos, quat))
        return frames
    finally:
        world.close()
