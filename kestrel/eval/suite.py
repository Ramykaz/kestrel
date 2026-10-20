"""Held-out courses, the fixed way the camera sees them, and parallel flights over them."""

from __future__ import annotations

import functools
import multiprocessing as mp
from dataclasses import dataclass
from pathlib import Path

from kestrel.perception.detect import GateDetector
from kestrel.pilot.controller import OwnState
from kestrel.pilot.flight_log import FlightLog
from kestrel.pilot.fly import CameraPilot
from kestrel.pilot.params import PilotParams
from kestrel.sim.camera import OnboardCamera
from kestrel.sim.course import (
    Course,
    ensure_training_seed,
    eval_suite_seeds,
    generate_course,
    load_preset,
)
from kestrel.sim.flight import CameraLook, fly_course
from kestrel.sim.oracle import Oracle
from kestrel.sim.scene import Lighting, SceneStyle
from kestrel.sim.world import World

# Fixed looks for evaluation flights. They never change between rounds, so every round is
# measured on identical images apart from motion blur, which follows the drone's own motion.
EVAL_LOOKS = {
    "day": CameraLook(
        style=SceneStyle(
            sky_horizon=(0.80, 0.86, 0.92),
            sky_zenith=(0.45, 0.62, 0.85),
            ground_pattern="grass",
            ground_colors=((0.30, 0.42, 0.22), (0.45, 0.55, 0.30)),
            gate_color=(0.95, 0.45, 0.10),
            gate_pattern="solid",
            gate_pattern_color=(0.95, 0.45, 0.10),
            lighting=Lighting((0.4, -0.3, 0.85), (1.0, 0.97, 0.92), 0.6, 0.6, 0.05),
            texture_seed=1,
        ),
        exposure_ev=0.0,
        noise_sigma=2.0,
        jpeg_quality=85,
        max_blur_px=9.0,
    ),
    "night": CameraLook(
        style=SceneStyle(
            sky_horizon=(0.10, 0.12, 0.20),
            sky_zenith=(0.02, 0.03, 0.08),
            ground_pattern="asphalt",
            ground_colors=((0.10, 0.10, 0.12), (0.22, 0.22, 0.25)),
            gate_color=(0.95, 0.45, 0.10),
            gate_pattern="solid",
            gate_pattern_color=(0.95, 0.45, 0.10),
            lighting=Lighting((-0.3, 0.5, 0.8), (0.55, 0.60, 0.85), 0.25, 0.35, 0.0),
            texture_seed=2,
        ),
        exposure_ev=0.0,
        noise_sigma=6.0,
        jpeg_quality=75,
        max_blur_px=9.0,
    ),
}
# Training seeds for the straight-course acceptance check of SPEC 14, phase 2.
STRAIGHT_CHECK_SEEDS = tuple(range(1000, 1010))


def eval_courses() -> list[Course]:
    """The 20 tech and 10 night held-out courses, in suite order."""
    seeds = eval_suite_seeds()
    return [generate_course(s, load_preset(preset)) for preset in seeds for s in seeds[preset]]


def straight_check_courses() -> list[Course]:
    for seed in STRAIGHT_CHECK_SEEDS:
        ensure_training_seed(seed)
    return [generate_course(s, load_preset("straight")) for s in STRAIGHT_CHECK_SEEDS]


@functools.cache
def _detector(onnx_path: str) -> GateDetector:
    return GateDetector(Path(onnx_path))


@dataclass(frozen=True)
class FlightJob:
    course: Course
    onnx_path: str
    params: PilotParams
    keep_crash_frames: bool


def fly_camera_pilot(job: FlightJob) -> FlightLog:
    """Fly one course with the camera-only pilot under the course's fixed look."""
    detector = _detector(job.onnx_path)

    def make_pilot(world: World, oracle: Oracle, camera: OnboardCamera, own: OwnState):
        return CameraPilot(job.params, detector, camera.intrinsics, own)

    look = EVAL_LOOKS[load_preset(job.course.preset).lighting]
    return fly_course(
        job.course, make_pilot, look, seed=job.course.seed, keep_crash_frames=job.keep_crash_frames
    )


def run_flights(
    courses: list[Course],
    onnx_path: Path,
    params: PilotParams,
    workers: int,
    keep_crash_frames: bool = True,
) -> list[FlightLog]:
    """One flight per course, in course order."""
    jobs = [FlightJob(c, str(onnx_path), params, keep_crash_frames) for c in courses]
    if workers <= 1:
        return [fly_camera_pilot(job) for job in jobs]
    with mp.get_context("spawn").Pool(workers) as pool:
        return pool.map(fly_camera_pilot, jobs, chunksize=1)
