"""Smoke test and render benchmark for the simulator."""

from __future__ import annotations

import multiprocessing as mp
import time
from dataclasses import dataclass

import numpy as np
import pybullet as p
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.utils.enums import DroneModel

from kestrel.sim.camera import OnboardCamera
from kestrel.sim.world import CAMERA_HZ, CONTROL_HZ, PHYSICS_HZ, World

HOVER_TOLERANCE_M = 0.10


@dataclass(frozen=True)
class SmokeResult:
    sim_seconds: float
    wall_seconds: float
    final_error_m: float
    frames_rendered: int
    render_seconds: float
    renderer: str
    last_frame: np.ndarray

    @property
    def passed(self) -> bool:
        return self.final_error_m < HOVER_TOLERANCE_M

    @property
    def render_fps(self) -> float:
        return self.frames_rendered / self.render_seconds


def run_hover(sim_seconds: float = 5.0, use_egl: bool = False) -> SmokeResult:
    """Take off to 1 m, hover with DSLPIDControl, render the onboard camera at 30 Hz."""
    world = World(use_egl=use_egl)
    try:
        camera = OnboardCamera(world.client, world.renderer)
        controller = DSLPIDControl(drone_model=DroneModel.CF2X)
        target = np.array([0.0, 0.0, 1.0])
        physics_per_control = PHYSICS_HZ // CONTROL_HZ
        total_steps = int(sim_seconds * PHYSICS_HZ)
        rpm = np.zeros(4)
        frames = 0
        render_seconds = 0.0
        next_frame_time = 0.0
        last_frame = np.zeros((camera.intrinsics.height, camera.intrinsics.width, 3), np.uint8)
        start = time.perf_counter()
        for step in range(total_steps):
            state = world.drone_state()
            if step % physics_per_control == 0:
                rpm, _, _ = controller.computeControl(
                    control_timestep=1.0 / CONTROL_HZ,
                    cur_pos=state.pos,
                    cur_quat=state.quat,
                    cur_vel=state.vel,
                    cur_ang_vel=state.ang_vel,
                    target_pos=target,
                )
            if world.time_s >= next_frame_time:
                render_start = time.perf_counter()
                last_frame = camera.render(state.pos, state.quat).rgb
                render_seconds += time.perf_counter() - render_start
                frames += 1
                next_frame_time += 1.0 / CAMERA_HZ
            world.step(rpm)
        wall = time.perf_counter() - start
        final_error = float(np.linalg.norm(world.drone_state().pos - target))
        return SmokeResult(
            sim_seconds=sim_seconds,
            wall_seconds=wall,
            final_error_m=final_error,
            frames_rendered=frames,
            render_seconds=render_seconds,
            renderer="egl" if world.uses_egl else "tiny",
            last_frame=last_frame,
        )
    finally:
        world.close()


def _add_bench_distractors(client: int, rng: np.random.Generator, count: int) -> None:
    for _ in range(count):
        half_extents = rng.uniform([0.05, 0.05, 0.3], [0.4, 0.4, 1.5])
        color = [*rng.uniform(0.1, 0.9, 3).tolist(), 1.0]
        shape = p.createVisualShape(
            p.GEOM_BOX, halfExtents=half_extents.tolist(), rgbaColor=color, physicsClientId=client
        )
        position = [*rng.uniform(-12, 12, 2).tolist(), float(half_extents[2])]
        p.createMultiBody(0, -1, shape, position, physicsClientId=client)


def _bench_worker(args: tuple[int, int, bool]) -> tuple[int, float, str]:
    seed, frames, use_egl = args
    rng = np.random.default_rng(seed)
    world = World(use_egl=use_egl)
    try:
        _add_bench_distractors(world.client, rng, count=12)
        camera = OnboardCamera(world.client, world.renderer)
        start = time.perf_counter()
        for _ in range(frames):
            pos = np.array([*rng.uniform(-10, 10, 2), rng.uniform(0.5, 3.0)])
            quat = np.array(p.getQuaternionFromEuler([0.0, 0.0, rng.uniform(-np.pi, np.pi)]))
            camera.render(pos, quat)
        return frames, time.perf_counter() - start, "egl" if world.uses_egl else "tiny"
    finally:
        world.close()


@dataclass(frozen=True)
class BenchResult:
    renderer: str
    workers: int
    frames: int
    wall_seconds: float

    @property
    def fps(self) -> float:
        return self.frames / self.wall_seconds


def run_render_bench(frames_per_worker: int, workers: int, use_egl: bool) -> BenchResult:
    """Render random onboard views of a ground plane with 12 box distractors."""
    jobs = [(seed, frames_per_worker, use_egl) for seed in range(workers)]
    start = time.perf_counter()
    if workers == 1:
        outcomes = [_bench_worker(jobs[0])]
    else:
        with mp.get_context("spawn").Pool(workers) as pool:
            outcomes = pool.map(_bench_worker, jobs)
    wall = time.perf_counter() - start
    renderers = {renderer for _, _, renderer in outcomes}
    if len(renderers) != 1:
        raise RuntimeError(f"workers used different renderers: {sorted(renderers)}")
    return BenchResult(renderers.pop(), workers, sum(n for n, _, _ in outcomes), wall)
