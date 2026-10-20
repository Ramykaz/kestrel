"""Synthetic frame and label generation, and the YOLO pose dataset writer."""

from __future__ import annotations

import json
import math
import multiprocessing as mp
import subprocess
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pybullet as p

from kestrel.data.labels import CLASS_NAMES, compute_gate_labels, yolo_pose_lines
from kestrel.data.postfx import apply_effects, encode_jpeg
from kestrel.data.randomize import (
    DomainRandomization,
    sample_distractors,
    sample_effects,
    sample_lighting,
    sample_style,
)
from kestrel.paths import REPO_ROOT
from kestrel.sim.camera import HEIGHT, WIDTH, OnboardCamera, onboard_intrinsics
from kestrel.sim.course import ensure_training_seed, generate_course, load_preset
from kestrel.sim.oracle import Oracle
from kestrel.sim.scene import Distractor, add_distractor, apply_style
from kestrel.sim.world import World, egl_available

SCATTER_DISTANCE_M = (0.5, 10.0)
SCATTER_BEARING_DEG = 60.0
SCATTER_HEIGHT_JITTER_M = 1.0
SCATTER_HEIGHT_RANGE_M = (0.3, 5.5)
SCATTER_YAW_SIGMA_DEG = 20.0
SCATTER_PITCH_SIGMA_DEG = 10.0
SCATTER_ROLL_DEG = 25.0
# Share of scatter frames taken from behind the chosen gate, so gate_back has examples.
BACK_VIEW_SHARE = 0.15
# Training course seeds come from below the held-out suite's seed range.
TRAINING_SEED_RANGE = (0, 800_000)
DISTRACTOR_CAMERA_CLEARANCE_M = 0.3
MAX_POSE_TRIES = 100
VAL_SHARE = 0.1
# Horizontal flip swaps left and right corners: TL <-> TR, BR <-> BL.
FLIP_INDEX = [1, 0, 3, 2]


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


def _inside_distractor(pos: np.ndarray, distractors: list[Distractor]) -> bool:
    for d in distractors:
        horizontal = math.dist(pos[:2], d.position)
        clearance = DISTRACTOR_CAMERA_CLEARANCE_M
        if horizontal < d.footprint_radius + clearance and pos[2] < d.size[1] + clearance:
            return True
    return False


@dataclass(frozen=True)
class CourseJob:
    first_frame: int
    frames: int
    course_seed: int
    preset: str
    split: str
    rng_seed: int


@dataclass(frozen=True)
class JobStats:
    frames: int
    labels_per_class: dict[int, int]
    empty_frames: int


def plan_jobs(dr: DomainRandomization, total_frames: int, seed: int) -> list[CourseJob]:
    """Split a dataset into one job per course. Whole courses go to train or val."""
    rng = np.random.default_rng(seed)
    names = list(dr.course_presets)
    weights = np.array([dr.course_presets[n] for n in names])
    course_count = math.ceil(total_frames / dr.frames_per_course)
    val_courses = set(rng.choice(course_count, max(1, round(course_count * VAL_SHARE)), False))
    child_seeds = np.random.SeedSequence(seed).spawn(course_count)
    jobs = []
    for i in range(course_count):
        course_seed = int(rng.integers(*TRAINING_SEED_RANGE))
        ensure_training_seed(course_seed)
        first = i * dr.frames_per_course
        jobs.append(
            CourseJob(
                first_frame=first,
                frames=min(dr.frames_per_course, total_frames - first),
                course_seed=course_seed,
                preset=str(rng.choice(names, p=weights / weights.sum())),
                split="val" if i in val_courses else "train",
                rng_seed=int(child_seeds[i].generate_state(1)[0]),
            )
        )
    return jobs


def render_job(job: CourseJob, dr: DomainRandomization, out: Path, use_egl: bool) -> JobStats:
    """Render one course's frames with randomized looks and write images and labels."""
    rng = np.random.default_rng(job.rng_seed)
    course = generate_course(job.course_seed, load_preset(job.preset))
    style, palette = sample_style(rng, dr)
    distractors = sample_distractors(rng, dr, course)
    world = World(use_egl=use_egl)
    try:
        if use_egl and not world.uses_egl:
            raise RuntimeError("EGL loaded in the parent process but not in this worker")
        oracle = Oracle(course, world.load_course(course))
        apply_style(world.client, world.ground_id, oracle.gate_body_ids, style)
        for distractor in distractors:
            add_distractor(world.client, distractor)
        camera = OnboardCamera(world.client, world.renderer)
        labels_per_class = dict.fromkeys(CLASS_NAMES, 0)
        empty = 0
        for offset in range(job.frames):
            for _ in range(MAX_POSE_TRIES):
                pos, quat = sample_scatter_pose(rng, oracle)
                if not _inside_distractor(pos, distractors):
                    break
            else:
                raise RuntimeError(f"no camera pose clear of distractors on course {course.seed}")
            world.reset_drone(pos)
            frame_style = replace(style, lighting=sample_lighting(rng, dr, palette))
            frame = camera.render(pos, quat, style=frame_style)
            labels = compute_gate_labels(
                oracle, world.client, pos, frame.view_matrix, camera.intrinsics, frame.segmentation
            )
            effects = sample_effects(rng, dr, camera.intrinsics)
            image = apply_effects(frame.rgb, effects, rng)
            name = f"{job.first_frame + offset:06d}"
            (out / "images" / job.split / f"{name}.jpg").write_bytes(
                encode_jpeg(image, effects.jpeg_quality)
            )
            lines = yolo_pose_lines(labels, WIDTH, HEIGHT)
            (out / "labels" / job.split / f"{name}.txt").write_text(
                "\n".join(lines) + ("\n" if lines else "")
            )
            for label in labels:
                labels_per_class[label.class_id] += 1
            empty += not labels
        return JobStats(job.frames, labels_per_class, empty)
    finally:
        world.close()


def _render_job_args(args: tuple) -> JobStats:
    return render_job(*args)


def _git_commit() -> tuple[str, bool]:
    root = REPO_ROOT
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=True
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=root,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    )
    return commit, dirty


def write_dataset(
    out: Path,
    dr: DomainRandomization,
    total_frames: int,
    seed: int,
    workers: int,
    use_egl: bool = False,
) -> dict:
    """Write a YOLO pose dataset with data.yaml and manifest.json, and return the manifest.

    Datasets are immutable, so an existing non-empty directory is refused. With use_egl the
    dataset uses EGL only if it is available, and the manifest records what rendered it.
    """
    use_egl = use_egl and egl_available()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"{out} already holds files; datasets are never overwritten")
    commit, dirty = _git_commit()
    for kind in ("images", "labels"):
        for split in ("train", "val"):
            (out / kind / split).mkdir(parents=True, exist_ok=True)
    jobs = plan_jobs(dr, total_frames, seed)
    started = datetime.now(UTC)
    args = [(job, dr, out, use_egl) for job in jobs]
    if workers <= 1:
        stats = [render_job(*a) for a in args]
    else:
        with mp.get_context("spawn").Pool(workers) as pool:
            stats = pool.map(_render_job_args, args, chunksize=1)
    seconds = (datetime.now(UTC) - started).total_seconds()

    (out / "data.yaml").write_text(
        f"path: {out.resolve().as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "kpt_shape: [4, 3]\n"
        f"flip_idx: {FLIP_INDEX}\n"
        f"names: {json.dumps([CLASS_NAMES[c] for c in sorted(CLASS_NAMES)])}\n"
    )
    split_frames = {
        split: sum(s.frames for s, j in zip(stats, jobs, strict=True) if j.split == split)
        for split in ("train", "val")
    }
    manifest = {
        "generator_version": version("kestrel"),
        "git_commit": commit,
        "git_dirty": dirty,
        "created_at": started.isoformat(timespec="seconds"),
        "seed": seed,
        "renderer": "egl" if use_egl else "tiny",
        "intrinsics": onboard_intrinsics(
            p.ER_BULLET_HARDWARE_OPENGL if use_egl else p.ER_TINY_RENDERER
        ).to_dict(),
        "domain_randomization": dr.to_dict(),
        "counts": {
            "frames": split_frames,
            "labels": {
                CLASS_NAMES[c]: sum(s.labels_per_class[c] for s in stats) for c in CLASS_NAMES
            },
            "empty_frames": sum(s.empty_frames for s in stats),
            "courses": len(jobs),
        },
        "courses": [
            {"seed": j.course_seed, "preset": j.preset, "split": j.split, "frames": j.frames}
            for j in jobs
        ],
        "render_seconds": round(seconds, 1),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest
