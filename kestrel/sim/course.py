"""Seeded course generator."""

from __future__ import annotations

import math
import zlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs"

VOLUME_HALF_XY_M = 15.0
VOLUME_HEIGHT_M = 6.0
WALL_MARGIN_M = 1.0
SPACING_M = (3.0, 7.0)
MIN_GATE_DISTANCE_M = 3.0
# Gate frames are 0.8 m wide, so a path passing 1.9 m from a gate center keeps 1.5 m clear.
MIN_PATH_CLEARANCE_M = 1.9
CONE_HALF_ANGLE_DEG = 35.0
MIN_IN_CONE_SHARE = 0.5
BEARING_JITTER_DEG = 10.0
GATE_HEIGHT_RANGE_M = (1.0, 3.5)
FIRST_GATE_HEIGHT_M = 1.5
START_DISTANCE_M = (3.0, 5.0)
START_YAW_JITTER_DEG = 30.0
MAX_ATTEMPTS = 1000


@dataclass(frozen=True)
class CoursePreset:
    name: str
    gates: int
    turn_deg: tuple[float, float]
    alternate_turns: bool
    height_change_m: float
    lighting: str


@dataclass(frozen=True)
class GatePose:
    center: tuple[float, float, float]
    yaw: float

    @property
    def forward(self) -> np.ndarray:
        return np.array([math.cos(self.yaw), math.sin(self.yaw), 0.0])


@dataclass(frozen=True)
class Course:
    seed: int
    preset: str
    gates: tuple[GatePose, ...]
    start_pos: tuple[float, float, float]
    start_yaw: float

    def to_dict(self) -> dict:
        return {
            "seed": self.seed,
            "preset": self.preset,
            "gates": [{"center": list(g.center), "yaw": g.yaw} for g in self.gates],
            "start_pos": list(self.start_pos),
            "start_yaw": self.start_yaw,
        }


def load_preset(name: str) -> CoursePreset:
    path = CONFIG_DIR / "course" / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"no course preset named {name!r} at {path}")
    raw = yaml.safe_load(path.read_text())
    return CoursePreset(
        name=name,
        gates=int(raw["gates"]),
        turn_deg=(float(raw["turn_deg"][0]), float(raw["turn_deg"][1])),
        alternate_turns=bool(raw["alternate_turns"]),
        height_change_m=float(raw["height_change_m"]),
        lighting=str(raw["lighting"]),
    )


def _point_segment_distance(point: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    ab = b - a
    t = np.clip(np.dot(point - a, ab) / np.dot(ab, ab), 0.0, 1.0)
    return float(np.linalg.norm(point - (a + t * ab)))


def _inside_volume(xy: np.ndarray) -> bool:
    return bool(np.all(np.abs(xy) <= VOLUME_HALF_XY_M - WALL_MARGIN_M))


def _try_layout(rng: np.random.Generator, preset: CoursePreset) -> Course | None:
    heading = rng.uniform(-math.pi, math.pi)
    centers = [np.array([*rng.uniform(-8.0, 8.0, 2), FIRST_GATE_HEIGHT_M])]
    yaws = [heading]
    bearings_deg = []
    sign = rng.choice([-1.0, 1.0])
    for _ in range(preset.gates - 1):
        if not preset.alternate_turns:
            sign = rng.choice([-1.0, 1.0])
        turn_deg = sign * rng.uniform(*preset.turn_deg)
        # A constant-curvature turn of angle theta leaves along a chord at theta / 2.
        bearing_deg = turn_deg / 2 + rng.uniform(-BEARING_JITTER_DEG, BEARING_JITTER_DEG)
        travel = heading + math.radians(bearing_deg)
        z = centers[-1][2] + rng.uniform(-preset.height_change_m, preset.height_change_m)
        z = float(np.clip(z, *GATE_HEIGHT_RANGE_M))
        # Spacing is a 3D distance; the climb is part of it.
        spacing = rng.uniform(*SPACING_M)
        horizontal = math.sqrt(spacing**2 - (z - centers[-1][2]) ** 2)
        xy = centers[-1][:2] + horizontal * np.array([math.cos(travel), math.sin(travel)])
        if not _inside_volume(xy):
            return None
        heading += math.radians(turn_deg)
        centers.append(np.array([*xy, z]))
        yaws.append(heading)
        bearings_deg.append(bearing_deg)
        if preset.alternate_turns:
            sign = -sign

    in_cone = sum(abs(b) <= CONE_HALF_ANGLE_DEG for b in bearings_deg)
    if in_cone < MIN_IN_CONE_SHARE * len(bearings_deg):
        return None
    for i, a in enumerate(centers):
        for b in centers[i + 1 :]:
            if np.linalg.norm(a - b) < MIN_GATE_DISTANCE_M:
                return None
    for i in range(len(centers) - 1):
        for j, other in enumerate(centers):
            if j in (i, i + 1):
                continue
            if _point_segment_distance(other, centers[i], centers[i + 1]) < MIN_PATH_CLEARANCE_M:
                return None

    first_forward = np.array([math.cos(yaws[0]), math.sin(yaws[0])])
    start_xy = centers[0][:2] - rng.uniform(*START_DISTANCE_M) * first_forward
    if not _inside_volume(start_xy):
        return None
    start_yaw = yaws[0] + math.radians(rng.uniform(-START_YAW_JITTER_DEG, START_YAW_JITTER_DEG))
    gates = tuple(
        GatePose((float(c[0]), float(c[1]), float(c[2])), float(math.remainder(y, math.tau)))
        for c, y in zip(centers, yaws, strict=True)
    )
    return Course(
        seed=-1,
        preset=preset.name,
        gates=gates,
        start_pos=(float(start_xy[0]), float(start_xy[1]), 0.05),
        start_yaw=float(math.remainder(start_yaw, math.tau)),
    )


def generate_course(seed: int, preset: CoursePreset) -> Course:
    """Deterministic course for (seed, preset). Raises if no valid layout is found."""
    rng = np.random.default_rng([seed, zlib.crc32(preset.name.encode())])
    for _ in range(MAX_ATTEMPTS):
        layout = _try_layout(rng, preset)
        if layout is not None:
            return Course(seed, layout.preset, layout.gates, layout.start_pos, layout.start_yaw)
    raise RuntimeError(
        f"no valid {preset.name} course for seed {seed} after {MAX_ATTEMPTS} attempts"
    )


def eval_suite_seeds() -> dict[str, list[int]]:
    raw = yaml.safe_load((CONFIG_DIR / "eval_suite.yaml").read_text())
    return {preset: [int(s) for s in seeds] for preset, seeds in raw.items()}


def ensure_training_seed(seed: int) -> None:
    """Raise if a seed belongs to the held-out evaluation suite."""
    for preset, seeds in eval_suite_seeds().items():
        if seed in seeds:
            raise ValueError(f"seed {seed} is a held-out {preset} eval seed")
