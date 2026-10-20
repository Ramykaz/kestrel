"""Domain randomization: presets from configs/dr and the sampler that draws scenes from them.

Every range in a preset must sit inside the hard limits below. Agents may edit presets; they
cannot edit these limits.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import yaml

from kestrel.data.postfx import CameraEffects, motion_blur_vector
from kestrel.sim.camera import Intrinsics
from kestrel.sim.course import CONFIG_DIR, VOLUME_HALF_XY_M, Course
from kestrel.sim.scene import (
    DISTRACTOR_KINDS,
    GATE_PATTERNS,
    GROUND_PATTERNS,
    Distractor,
    Lighting,
    SceneStyle,
    hsv,
)

HARD_LIMITS = {
    "frames_per_course": (1, 64),
    "gate_hue": (0.0, 1.0),
    "gate_saturation": (0.0, 1.0),
    "gate_value": (0.1, 1.0),
    "ground_value": (0.05, 0.95),
    "light_intensity": (0.1, 1.5),
    "distractors": (0, 20),
    # SPEC 6.1: the drone never flies faster than 4 m/s.
    "speed_m_s": (0.0, 4.0),
    "angular_rate_rad_s": (0.0, 6.0),
    "motion_blur_px": (0, 12),
    "noise_sigma": (0, 12),
    "exposure_ev": (-2.5, 1.5),
    "jpeg_quality": (30, 100),
}

PALETTES = {
    "day": {
        "horizon": (0.80, 0.86, 0.92),
        "zenith": (0.45, 0.62, 0.85),
        "tint": (1.0, 0.97, 0.92),
        "ambient": 0.6,
        "diffuse": 0.6,
    },
    "dusk": {
        "horizon": (0.95, 0.65, 0.45),
        "zenith": (0.35, 0.35, 0.60),
        "tint": (1.0, 0.75, 0.55),
        "ambient": 0.45,
        "diffuse": 0.5,
    },
    "night": {
        "horizon": (0.10, 0.12, 0.20),
        "zenith": (0.02, 0.03, 0.08),
        "tint": (0.55, 0.60, 0.85),
        "ambient": 0.25,
        "diffuse": 0.35,
    },
}
SKY_JITTER = 0.06
# Distractors keep this much clearance from every gate (SPEC 6.3).
GATE_CLEARANCE_M = 1.5
DISTRACTOR_SIZES_M = {
    "pole": ((0.05, 0.2), (1.0, 4.0)),
    "box": ((0.3, 1.5), (0.3, 1.5)),
    "cone": ((0.3, 0.6), (0.4, 1.0)),
}
MAX_PLACEMENT_TRIES = 50

Range = tuple[float, float]


@dataclass(frozen=True)
class DomainRandomization:
    name: str
    course_presets: dict[str, float]
    frames_per_course: int
    gate_hue: Range
    gate_saturation: Range
    gate_value: Range
    gate_patterns: tuple[str, ...]
    ground_patterns: tuple[str, ...]
    ground_value: Range
    palettes: tuple[str, ...]
    light_intensity: Range
    distractors: tuple[int, int]
    speed_m_s: Range
    angular_rate_rad_s: Range
    motion_blur_px: Range
    noise_sigma: Range
    exposure_ev: Range
    jpeg_quality: tuple[int, int]

    def to_dict(self) -> dict:
        return asdict(self)


def _check_range(name: str, value: list) -> tuple:
    if len(value) != 2 or value[0] > value[1]:
        raise ValueError(f"{name} must be [low, high] with low <= high, got {value}")
    low, high = HARD_LIMITS[name]
    if value[0] < low or value[1] > high:
        raise ValueError(f"{name} {value} is outside the hard limits [{low}, {high}]")
    return tuple(value)


def _check_choices(name: str, value: list, allowed: tuple[str, ...]) -> tuple[str, ...]:
    unknown = sorted(set(value) - set(allowed))
    if not value or unknown:
        raise ValueError(f"{name} must be a non-empty subset of {list(allowed)}, got {value}")
    return tuple(value)


def parse_dr(name: str, raw: dict) -> DomainRandomization:
    """Validate a preset dictionary against the hard limits."""
    expected = set(DomainRandomization.__dataclass_fields__) - {"name"}
    if set(raw) != expected:
        missing, extra = sorted(expected - set(raw)), sorted(set(raw) - expected)
        raise ValueError(f"DR preset {name!r}: missing {missing}, unexpected {extra}")
    presets = {str(k): float(v) for k, v in raw["course_presets"].items()}
    if not presets or min(presets.values()) < 0 or sum(presets.values()) <= 0:
        raise ValueError(f"course_presets needs positive weights, got {presets}")
    frames = int(raw["frames_per_course"])
    low, high = HARD_LIMITS["frames_per_course"]
    if not low <= frames <= high:
        raise ValueError(f"frames_per_course {frames} is outside [{low}, {high}]")
    ranges = {key: _check_range(key, raw[key]) for key in HARD_LIMITS if key != "frames_per_course"}
    return DomainRandomization(
        name=name,
        course_presets=presets,
        frames_per_course=frames,
        gate_patterns=_check_choices("gate_patterns", raw["gate_patterns"], GATE_PATTERNS),
        ground_patterns=_check_choices("ground_patterns", raw["ground_patterns"], GROUND_PATTERNS),
        palettes=_check_choices("palettes", raw["palettes"], tuple(PALETTES)),
        **ranges,
    )


def load_dr_preset(name: str) -> DomainRandomization:
    path: Path = CONFIG_DIR / "dr" / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"no DR preset named {name!r} at {path}")
    return parse_dr(name, yaml.safe_load(path.read_text()))


def _uniform(rng: np.random.Generator, bounds: Range) -> float:
    return float(rng.uniform(*bounds))


def _jitter(rng: np.random.Generator, color: tuple[float, ...]) -> tuple[float, float, float]:
    r, g, b = np.clip(np.array(color) + rng.uniform(-SKY_JITTER, SKY_JITTER, 3), 0.0, 1.0)
    return (float(r), float(g), float(b))


def sample_style(rng: np.random.Generator, dr: DomainRandomization) -> tuple[SceneStyle, str]:
    """A scene look and the palette it was drawn from. Lighting is sampled per frame."""
    palette_name = str(rng.choice(dr.palettes))
    palette = PALETTES[palette_name]
    hue = _uniform(rng, dr.gate_hue)
    gate_color = hsv(hue, _uniform(rng, dr.gate_saturation), _uniform(rng, dr.gate_value))
    pattern_color = hsv(hue + rng.uniform(0.3, 0.7), rng.uniform(0, 1), rng.uniform(0.05, 1))
    ground_hue = rng.uniform()
    ground_colors = (
        hsv(ground_hue, rng.uniform(0.05, 0.6), _uniform(rng, dr.ground_value)),
        hsv(
            ground_hue + rng.normal(0, 0.08), rng.uniform(0.05, 0.6), _uniform(rng, dr.ground_value)
        ),
    )
    style = SceneStyle(
        sky_horizon=_jitter(rng, palette["horizon"]),
        sky_zenith=_jitter(rng, palette["zenith"]),
        ground_pattern=str(rng.choice(dr.ground_patterns)),
        ground_colors=ground_colors,
        gate_color=gate_color,
        gate_pattern=str(rng.choice(dr.gate_patterns)),
        gate_pattern_color=pattern_color,
        lighting=sample_lighting(rng, dr, palette_name),
        texture_seed=int(rng.integers(2**31)),
    )
    return style, palette_name


def sample_lighting(rng: np.random.Generator, dr: DomainRandomization, palette: str) -> Lighting:
    """Light from anywhere on the sphere, colored by the palette and scaled by intensity."""
    direction = rng.normal(size=3)
    direction /= np.linalg.norm(direction)
    colors = PALETTES[palette]
    intensity = _uniform(rng, dr.light_intensity)
    r, g, b = np.array(colors["tint"]) * intensity
    return Lighting(
        direction=(float(direction[0]), float(direction[1]), float(direction[2])),
        color=(float(r), float(g), float(b)),
        ambient=float(colors["ambient"]),
        diffuse=float(colors["diffuse"]),
        specular=float(rng.uniform(0.0, 0.1)),
    )


def sample_distractors(
    rng: np.random.Generator, dr: DomainRandomization, course: Course
) -> list[Distractor]:
    """Poles, boxes and cones inside the flight volume, clear of every gate."""
    gate_xy = np.array([g.center[:2] for g in course.gates])
    count = int(rng.integers(dr.distractors[0], dr.distractors[1] + 1))
    distractors: list[Distractor] = []
    for _ in range(count):
        kind = str(rng.choice(DISTRACTOR_KINDS))
        width_range, height_range = DISTRACTOR_SIZES_M[kind]
        size = (_uniform(rng, width_range), _uniform(rng, height_range))
        color = hsv(rng.uniform(), rng.uniform(0, 1), rng.uniform(0.2, 1))
        for _ in range(MAX_PLACEMENT_TRIES):
            xy = rng.uniform(-VOLUME_HALF_XY_M, VOLUME_HALF_XY_M, 2)
            candidate = Distractor(
                kind, (float(xy[0]), float(xy[1])), size, rng.uniform(0, math.tau), color
            )
            clearance = np.linalg.norm(gate_xy - xy, axis=1).min()
            if clearance >= GATE_CLEARANCE_M + candidate.footprint_radius:
                distractors.append(candidate)
                break
    return distractors


def sample_effects(
    rng: np.random.Generator, dr: DomainRandomization, intrinsics: Intrinsics
) -> CameraEffects:
    """Camera effects for a scatter frame, with blur from a randomly drawn drone motion."""
    velocity = rng.normal(size=3)
    velocity *= _uniform(rng, dr.speed_m_s) / np.linalg.norm(velocity)
    angular = rng.normal(size=3)
    angular *= _uniform(rng, dr.angular_rate_rad_s) / np.linalg.norm(angular)
    max_blur = _uniform(rng, dr.motion_blur_px)
    return CameraEffects(
        exposure_ev=_uniform(rng, dr.exposure_ev),
        blur_px=motion_blur_vector(velocity, angular, intrinsics, max_blur),
        noise_sigma=_uniform(rng, dr.noise_sigma),
        jpeg_quality=int(rng.integers(dr.jpeg_quality[0], dr.jpeg_quality[1] + 1)),
    )
