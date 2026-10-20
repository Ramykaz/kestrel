import numpy as np
import pybullet as p
import pytest
import yaml

from kestrel.data.postfx import CameraEffects, apply_effects, degrade, motion_blur_vector
from kestrel.data.randomize import (
    GATE_CLEARANCE_M,
    load_dr_preset,
    parse_dr,
    sample_distractors,
    sample_effects,
    sample_style,
)
from kestrel.paths import CONFIG_DIR
from kestrel.sim.camera import onboard_intrinsics
from kestrel.sim.course import generate_course, load_preset
from kestrel.sim.scene import GROUND_PATTERNS, ground_texture


def _base_raw() -> dict:
    return yaml.safe_load((CONFIG_DIR / "dr" / "base.yaml").read_text())


def test_base_preset_loads():
    dr = load_dr_preset("base")
    assert dr.noise_sigma == (0, 8)
    assert set(dr.palettes) == {"day", "dusk", "night"}


@pytest.mark.parametrize(
    ("key", "value"),
    [("noise_sigma", [0, 20]), ("speed_m_s", [0.0, 5.0]), ("jpeg_quality", [90, 40])],
)
def test_preset_outside_hard_limits_is_rejected(key, value):
    raw = _base_raw()
    raw[key] = value
    with pytest.raises(ValueError, match=key):
        parse_dr("bad", raw)


def test_preset_with_unknown_pattern_or_missing_key_is_rejected():
    raw = _base_raw()
    raw["gate_patterns"] = ["solid", "plaid"]
    with pytest.raises(ValueError, match="gate_patterns"):
        parse_dr("bad", raw)
    raw = _base_raw()
    del raw["exposure_ev"]
    with pytest.raises(ValueError, match="missing"):
        parse_dr("bad", raw)


def test_distractors_keep_clear_of_gates():
    dr = load_dr_preset("base")
    course = generate_course(123, load_preset("tech"))
    rng = np.random.default_rng(0)
    gates = np.array([g.center[:2] for g in course.gates])
    for _ in range(20):
        for d in sample_distractors(rng, dr, course):
            gap = np.linalg.norm(gates - np.array(d.position), axis=1).min()
            assert gap >= GATE_CLEARANCE_M + d.footprint_radius


def test_sampled_effects_stay_in_preset_ranges():
    dr = load_dr_preset("base")
    intrinsics = onboard_intrinsics(p.ER_TINY_RENDERER)
    rng = np.random.default_rng(1)
    for _ in range(200):
        fx = sample_effects(rng, dr, intrinsics)
        assert np.hypot(*fx.blur_px) <= dr.motion_blur_px[1] + 1e-9
        assert dr.jpeg_quality[0] <= fx.jpeg_quality <= dr.jpeg_quality[1]
        assert dr.exposure_ev[0] <= fx.exposure_ev <= dr.exposure_ev[1]
    style, palette = sample_style(rng, dr)
    assert palette in dr.palettes and style.ground_pattern in dr.ground_patterns


def test_yaw_rate_blurs_horizontally_and_climb_vertically():
    intrinsics = onboard_intrinsics(p.ER_TINY_RENDERER)
    # Yawing left is a rotation about camera -y; the scene streaks to the right.
    du, dv = motion_blur_vector(np.zeros(3), np.array([0.0, -2.0, 0.0]), intrinsics, 9.0)
    assert du == pytest.approx(160 * 2.0 / 60) and dv == 0.0
    du, dv = motion_blur_vector(np.array([0.0, -3.0, 0.0]), np.zeros(3), intrinsics, 9.0)
    assert du == 0.0 and dv == pytest.approx(160 * 3.0 / 3.0 / 60)
    du, dv = motion_blur_vector(np.zeros(3), np.array([0.0, -50.0, 0.0]), intrinsics, 9.0)
    assert np.hypot(du, dv) == pytest.approx(9.0)


def test_effects_keep_mean_brightness_and_change_pixels():
    rng = np.random.default_rng(2)
    image = np.full((240, 320, 3), 120, dtype=np.uint8)
    image[:, 150:170] = 250
    neutral = CameraEffects(0.0, (0.0, 0.0), 0.0, 95)
    assert np.array_equal(apply_effects(image, neutral, rng), image)
    blurred = apply_effects(image, CameraEffects(0.0, (8.0, 0.0), 0.0, 95), rng)
    assert abs(float(blurred.mean()) - float(image.mean())) < 1.0
    assert blurred[:, 147].mean() > 125
    darker = degrade(image, CameraEffects(-1.0, (0.0, 0.0), 4.0, 50), rng)
    assert darker.shape == image.shape and darker.mean() < image.mean() * 0.8


@pytest.mark.parametrize("pattern", GROUND_PATTERNS)
def test_ground_textures_tile_without_seams(pattern):
    texture = ground_texture(pattern, ((0.9, 0.9, 0.9), (0.1, 0.1, 0.1)), 7).astype(float)
    # Patterns with hard edges put one on the tile border, so the wrap-around step is compared
    # with the largest step inside the tile, not the average one.
    for axis in (0, 1):
        steps = np.abs(np.diff(texture, axis=axis)).mean(axis=(1 - axis, 2))
        wrap = np.abs(np.take(texture, 0, axis) - np.take(texture, -1, axis)).mean()
        assert wrap <= steps.max() * 1.05 + 1.0, f"{pattern} axis {axis}: seam {wrap:.1f}"
