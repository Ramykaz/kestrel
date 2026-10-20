import itertools
import math

import numpy as np
import pytest

from kestrel.sim.course import (
    CONE_HALF_ANGLE_DEG,
    MIN_GATE_DISTANCE_M,
    SPACING_M,
    VOLUME_HALF_XY_M,
    ensure_training_seed,
    eval_suite_seeds,
    generate_course,
    load_preset,
)

PRESETS = ["straight", "slalom", "tech", "night"]


@pytest.mark.parametrize("name", PRESETS)
def test_courses_respect_generator_constraints(name):
    preset = load_preset(name)
    for seed in range(40):
        course = generate_course(seed, preset)
        assert len(course.gates) == preset.gates
        centers = np.array([g.center for g in course.gates])
        assert np.all(np.abs(centers[:, :2]) <= VOLUME_HALF_XY_M)
        spacing = np.linalg.norm(np.diff(centers, axis=0), axis=1)
        assert np.all(spacing >= SPACING_M[0] - 1e-6)
        assert np.all(spacing <= SPACING_M[1] + 1e-6)
        for i in range(len(centers)):
            for j in range(i + 1, len(centers)):
                assert np.linalg.norm(centers[i] - centers[j]) >= MIN_GATE_DISTANCE_M
        in_cone = 0
        for prev, nxt in itertools.pairwise(course.gates):
            offset = np.array(nxt.center[:2]) - np.array(prev.center[:2])
            bearing = math.degrees(
                math.remainder(math.atan2(offset[1], offset[0]) - prev.yaw, math.tau)
            )
            in_cone += abs(bearing) <= CONE_HALF_ANGLE_DEG
        assert in_cone >= 0.5 * (len(course.gates) - 1)


@pytest.mark.parametrize("name", PRESETS)
def test_turn_angles_within_preset(name):
    preset = load_preset(name)
    for seed in range(20):
        gates = generate_course(seed, preset).gates
        for prev, nxt in itertools.pairwise(gates):
            turn = abs(math.degrees(math.remainder(nxt.yaw - prev.yaw, math.tau)))
            assert preset.turn_deg[0] - 1e-6 <= turn <= preset.turn_deg[1] + 1e-6


def test_straight_courses_are_level():
    preset = load_preset("straight")
    heights = {g.center[2] for seed in range(10) for g in generate_course(seed, preset).gates}
    assert heights == {1.5}


def test_same_seed_same_course():
    preset = load_preset("tech")
    assert generate_course(7, preset) == generate_course(7, preset)
    assert generate_course(7, preset) != generate_course(8, preset)


def test_eval_suite_has_20_tech_and_10_night_seeds():
    seeds = eval_suite_seeds()
    assert len(set(seeds["tech"])) == 20
    assert len(set(seeds["night"])) == 10


def test_eval_seeds_are_refused_for_training():
    eval_seed = eval_suite_seeds()["tech"][0]
    with pytest.raises(ValueError, match="held-out"):
        ensure_training_seed(eval_seed)
    ensure_training_seed(12345)


def test_eval_suite_courses_generate():
    seeds = eval_suite_seeds()
    for name in ("tech", "night"):
        preset = load_preset(name)
        for seed in seeds[name]:
            generate_course(seed, preset)
