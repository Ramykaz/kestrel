"""Projected keypoints must land on the rendered gate corners within 1 px (tested at 0.25 px).

The gate is rendered paper thin so the opening's silhouette is exactly the front face. The
quad through the projected keypoints predicts, for every pixel center, whether it is in the
opening. If the projection were off by e px, rendered pixels would disagree with that
prediction up to e px from the quad's edges. So the farthest disagreeing pixel bounds the
error, without fitting lines to a pixel staircase (which misjudges near-axis-aligned edges).
"""

import math

import numpy as np
import pybullet as p
import pytest

from kestrel.data.labels import VISIBLE, compute_gate_labels
from kestrel.sim.camera import OnboardCamera
from kestrel.sim.course import Course, GatePose
from kestrel.sim.oracle import Oracle
from kestrel.sim.world import World

THIN_DEPTH_M = 0.002
FRAMES = 50
# The spec asks for 1 px. A 0.5 px principal point error measures as about 0.5 px here, so
# 1 px would let it through; 0.25 px catches it.
MAX_ERROR_PX = 0.25
IMAGE_MARGIN_PX = 6


def _inside_convex_quad(points: np.ndarray, quad: np.ndarray) -> np.ndarray:
    signs = []
    for i in range(4):
        a, b = quad[i], quad[(i + 1) % 4]
        edge = b - a
        rel = points - a
        signs.append(edge[0] * rel[:, 1] - edge[1] * rel[:, 0])
    signs = np.stack(signs, axis=1)
    return np.all(signs > 0, axis=1) | np.all(signs < 0, axis=1)


def _distance_to_quad_boundary(points: np.ndarray, quad: np.ndarray) -> np.ndarray:
    distances = []
    for i in range(4):
        a, b = quad[i], quad[(i + 1) % 4]
        ab = b - a
        t = np.clip((points - a) @ ab / (ab @ ab), 0.0, 1.0)
        distances.append(np.linalg.norm(points - (a + t[:, None] * ab), axis=1))
    return np.min(distances, axis=0)


def _worst_disagreement_px(gate_mask: np.ndarray, expected: np.ndarray) -> float:
    """Largest distance from the projected opening's boundary to a pixel whose rendered
    class (opening or gate) disagrees with the projection.

    Pixels are taken up to 1.5 px outside the projected opening, which stays inside the
    gate bar (at least 2.6 px wide at the test distances).
    """
    rows, cols = np.indices(gate_mask.shape)
    centers = np.c_[cols.ravel(), rows.ravel()].astype(float)
    inside = _inside_convex_quad(centers, expected)
    distance = _distance_to_quad_boundary(centers, expected)
    candidates = inside | (distance <= 1.5)
    rendered_opening = ~gate_mask.ravel()
    disagree = candidates & (rendered_opening != inside)
    return float(distance[disagree].max()) if disagree.any() else 0.0


def _sample_camera(rng: np.random.Generator, gate: GatePose) -> tuple[np.ndarray, np.ndarray]:
    distance = rng.uniform(1.5, 4.0)
    bearing = gate.yaw + math.pi + math.radians(rng.uniform(-50, 50))
    center = np.array(gate.center)
    pos = center + distance * np.array([math.cos(bearing), math.sin(bearing), 0.0])
    pos[2] += rng.uniform(-0.6, 0.6)
    to_gate = center - pos
    yaw = math.atan2(to_gate[1], to_gate[0]) + math.radians(rng.normal(0, 8))
    elevation = math.atan2(to_gate[2], np.linalg.norm(to_gate[:2]))
    pitch = math.radians(15.0) - elevation + math.radians(rng.normal(0, 5))
    roll = math.radians(rng.uniform(-20, 20))
    quat = np.array(p.getQuaternionFromEuler([roll, pitch, yaw]))
    return pos, quat


@pytest.mark.parametrize("use_egl", [False, True], ids=["tiny", "egl"])
def test_projected_keypoints_match_rendered_corners(use_egl: bool):
    rng = np.random.default_rng(1234)
    gate = GatePose(center=(0.0, 0.0, 2.0), yaw=0.7)
    course = Course(seed=0, preset="test", gates=(gate,), start_pos=(0, 0, 0), start_yaw=0.0)
    world = World(use_egl=use_egl)
    try:
        if use_egl and not world.uses_egl:
            pytest.skip("EGL renderer is not available on this machine")
        body = world.add_gate(gate, depth=THIN_DEPTH_M)
        oracle = Oracle(course, [body], frame_depth=THIN_DEPTH_M)
        camera = OnboardCamera(world.client, world.renderer)
        intrinsics = camera.intrinsics
        errors = []
        while len(errors) < FRAMES:
            pos, quat = _sample_camera(rng, gate)
            view = camera.view_matrix(pos, quat)
            labels = compute_gate_labels(oracle, world.client, pos, view, intrinsics)
            if len(labels) != 1 or not np.all(labels[0].visibility == VISIBLE):
                continue
            expected = labels[0].keypoints
            margin_ok = (
                expected.min() >= IMAGE_MARGIN_PX
                and expected[:, 0].max() <= intrinsics.width - 1 - IMAGE_MARGIN_PX
                and expected[:, 1].max() <= intrinsics.height - 1 - IMAGE_MARGIN_PX
            )
            if not margin_ok:
                continue
            frame = camera.render(pos, quat, segmentation=True)
            assert frame.segmentation is not None
            errors.append(_worst_disagreement_px(frame.segmentation == body, expected))
        errors = np.array(errors)
        print(
            f"worst edge disagreement per frame: mean {errors.mean():.3f} px, "
            f"max {errors.max():.3f} px"
        )
        assert errors.max() <= MAX_ERROR_PX
    finally:
        world.close()
