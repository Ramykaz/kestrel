import math

import numpy as np
import pybullet as p
import pytest

from kestrel.perception.detect import (
    CLASS_GATE,
    CLASS_GATE_BACK,
    Detection,
    corner_winding_matches_class,
    decode_predictions,
    non_max_suppression,
)
from kestrel.perception.pose import GATE_INNER_SIZE_M, GateObservation, estimate_gate
from kestrel.perception.tracker import CONFIRM_HITS, GateTracker
from kestrel.pilot.params import PilotParams, load_pilot_preset
from kestrel.sim.camera import (
    camera_view_matrix,
    onboard_intrinsics,
    project_camera_points,
    world_to_camera,
)
from kestrel.sim.course import GatePose
from kestrel.sim.gates import FRAME_DEPTH_M, INNER_SIZE_M, inner_corners_local
from kestrel.sim.oracle import gate_to_world

INTRINSICS = onboard_intrinsics(p.ER_TINY_RENDERER)
GATE = GatePose((4.0, 1.0, 2.0), math.radians(20))


def _drone_quat(yaw: float, pitch: float = 0.0, roll: float = 0.0) -> np.ndarray:
    return np.array(p.getQuaternionFromEuler([roll, pitch, yaw]))


def _true_detection(pos, quat, class_id=CLASS_GATE, noise_px=0.0, seed=0):
    view = camera_view_matrix(pos, quat)
    corners = gate_to_world(GATE, inner_corners_local())
    uv = project_camera_points(world_to_camera(corners, view), INTRINSICS)
    uv += np.random.default_rng(seed).normal(0, noise_px, uv.shape)
    box = np.r_[uv.min(axis=0), uv.max(axis=0)]
    return Detection(class_id, 0.9, box, uv, np.full(4, 2)), view


def test_pilot_gate_size_matches_the_world():
    assert GATE_INNER_SIZE_M == INNER_SIZE_M


@pytest.mark.parametrize(
    ("offset", "roll_deg"), [((-4.0, 0.0, 0.0), 0.0), ((-2.5, 1.2, -0.4), 20.0)]
)
def test_pnp_recovers_gate_pose_from_exact_corners(offset, roll_deg):
    forward = GATE.forward
    left = np.array([-forward[1], forward[0], 0.0])
    pos = np.array(GATE.center) + offset[0] * forward + offset[1] * left
    pos[2] += offset[2]
    to_gate = np.array(GATE.center) - pos
    quat = _drone_quat(math.atan2(to_gate[1], to_gate[0]), roll=math.radians(roll_deg))
    detection, view = _true_detection(pos, quat)
    observation = estimate_gate(detection, INTRINSICS, view, max_reproj_px=1.0)
    assert observation is not None
    front_face_center = np.array(GATE.center) - forward * FRAME_DEPTH_M / 2
    assert np.linalg.norm(observation.center - front_face_center) < 1e-3
    assert math.degrees(math.acos(np.clip(observation.forward @ forward, -1, 1))) < 0.1
    assert observation.seen_from_front and observation.reprojection_px < 0.01


def test_pnp_from_behind_gives_the_same_travel_direction():
    pos = np.array(GATE.center) + 3.0 * GATE.forward
    quat = _drone_quat(GATE.yaw + math.pi)
    detection, view = _true_detection(pos, quat, class_id=CLASS_GATE_BACK)
    observation = estimate_gate(detection, INTRINSICS, view, max_reproj_px=1.0)
    assert observation is not None and not observation.seen_from_front
    assert observation.forward @ GATE.forward > 0.999


def test_pnp_with_half_pixel_noise_stays_within_centimetres():
    pos = np.array(GATE.center) - 4.0 * GATE.forward
    errors = []
    for seed in range(20):
        detection, view = _true_detection(pos, _drone_quat(GATE.yaw), noise_px=0.5, seed=seed)
        observation = estimate_gate(detection, INTRINSICS, view, max_reproj_px=3.0)
        assert observation is not None
        front_face_center = np.array(GATE.center) - GATE.forward * FRAME_DEPTH_M / 2
        errors.append(np.linalg.norm(observation.center - front_face_center))
    assert np.median(errors) < 0.10


def test_pnp_rejects_hidden_corners_wrong_winding_and_bad_fits():
    pos = np.array(GATE.center) - 4.0 * GATE.forward
    detection, view = _true_detection(pos, _drone_quat(GATE.yaw))
    hidden = Detection(0, 0.9, detection.box_xyxy, detection.keypoints, np.array([2, 2, 0, 2]))
    assert estimate_gate(hidden, INTRINSICS, view, 3.0) is None
    mislabeled = Detection(
        CLASS_GATE_BACK, 0.9, detection.box_xyxy, detection.keypoints, detection.visibility
    )
    assert not corner_winding_matches_class(mislabeled)
    assert estimate_gate(mislabeled, INTRINSICS, view, 3.0) is None
    skewed = detection.keypoints.copy()
    skewed[0] += [6.0, 6.0]
    bad = Detection(0, 0.9, detection.box_xyxy, skewed, detection.visibility)
    assert estimate_gate(bad, INTRINSICS, view, 1.0) is None


def test_decode_maps_letterbox_back_and_suppresses_duplicates():
    # Letterbox of a 320x240 frame into 320x256: no scaling, 8 px of padding on top.
    scale, offset = 1.0, np.array([0.0, 8.0])
    keypoints = [100, 58, 0.9, 140, 58, 0.9, 140, 98, 0.9, 100, 98, 0.2]
    best = [120, 78, 60, 60, 0.95, 0.02, *keypoints]
    duplicate = [122, 79, 60, 60, 0.80, 0.05, *keypoints]
    weak = [250, 150, 20, 20, 0.10, 0.05, *keypoints]
    back = [40, 200, 30, 30, 0.10, 0.70, *keypoints]
    detections = decode_predictions(np.array([best, duplicate, weak, back]), scale, offset, 0.4)
    assert [d.class_id for d in detections] == [CLASS_GATE, CLASS_GATE_BACK]
    first = detections[0]
    assert first.confidence == pytest.approx(0.95)
    np.testing.assert_allclose(first.box_xyxy, [89.5, 39.5, 149.5, 99.5])
    np.testing.assert_allclose(first.keypoints[0], [99.5, 49.5])
    assert first.visibility.tolist() == [2, 2, 2, 0]


def test_nms_keeps_separate_boxes():
    boxes = np.array([[0, 0, 10, 10], [1, 1, 11, 11], [50, 50, 60, 60]], dtype=float)
    assert non_max_suppression(boxes, np.array([0.9, 0.8, 0.7]), 0.5) == [0, 2]


def _observation(center, forward=(1.0, 0.0, 0.0)):
    return GateObservation(np.array(center, float), np.array(forward, float), True, 0.5, 3.0)


def test_tracker_averages_matches_and_forgets_unseen_gates():
    tracker = GateTracker(ema_alpha=0.5, forget_s=1.0)
    tracker.update([_observation([0, 0, 1]), _observation([5, 0, 1])], 0.0, keep_id=None)
    tracker.update([_observation([0.4, 0, 1])], 0.1, keep_id=None)
    near = tracker.hypotheses[0]
    np.testing.assert_allclose(near.center, [0.2, 0, 1])
    assert near.hits == 2 and not near.confirmed
    for i in range(CONFIRM_HITS):
        tracker.update([_observation([0.2, 0, 1])], 0.2 + 0.1 * i, keep_id=None)
    assert near.confirmed
    far_id = tracker.hypotheses[1].id
    tracker.update([], 1.3, keep_id=far_id)
    assert [h.id for h in tracker.hypotheses] == [near.id, far_id]
    tracker.update([], 2.0, keep_id=None)
    assert tracker.hypotheses == []


def test_pilot_params_enforce_bounds():
    params = load_pilot_preset("base")
    assert params.speed_cap <= 4.0
    raw = params.to_dict() | {"speed_cap": 5.0}
    with pytest.raises(ValueError, match="speed_cap"):
        PilotParams(**raw)
