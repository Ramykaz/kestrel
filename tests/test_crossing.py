import math

import numpy as np
import pytest

from kestrel.sim.course import GatePose
from kestrel.sim.crossing import PASS_HALF_SIZE_M, CrossingKind, detect_crossing
from kestrel.sim.oracle import gate_to_world

GATES = [
    GatePose(center=(0.0, 0.0, 1.5), yaw=0.0),
    GatePose(center=(4.0, -2.0, 2.3), yaw=math.radians(130)),
]


def _segment_through(gate: GatePose, y: float, z: float, forward: bool = True):
    """Segment from 5 cm before to 5 cm after the gate plane through local point (0, y, z)."""
    xs = (-0.05, 0.05) if forward else (0.05, -0.05)
    local = np.array([[xs[0], y, z], [xs[1], y, z]])
    start, end = gate_to_world(gate, local)
    return start, end


@pytest.mark.parametrize("gate", GATES)
def test_straight_through_center_passes(gate):
    crossing = detect_crossing(*_segment_through(gate, 0.0, 0.0), gate)
    assert crossing is not None
    assert crossing.kind is CrossingKind.PASSED
    assert np.allclose(crossing.point_gate, 0.0, atol=1e-9)


@pytest.mark.parametrize("gate", GATES)
def test_through_corner_fails(gate):
    crossing = detect_crossing(*_segment_through(gate, 0.29, 0.29), gate)
    assert crossing is not None
    assert crossing.kind is CrossingKind.OUTSIDE_OPENING


@pytest.mark.parametrize("gate", GATES)
def test_backwards_fails(gate):
    crossing = detect_crossing(*_segment_through(gate, 0.0, 0.0, forward=False), gate)
    assert crossing is not None
    assert crossing.kind is CrossingKind.BACKWARDS


@pytest.mark.parametrize("gate", GATES)
def test_grazing_the_edge(gate):
    inside = detect_crossing(*_segment_through(gate, PASS_HALF_SIZE_M - 0.005, 0.1), gate)
    grazing = detect_crossing(*_segment_through(gate, PASS_HALF_SIZE_M + 0.005, 0.1), gate)
    assert inside is not None and inside.kind is CrossingKind.PASSED
    assert grazing is not None and grazing.kind is CrossingKind.OUTSIDE_OPENING


@pytest.mark.parametrize("gate", GATES)
def test_far_from_gate_is_ignored(gate):
    assert detect_crossing(*_segment_through(gate, 3.0, 0.0), gate) is None


@pytest.mark.parametrize("gate", GATES)
def test_segment_not_reaching_plane_is_ignored(gate):
    start, _ = _segment_through(gate, 0.0, 0.0)
    before = gate_to_world(gate, np.array([[-0.2, 0.0, 0.0]]))[0]
    assert detect_crossing(before, start, gate) is None
