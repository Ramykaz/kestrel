"""Gate pass detection between two physics steps."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

from kestrel.sim.course import GatePose
from kestrel.sim.gates import INNER_SIZE_M, OUTER_SIZE_M
from kestrel.sim.oracle import world_to_gate

DRONE_RADIUS_M = 0.06
PASS_HALF_SIZE_M = INNER_SIZE_M / 2 - DRONE_RADIUS_M
# Plane crossings farther than this from the gate center are unrelated to the gate.
NEAR_GATE_HALF_SIZE_M = OUTER_SIZE_M / 2 + DRONE_RADIUS_M


class CrossingKind(Enum):
    PASSED = "passed"
    BACKWARDS = "backwards"
    OUTSIDE_OPENING = "outside_opening"


@dataclass(frozen=True)
class Crossing:
    kind: CrossingKind
    point_gate: np.ndarray
    """Crossing point in gate coordinates; x is 0 by construction."""


def detect_crossing(prev_pos: np.ndarray, cur_pos: np.ndarray, gate: GatePose) -> Crossing | None:
    """Classify the segment prev_pos -> cur_pos against the gate's mid-depth plane.

    Returns None when the segment does not cross the plane near the gate.
    """
    prev_local, cur_local = world_to_gate(gate, np.stack([prev_pos, cur_pos]))
    x0, x1 = prev_local[0], cur_local[0]
    forward = x0 < 0.0 <= x1
    backward = x0 > 0.0 >= x1
    if not (forward or backward):
        return None
    t = x0 / (x0 - x1)
    point = prev_local + t * (cur_local - prev_local)
    point[0] = 0.0
    lateral, vertical = abs(point[1]), abs(point[2])
    if max(lateral, vertical) > NEAR_GATE_HALF_SIZE_M:
        return None
    if max(lateral, vertical) > PASS_HALF_SIZE_M:
        return Crossing(CrossingKind.OUTSIDE_OPENING, point)
    if backward:
        return Crossing(CrossingKind.BACKWARDS, point)
    return Crossing(CrossingKind.PASSED, point)
