"""Ground truth about the course. The only module the rest of the code may ask where gates are.

Used for labels, scoring and crash causes. Nothing under kestrel/pilot or kestrel/perception
may import it; tests/test_world_rules.py enforces that.
"""

from __future__ import annotations

import math

import numpy as np

from kestrel.sim.course import Course, GatePose
from kestrel.sim.gates import FRAME_DEPTH_M, inner_corners_local, outer_box_corners_local


def gate_to_world(gate: GatePose, points_local: np.ndarray) -> np.ndarray:
    c, s = math.cos(gate.yaw), math.sin(gate.yaw)
    rotation = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    return points_local @ rotation.T + np.array(gate.center)


def world_to_gate(gate: GatePose, points_world: np.ndarray) -> np.ndarray:
    c, s = math.cos(gate.yaw), math.sin(gate.yaw)
    rotation = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    return (points_world - np.array(gate.center)) @ rotation


class Oracle:
    """Gate poses and geometry for one loaded course."""

    def __init__(
        self, course: Course, gate_body_ids: list[int], frame_depth: float = FRAME_DEPTH_M
    ) -> None:
        if len(gate_body_ids) != len(course.gates):
            raise ValueError(
                f"{len(gate_body_ids)} gate bodies for a course with {len(course.gates)} gates"
            )
        self.course = course
        self.gate_body_ids = list(gate_body_ids)
        self.frame_depth = frame_depth
        self._gate_index_by_body = {body: i for i, body in enumerate(gate_body_ids)}

    @property
    def gate_count(self) -> int:
        return len(self.course.gates)

    def gate_pose(self, index: int) -> GatePose:
        return self.course.gates[index]

    def gate_index_of_body(self, body_id: int) -> int | None:
        return self._gate_index_by_body.get(body_id)

    def inner_corners_world(self, index: int) -> np.ndarray:
        """4x3 front-face inner corners: top left, top right, bottom right, bottom left."""
        return gate_to_world(self.gate_pose(index), inner_corners_local(self.frame_depth))

    def outer_box_corners_world(self, index: int) -> np.ndarray:
        return gate_to_world(self.gate_pose(index), outer_box_corners_local(self.frame_depth))

    def is_viewed_from_front(self, index: int, viewer_pos: np.ndarray) -> bool:
        gate = self.gate_pose(index)
        return float(np.dot(viewer_pos - np.array(gate.center), gate.forward)) < 0.0
