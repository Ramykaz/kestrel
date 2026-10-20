"""Ground-truth crash causes from a flight log and the course (SPEC 9.3).

decoy_followed needs decoys, which courses do not have before Phase 4, so it never fires yet.
A crash no rule explains is labeled unclassified rather than forced into a cause.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pybullet as p

from kestrel.pilot.flight_log import FlightEnd, FlightLog, PilotState
from kestrel.sim.camera import (
    NEAR_M,
    camera_view_matrix,
    onboard_intrinsics,
    project_camera_points,
    world_to_camera,
)
from kestrel.sim.course import Course
from kestrel.sim.gates import FRAME_DEPTH_M

WINDOW_FRAMES = 30
MISSED_MIN_IN_VIEW = 10
MISSED_MAX_DETECTED = 3
BAD_POSE_M = 0.5
WRONG_SIDE_MATCH_M = 1.0
CAUSES = ("decoy_followed", "wrong_side", "missed_gate", "bad_pose", "lost_track", "late_turn")
UNCLASSIFIED = "unclassified"


@dataclass(frozen=True)
class CrashCause:
    cause: str
    gate_index: int | None
    """The gate being approached when the drone crashed."""
    frames_in_view: int
    frames_detected: int
    pose_error_m: float | None
    """Median distance from the believed target to the nearest true gate, last second."""

    def to_dict(self) -> dict:
        return {
            "cause": self.cause,
            "gate_index": self.gate_index,
            "frames_in_view": self.frames_in_view,
            "frames_detected": self.frames_detected,
            "pose_error_m": None if self.pose_error_m is None else round(self.pose_error_m, 3),
        }


def _front_centers(course: Course) -> np.ndarray:
    return np.array([np.array(g.center) - g.forward * FRAME_DEPTH_M / 2 for g in course.gates])


def crash_cause(log: FlightLog, course: Course) -> CrashCause:
    """Label a crashed flight. Raises for flights that did not crash."""
    if log.end is not FlightEnd.CRASH:
        raise ValueError(f"flight on course {log.course_seed} ended {log.end}, not in a crash")
    intrinsics = onboard_intrinsics(p.ER_TINY_RENDERER)
    gate_index = log.gates_passed if log.gates_passed < len(course.gates) else None
    window = range(max(0, len(log.times) - WINDOW_FRAMES), len(log.times))
    centers = _front_centers(course)

    in_view = detected = 0
    if gate_index is not None:
        for i in window:
            view = camera_view_matrix(log.positions[i], log.quaternions[i])
            center_cam = world_to_camera(centers[gate_index][None], view)
            if center_cam[0, 2] < NEAR_M:
                continue
            u, v = project_camera_points(center_cam, intrinsics)[0]
            if not (-0.5 <= u <= intrinsics.width - 0.5 and -0.5 <= v <= intrinsics.height - 0.5):
                continue
            in_view += 1
            boxes = [d.box_xyxy for d in log.detections[i]]
            if any(b[0] <= u <= b[2] and b[1] <= v <= b[3] for b in boxes):
                detected += 1

    errors = []
    wrong_side = False
    for i in window:
        believed = log.believed_gates[i]
        if np.any(np.isnan(believed)):
            continue
        distances = np.linalg.norm(centers - believed, axis=1)
        nearest = int(np.argmin(distances))
        errors.append(float(distances[nearest]))
        gate = course.gates[nearest]
        behind = float(np.dot(log.positions[i] - centers[nearest], gate.forward)) > 0.0
        if (
            log.states[i] is PilotState.APPROACH
            and distances[nearest] <= WRONG_SIDE_MATCH_M
            and behind
        ):
            wrong_side = True
    pose_error = float(np.median(errors)) if errors else None
    no_detections = all(not log.detections[i] for i in window) and len(window) == WINDOW_FRAMES

    if wrong_side:
        cause = "wrong_side"
    elif in_view >= MISSED_MIN_IN_VIEW and detected < MISSED_MAX_DETECTED:
        cause = "missed_gate"
    elif pose_error is not None and pose_error > BAD_POSE_M:
        cause = "bad_pose"
    elif no_detections:
        cause = "lost_track"
    elif log.crash_body_kind in ("gate_frame", "gate_post"):
        cause = "late_turn"
    else:
        cause = UNCLASSIFIED
    return CrashCause(cause, gate_index, in_view, detected, pose_error)
