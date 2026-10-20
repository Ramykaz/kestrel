"""From a gate detection to the gate's position and direction in the world frame."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from kestrel.perception.detect import CLASS_GATE, Detection, corner_winding_matches_class
from kestrel.sim.camera import Intrinsics, camera_to_world

# The pilot knows the physical size of a gate (SPEC section 2).
GATE_INNER_SIZE_M = 0.60
_HALF = GATE_INNER_SIZE_M / 2
# Gate opening in the object frame solvePnP's IPPE_SQUARE expects: x to the right and y up as
# seen from the front, z out of the front face. Order: top left, top right, bottom right,
# bottom left, the same order as the detector's keypoints.
OBJECT_POINTS = np.array(
    [[-_HALF, _HALF, 0.0], [_HALF, _HALF, 0.0], [_HALF, -_HALF, 0.0], [-_HALF, -_HALF, 0.0]]
)


@dataclass(frozen=True)
class GateObservation:
    center: np.ndarray
    """World position of the center of the opening's front face."""
    forward: np.ndarray
    """Unit world vector of the direction of travel through the gate."""
    seen_from_front: bool
    reprojection_px: float
    distance_m: float


def estimate_gate(
    detection: Detection,
    intrinsics: Intrinsics,
    view_matrix: tuple[float, ...],
    max_reproj_px: float,
) -> GateObservation | None:
    """Solve the gate pose from its 4 corners. None when a corner is not visible, the corners
    wind against the class, or the fit's mean reprojection error exceeds max_reproj_px."""
    if np.any(detection.visibility < 1) or not corner_winding_matches_class(detection):
        return None
    image_points = detection.keypoints.astype(np.float64)
    camera_matrix = intrinsics.matrix()
    ok, rvec, tvec = cv2.solvePnP(
        OBJECT_POINTS, image_points, camera_matrix, None, flags=cv2.SOLVEPNP_IPPE_SQUARE
    )
    if not ok:
        return None
    projected, _ = cv2.projectPoints(OBJECT_POINTS, rvec, tvec, camera_matrix, None)
    error = float(np.linalg.norm(projected.reshape(-1, 2) - image_points, axis=1).mean())
    if error > max_reproj_px:
        return None
    rotation, _ = cv2.Rodrigues(rvec)
    center_cam = tvec.reshape(3)
    # Travel goes into the front face, against the object z axis.
    forward_cam = -rotation[:, 2]
    center_world, tip_world = camera_to_world(
        np.stack([center_cam, center_cam + forward_cam]), view_matrix
    )
    forward = tip_world - center_world
    return GateObservation(
        center=center_world,
        forward=forward / np.linalg.norm(forward),
        seen_from_front=detection.class_id == CLASS_GATE,
        reprojection_px=error,
        distance_m=float(np.linalg.norm(center_cam)),
    )
