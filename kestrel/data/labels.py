"""Gate labels from ground truth, and the YOLO pose label writer.

Pixel coordinates follow the OpenCV convention: the top left pixel center is (0, 0), so the
image spans [-0.5, width - 0.5] horizontally.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pybullet as p

from kestrel.sim.camera import NEAR_M, Intrinsics, project_camera_points, world_to_camera
from kestrel.sim.oracle import Oracle

CLASS_GATE = 0
CLASS_GATE_BACK = 1
CLASS_NAMES = {CLASS_GATE: "gate", CLASS_GATE_BACK: "gate_back"}

VISIBLE = 2
OCCLUDED = 1
OUTSIDE_FRAME = 0

MIN_BOX_AREA_PX = 36.0
# A ray to a keypoint counts as unobstructed if it hits something within this distance of it.
OCCLUSION_TOLERANCE_M = 0.03
# Rays start this far from the camera so they clear the drone's own 0.06 m collision cylinder.
RAY_START_OFFSET_M = 0.10

_CUBOID_EDGES = [
    (a, b)
    for a in range(8)
    for b in range(a + 1, 8)
    if bin(a ^ b).count("1") == 1  # corners differ in exactly one of x, y, z
]


@dataclass(frozen=True)
class GateLabel:
    gate_index: int
    class_id: int
    box_xyxy: np.ndarray
    keypoints: np.ndarray
    visibility: np.ndarray


def _clip_segment_to_near_plane(
    a: np.ndarray, b: np.ndarray
) -> tuple[np.ndarray, np.ndarray] | None:
    if a[2] < NEAR_M and b[2] < NEAR_M:
        return None
    if a[2] >= NEAR_M and b[2] >= NEAR_M:
        return a, b
    t = (NEAR_M - a[2]) / (b[2] - a[2])
    crossing = a + t * (b - a)
    return (crossing, b) if a[2] < NEAR_M else (a, crossing)


def _projected_box(outer_cam: np.ndarray, intrinsics: Intrinsics) -> np.ndarray | None:
    points = []
    for a, b in _CUBOID_EDGES:
        clipped = _clip_segment_to_near_plane(outer_cam[a], outer_cam[b])
        if clipped is not None:
            points.extend(clipped)
    if not points:
        return None
    uv = project_camera_points(np.array(points), intrinsics)
    low = np.array([-0.5, -0.5])
    high = np.array([intrinsics.width - 0.5, intrinsics.height - 0.5])
    top_left = np.clip(uv.min(axis=0), low, high)
    bottom_right = np.clip(uv.max(axis=0), low, high)
    return np.r_[top_left, bottom_right]


def _keypoint_visibility(
    client: int,
    camera_pos: np.ndarray,
    corners_world: np.ndarray,
    corners_cam: np.ndarray,
    uv: np.ndarray,
    intrinsics: Intrinsics,
) -> np.ndarray:
    in_front = corners_cam[:, 2] >= NEAR_M
    in_image = (
        in_front
        & (uv[:, 0] >= -0.5)
        & (uv[:, 0] <= intrinsics.width - 0.5)
        & (uv[:, 1] >= -0.5)
        & (uv[:, 1] <= intrinsics.height - 0.5)
    )
    visibility = np.where(in_image, VISIBLE, OUTSIDE_FRAME)
    offsets = corners_world - camera_pos
    distances = np.linalg.norm(offsets, axis=1)
    to_test = np.flatnonzero(in_image & (distances > RAY_START_OFFSET_M))
    if len(to_test) == 0:
        return visibility
    starts = camera_pos + offsets[to_test] / distances[to_test, None] * RAY_START_OFFSET_M
    hits = p.rayTestBatch(starts.tolist(), corners_world[to_test].tolist(), physicsClientId=client)
    for index, hit in zip(to_test, hits, strict=True):
        if hit[0] < 0:
            continue
        free_length = hit[2] * (distances[index] - RAY_START_OFFSET_M)
        if free_length < distances[index] - RAY_START_OFFSET_M - OCCLUSION_TOLERANCE_M:
            visibility[index] = OCCLUDED
    return visibility


def compute_gate_labels(
    oracle: Oracle,
    client: int,
    camera_pos: np.ndarray,
    view_matrix: tuple[float, ...],
    intrinsics: Intrinsics,
) -> list[GateLabel]:
    """Labels for every gate in view. Gates behind the camera or under 36 px^2 are skipped."""
    labels = []
    for index in range(oracle.gate_count):
        outer_cam = world_to_camera(oracle.outer_box_corners_world(index), view_matrix)
        box = _projected_box(outer_cam, intrinsics)
        if box is None:
            continue
        if (box[2] - box[0]) * (box[3] - box[1]) < MIN_BOX_AREA_PX:
            continue
        corners_world = oracle.inner_corners_world(index)
        corners_cam = world_to_camera(corners_world, view_matrix)
        safe_cam = corners_cam.copy()
        safe_cam[:, 2] = np.maximum(safe_cam[:, 2], NEAR_M)
        uv = project_camera_points(safe_cam, intrinsics)
        visibility = _keypoint_visibility(
            client, camera_pos, corners_world, corners_cam, uv, intrinsics
        )
        uv[visibility == OUTSIDE_FRAME] = 0.0
        front = oracle.is_viewed_from_front(index, camera_pos)
        labels.append(
            GateLabel(
                gate_index=index,
                class_id=CLASS_GATE if front else CLASS_GATE_BACK,
                box_xyxy=box,
                keypoints=uv,
                visibility=visibility,
            )
        )
    return labels


def yolo_pose_lines(labels: list[GateLabel], width: int, height: int) -> list[str]:
    """One line per gate: class, box center and size, then x y v per keypoint, normalized."""
    lines = []
    size = np.array([width, height], dtype=float)
    for label in labels:
        top_left = (label.box_xyxy[:2] + 0.5) / size
        bottom_right = (label.box_xyxy[2:] + 0.5) / size
        center = (top_left + bottom_right) / 2
        extent = bottom_right - top_left
        fields = [str(label.class_id), *(f"{v:.6f}" for v in (*center, *extent))]
        for (u, v), vis in zip(label.keypoints, label.visibility, strict=True):
            if vis == OUTSIDE_FRAME:
                fields += ["0", "0", "0"]
            else:
                fields += [f"{(u + 0.5) / width:.6f}", f"{(v + 0.5) / height:.6f}", str(int(vis))]
        lines.append(" ".join(fields))
    return lines


def write_yolo_label(path: Path, labels: list[GateLabel], width: int, height: int) -> None:
    lines = yolo_pose_lines(labels, width, height)
    path.write_text("\n".join(lines) + ("\n" if lines else ""))
