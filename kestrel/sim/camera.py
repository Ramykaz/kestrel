"""Onboard forward camera: intrinsics, view and projection matrices, rendering."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pybullet as p

WIDTH = 320
HEIGHT = 240
HFOV_DEG = 90.0
PITCH_UP_DEG = 15.0
NEAR_M = 0.05
FAR_M = 60.0


@dataclass(frozen=True)
class Intrinsics:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float

    def matrix(self) -> np.ndarray:
        return np.array([[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]])

    def to_dict(self) -> dict[str, float]:
        return {
            "width": self.width,
            "height": self.height,
            "fx": self.fx,
            "fy": self.fy,
            "cx": self.cx,
            "cy": self.cy,
        }


def onboard_intrinsics() -> Intrinsics:
    focal_px = (WIDTH / 2) / math.tan(math.radians(HFOV_DEG) / 2)
    return Intrinsics(WIDTH, HEIGHT, focal_px, focal_px, WIDTH / 2, HEIGHT / 2)


@dataclass(frozen=True)
class CameraFrame:
    rgb: np.ndarray
    view_matrix: tuple[float, ...]
    projection_matrix: tuple[float, ...]


class OnboardCamera:
    """Camera rigidly mounted on the drone, looking along body +x, pitched up."""

    def __init__(self, client: int, renderer: int) -> None:
        self.client = client
        self.renderer = renderer
        self.intrinsics = onboard_intrinsics()
        vfov_deg = math.degrees(2 * math.atan((HEIGHT / 2) / self.intrinsics.fy))
        self.projection_matrix = p.computeProjectionMatrixFOV(
            fov=vfov_deg, aspect=WIDTH / HEIGHT, nearVal=NEAR_M, farVal=FAR_M
        )
        pitch = math.radians(PITCH_UP_DEG)
        self._forward_body = np.array([math.cos(pitch), 0.0, math.sin(pitch)])
        self._up_body = np.array([-math.sin(pitch), 0.0, math.cos(pitch)])

    def view_matrix(self, drone_pos: np.ndarray, drone_quat: np.ndarray) -> tuple[float, ...]:
        body_to_world = np.array(p.getMatrixFromQuaternion(drone_quat.tolist())).reshape(3, 3)
        forward = body_to_world @ self._forward_body
        up = body_to_world @ self._up_body
        return p.computeViewMatrix(
            cameraEyePosition=drone_pos.tolist(),
            cameraTargetPosition=(drone_pos + forward).tolist(),
            cameraUpVector=up.tolist(),
        )

    def render(self, drone_pos: np.ndarray, drone_quat: np.ndarray) -> CameraFrame:
        view = self.view_matrix(drone_pos, drone_quat)
        _, _, rgba, _, _ = p.getCameraImage(
            WIDTH,
            HEIGHT,
            viewMatrix=view,
            projectionMatrix=self.projection_matrix,
            renderer=self.renderer,
            flags=p.ER_NO_SEGMENTATION_MASK,
            physicsClientId=self.client,
        )
        rgb = np.asarray(rgba, dtype=np.uint8).reshape(HEIGHT, WIDTH, 4)[:, :, :3]
        return CameraFrame(rgb, view, self.projection_matrix)
