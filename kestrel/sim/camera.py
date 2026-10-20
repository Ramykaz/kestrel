"""Onboard forward camera: intrinsics, view and projection matrices, rendering."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pybullet as p

from kestrel.sim.scene import SceneStyle

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


def onboard_intrinsics(renderer: int) -> Intrinsics:
    """Pinhole intrinsics of the rendered image in the OpenCV convention: the top left pixel
    center is (0, 0). The principal point depends on the renderer."""
    focal_px = (WIDTH / 2) / math.tan(math.radians(HFOV_DEG) / 2)
    if renderer == p.ER_TINY_RENDERER:
        # TinyRenderer samples pixel i at viewport coordinate i instead of i + 0.5, then flips
        # rows, which shifts its image by (+0.5, -0.5) px against an ideal pinhole camera.
        # Measured in tests/test_label_projection.py; EGL shows no offset.
        return Intrinsics(WIDTH, HEIGHT, focal_px, focal_px, WIDTH / 2, HEIGHT / 2 - 1)
    return Intrinsics(WIDTH, HEIGHT, focal_px, focal_px, WIDTH / 2 - 0.5, HEIGHT / 2 - 0.5)


def world_to_camera(points_world: np.ndarray, view_matrix: tuple[float, ...]) -> np.ndarray:
    """Transform Nx3 world points to OpenCV camera coordinates (x right, y down, z forward)."""
    view = np.array(view_matrix).reshape(4, 4, order="F")
    homogeneous = np.c_[points_world, np.ones(len(points_world))]
    gl = (view @ homogeneous.T).T[:, :3]
    return gl * np.array([1.0, -1.0, -1.0])


def project_camera_points(points_cam: np.ndarray, intrinsics: Intrinsics) -> np.ndarray:
    """Project Nx3 camera points with positive depth to Nx2 pixel coordinates."""
    z = points_cam[:, 2]
    u = intrinsics.fx * points_cam[:, 0] / z + intrinsics.cx
    v = intrinsics.fy * points_cam[:, 1] / z + intrinsics.cy
    return np.c_[u, v]


@dataclass(frozen=True)
class CameraFrame:
    rgb: np.ndarray
    view_matrix: tuple[float, ...]
    projection_matrix: tuple[float, ...]
    segmentation: np.ndarray | None = None


class OnboardCamera:
    """Camera rigidly mounted on the drone, looking along body +x, pitched up."""

    def __init__(self, client: int, renderer: int) -> None:
        self.client = client
        self.renderer = renderer
        self.intrinsics = onboard_intrinsics(renderer)
        vfov_deg = math.degrees(2 * math.atan((HEIGHT / 2) / self.intrinsics.fy))
        self.projection_matrix = p.computeProjectionMatrixFOV(
            fov=vfov_deg, aspect=WIDTH / HEIGHT, nearVal=NEAR_M, farVal=FAR_M
        )
        pitch = math.radians(PITCH_UP_DEG)
        self._forward_body = np.array([math.cos(pitch), 0.0, math.sin(pitch)])
        self._up_body = np.array([-math.sin(pitch), 0.0, math.cos(pitch)])
        k = self.intrinsics
        v, u = np.mgrid[0:HEIGHT, 0:WIDTH].astype(float)
        rays = np.stack([(u - k.cx) / k.fx, (v - k.cy) / k.fy, np.ones_like(u)], axis=-1)
        self._pixel_rays = rays / np.linalg.norm(rays, axis=-1, keepdims=True)

    def view_matrix(self, drone_pos: np.ndarray, drone_quat: np.ndarray) -> tuple[float, ...]:
        body_to_world = np.array(p.getMatrixFromQuaternion(drone_quat.tolist())).reshape(3, 3)
        forward = body_to_world @ self._forward_body
        up = body_to_world @ self._up_body
        return p.computeViewMatrix(
            cameraEyePosition=drone_pos.tolist(),
            cameraTargetPosition=(drone_pos + forward).tolist(),
            cameraUpVector=up.tolist(),
        )

    def render(
        self,
        drone_pos: np.ndarray,
        drone_quat: np.ndarray,
        segmentation: bool = False,
        style: SceneStyle | None = None,
    ) -> CameraFrame:
        """Render the onboard view. With segmentation or a style, also return per-pixel body
        ids (-1 for background). A style sets the lighting and paints the sky."""
        view = self.view_matrix(drone_pos, drone_quat)
        with_seg = segmentation or style is not None
        _, _, rgba, _, seg = p.getCameraImage(
            WIDTH,
            HEIGHT,
            viewMatrix=view,
            projectionMatrix=self.projection_matrix,
            renderer=self.renderer,
            flags=0 if with_seg else p.ER_NO_SEGMENTATION_MASK,
            physicsClientId=self.client,
            **(style.lighting.render_kwargs() if style is not None else {}),
        )
        rgb = np.asarray(rgba, dtype=np.uint8).reshape(HEIGHT, WIDTH, 4)[:, :, :3].copy()
        seg_mask = np.asarray(seg, dtype=np.int32).reshape(HEIGHT, WIDTH) if with_seg else None
        if style is not None:
            background = seg_mask == -1
            rgb[background] = self._sky(view, style, background)
        return CameraFrame(rgb, view, self.projection_matrix, seg_mask)

    def _sky(self, view: tuple[float, ...], style: SceneStyle, mask: np.ndarray) -> np.ndarray:
        """Sky colors for the masked pixels, blended from horizon to zenith by elevation."""
        world_to_cv = np.diag([1.0, -1.0, -1.0]) @ np.array(view).reshape(4, 4, order="F")[:3, :3]
        up = self._pixel_rays[mask] @ world_to_cv[:, 2]
        # Below the horizon the far plane cuts off the ground, which reads as haze.
        t = np.clip(up / 0.6, 0.0, 1.0) ** 0.7
        horizon = np.array(style.sky_horizon)
        zenith = np.array(style.sky_zenith)
        return ((horizon + t[:, None] * (zenith - horizon)) * 255).astype(np.uint8)
