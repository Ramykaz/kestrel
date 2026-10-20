"""Gate geometry and URDF generation.

Gate frame convention: origin at the center of the opening, halfway through the frame
depth. +x is the direction of travel through the gate, so the front face (the side a
drone approaches from) is at x = -depth / 2. +z is up and +y is to the left when looking
along +x.
"""

from __future__ import annotations

import hashlib
import os
import tempfile
from pathlib import Path

import numpy as np

INNER_SIZE_M = 0.60
OUTER_SIZE_M = 0.80
FRAME_DEPTH_M = 0.05
POST_WIDTH_M = 0.04

GATE_RGBA = (0.95, 0.45, 0.10, 1.0)

_URDF_DIR = Path(tempfile.gettempdir()) / "kestrel_urdf"


def inner_corners_local(depth: float = FRAME_DEPTH_M) -> np.ndarray:
    """Inner corners on the front face, ordered top left, top right, bottom right, bottom left
    as seen from the front (looking along +x)."""
    half = INNER_SIZE_M / 2
    front = -depth / 2
    return np.array(
        [
            [front, half, half],
            [front, -half, half],
            [front, -half, -half],
            [front, half, -half],
        ]
    )


def outer_box_corners_local(depth: float = FRAME_DEPTH_M) -> np.ndarray:
    """The 8 corners of the frame's outer bounding cuboid."""
    half = OUTER_SIZE_M / 2
    return np.array(
        [[x, y, z] for x in (-depth / 2, depth / 2) for y in (half, -half) for z in (half, -half)]
    )


def _box_link_xml(name: str, size: tuple[float, float, float], xyz: tuple[float, ...]) -> str:
    size_str = " ".join(f"{v:.5f}" for v in size)
    xyz_str = " ".join(f"{v:.5f}" for v in xyz)
    rgba = " ".join(f"{v:.3f}" for v in GATE_RGBA)
    geometry = f'<geometry><box size="{size_str}"/></geometry>'
    material = f'<material name="gate"><color rgba="{rgba}"/></material>'
    return f"""
  <link name="{name}">
    <inertial><mass value="0"/><inertia ixx="0" ixy="0" ixz="0" iyy="0" iyz="0" izz="0"/></inertial>
    <visual><origin xyz="{xyz_str}"/>{geometry}{material}</visual>
    <collision><origin xyz="{xyz_str}"/>{geometry}</collision>
  </link>"""


def gate_urdf_text(center_height: float, depth: float = FRAME_DEPTH_M) -> str:
    """URDF for one gate whose base sits on the ground and whose opening is centered at
    center_height. The base link origin is on the ground directly below the gate center."""
    bar = (OUTER_SIZE_M - INNER_SIZE_M) / 2
    bar_center = (INNER_SIZE_M + bar) / 2
    post_height = center_height - OUTER_SIZE_M / 2
    if post_height <= 0:
        raise ValueError(f"gate center height {center_height} m leaves no room for the post")
    parts = [
        ("top", (depth, OUTER_SIZE_M, bar), (0.0, 0.0, center_height + bar_center)),
        ("bottom", (depth, OUTER_SIZE_M, bar), (0.0, 0.0, center_height - bar_center)),
        ("left", (depth, bar, INNER_SIZE_M), (0.0, bar_center, center_height)),
        ("right", (depth, bar, INNER_SIZE_M), (0.0, -bar_center, center_height)),
        ("post", (POST_WIDTH_M, POST_WIDTH_M, post_height), (0.0, 0.0, post_height / 2)),
    ]
    links = "".join(_box_link_xml(name, size, xyz) for name, size, xyz in parts)
    joints = "".join(
        f"""
  <joint name="base_{name}" type="fixed"><parent link="base"/><child link="{name}"/></joint>"""
        for name, _, _ in parts
    )
    base = """
  <link name="base">
    <inertial><mass value="0"/><inertia ixx="0" ixy="0" ixz="0" iyy="0" iyz="0" izz="0"/></inertial>
  </link>"""
    return f'<?xml version="1.0"?>\n<robot name="gate">{base}{links}{joints}\n</robot>\n'


def gate_urdf_path(center_height: float, depth: float = FRAME_DEPTH_M) -> Path:
    """Write the gate URDF once per distinct geometry and return its path."""
    text = gate_urdf_text(center_height, depth)
    digest = hashlib.sha256(text.encode()).hexdigest()[:16]
    path = _URDF_DIR / f"gate_{digest}.urdf"
    if not path.exists():
        # Written per process then renamed, so parallel data workers never read a partial file.
        _URDF_DIR.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(f".{os.getpid()}.tmp")
        tmp.write_text(text)
        tmp.replace(path)
    return path
