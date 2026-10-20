"""How a scene looks: textures, colors, lighting, sky and distractor objects.

The ground mesh (pybullet_data plane100) repeats its texture every 2 m, so ground textures are
drawn as seamless 2 m tiles.
"""

from __future__ import annotations

import colorsys
import os
import tempfile
from dataclasses import dataclass

import numpy as np
import pybullet as p
from PIL import Image

from kestrel.sim.gates import FRONT_MARKER_LINK_INDICES

GROUND_TEXTURE_PX = 256
GROUND_TILE_M = 2.0
GATE_TEXTURE_PX = 64

GROUND_PATTERNS = (
    "checker",
    "tiles",
    "stripes",
    "grass",
    "dirt",
    "gravel",
    "sand",
    "concrete",
    "asphalt",
    "voronoi",
    "dots",
    "waves",
)
GATE_PATTERNS = ("solid", "stripes", "checker", "noise")
DISTRACTOR_KINDS = ("pole", "box", "cone")

RGB = tuple[float, float, float]


@dataclass(frozen=True)
class Lighting:
    direction: tuple[float, float, float]
    """Points from the scene toward the light."""
    color: RGB
    ambient: float
    diffuse: float
    specular: float

    def render_kwargs(self) -> dict:
        """Every lighting argument of getCameraImage. TinyRenderer keeps the previous call's
        light state for arguments that are left out, so all of them are always passed."""
        return {
            "lightDirection": list(self.direction),
            "lightColor": list(self.color),
            "lightAmbientCoeff": self.ambient,
            "lightDiffuseCoeff": self.diffuse,
            "lightSpecularCoeff": self.specular,
            "shadow": 0,
        }


@dataclass(frozen=True)
class SceneStyle:
    sky_horizon: RGB
    sky_zenith: RGB
    ground_pattern: str
    ground_colors: tuple[RGB, RGB]
    gate_color: RGB
    gate_pattern: str
    gate_pattern_color: RGB
    lighting: Lighting
    texture_seed: int


@dataclass(frozen=True)
class Distractor:
    kind: str
    position: tuple[float, float]
    """Ground point under the object's center."""
    size: tuple[float, float]
    """Width (or diameter) and height in metres."""
    yaw: float
    color: RGB

    @property
    def footprint_radius(self) -> float:
        return self.size[0] / 2 * (2**0.5 if self.kind == "box" else 1.0)


def _periodic_noise(rng: np.random.Generator, size: int, cutoff: float) -> np.ndarray:
    """Seamless noise in [0, 1]: white noise low-passed in the frequency domain."""
    spectrum = np.fft.fft2(rng.normal(size=(size, size)))
    freq = np.fft.fftfreq(size)
    radius = np.hypot(*np.meshgrid(freq, freq))
    field = np.real(np.fft.ifft2(spectrum * np.exp(-((radius / cutoff) ** 2))))
    field -= field.min()
    return field / max(field.max(), 1e-9)


def _blend(mask: np.ndarray, colors: tuple[RGB, RGB]) -> np.ndarray:
    a, b = (np.array(c, dtype=float) for c in colors)
    return (a + mask[..., None] * (b - a)) * 255


def ground_texture(pattern: str, colors: tuple[RGB, RGB], seed: int) -> np.ndarray:
    """A seamless GROUND_TEXTURE_PX square tile covering GROUND_TILE_M on each side."""
    if pattern not in GROUND_PATTERNS:
        raise ValueError(f"unknown ground pattern {pattern!r}")
    rng = np.random.default_rng(seed)
    n = GROUND_TEXTURE_PX
    y, x = np.mgrid[0:n, 0:n] / n
    if pattern == "checker":
        mask = ((np.floor(x * 2) + np.floor(y * 2)) % 2).astype(float)
    elif pattern == "tiles":
        mask = ((x * 4 % 1 < 0.05) | (y * 4 % 1 < 0.05)).astype(float)
    elif pattern == "stripes":
        mask = (x * 2 % 1 < 0.5).astype(float) * 0.6 + _periodic_noise(rng, n, 0.08) * 0.4
    elif pattern == "grass":
        mask = _periodic_noise(rng, n, 0.25) * 0.6 + _periodic_noise(rng, n, 0.02) * 0.4
    elif pattern == "dirt":
        mask = _periodic_noise(rng, n, 0.015)
    elif pattern == "gravel":
        mask = (_periodic_noise(rng, n, 0.3) > 0.55).astype(float)
    elif pattern == "sand":
        ripples = 0.5 + 0.5 * np.sin(2 * np.pi * (6 * y + 0.8 * _periodic_noise(rng, n, 0.02)))
        mask = 0.7 * _periodic_noise(rng, n, 0.04) + 0.3 * ripples
    elif pattern == "concrete":
        seams = ((x % 0.5 < 0.012) | (y % 0.5 < 0.012)).astype(float)
        mask = np.maximum(0.35 * _periodic_noise(rng, n, 0.05), seams)
    elif pattern == "asphalt":
        mask = 0.8 * _periodic_noise(rng, n, 0.4) + 0.2 * _periodic_noise(rng, n, 0.01)
    elif pattern == "voronoi":
        seeds = rng.random((12, 2))
        offsets = np.array([[dx, dy] for dx in (-1, 0, 1) for dy in (-1, 0, 1)])
        tiled = (seeds[:, None, :] + offsets[None, :, :]).reshape(-1, 2)
        distances = np.hypot(x[..., None] - tiled[:, 0], y[..., None] - tiled[:, 1])
        # Rows of tiled run seed by seed, each over all offsets, so a seed's copies share a shade.
        mask = (np.argmin(distances, axis=-1) // len(offsets) % 3) / 2
    elif pattern == "dots":
        cell = (np.stack([x, y], axis=-1) * 4) % 1 - 0.5
        mask = (np.hypot(cell[..., 0], cell[..., 1]) < 0.22).astype(float)
    else:
        warp = 0.15 * _periodic_noise(rng, n, 0.02)
        mask = 0.5 + 0.5 * np.sin(2 * np.pi * (3 * x + 2 * y + warp))
    return np.clip(_blend(mask, colors), 0, 255).astype(np.uint8)


def gate_texture(pattern: str, color: RGB, pattern_color: RGB, seed: int) -> np.ndarray:
    if pattern not in GATE_PATTERNS:
        raise ValueError(f"unknown gate pattern {pattern!r}")
    n = GATE_TEXTURE_PX
    y, x = np.mgrid[0:n, 0:n] / n
    if pattern == "solid":
        mask = np.zeros((n, n))
    elif pattern == "stripes":
        mask = ((x + y) * 4 % 1 < 0.5).astype(float)
    elif pattern == "checker":
        mask = ((np.floor(x * 4) + np.floor(y * 4)) % 2).astype(float)
    else:
        mask = _periodic_noise(np.random.default_rng(seed), n, 0.15)
    return np.clip(_blend(mask, (color, pattern_color)), 0, 255).astype(np.uint8)


def _load_texture(client: int, rgb: np.ndarray) -> int:
    # loadTexture only reads files; the pixels stay in the client after the file is gone.
    handle, path = tempfile.mkstemp(suffix=".png", prefix="kestrel_tex_")
    os.close(handle)
    try:
        Image.fromarray(rgb).save(path)
        texture_id = p.loadTexture(path, physicsClientId=client)
    finally:
        os.unlink(path)
    if texture_id < 0:
        raise RuntimeError("pybullet failed to load a generated texture")
    return texture_id


def apply_style(client: int, ground_id: int, gate_body_ids: list[int], style: SceneStyle) -> None:
    """Texture the ground and every gate of a loaded course. The white front markers keep
    their color."""
    ground = ground_texture(style.ground_pattern, style.ground_colors, style.texture_seed)
    p.changeVisualShape(
        ground_id,
        -1,
        textureUniqueId=_load_texture(client, ground),
        rgbaColor=[1, 1, 1, 1],
        physicsClientId=client,
    )
    gate = gate_texture(
        style.gate_pattern, style.gate_color, style.gate_pattern_color, style.texture_seed
    )
    gate_texture_id = _load_texture(client, gate)
    for body in gate_body_ids:
        for link in range(p.getNumJoints(body, physicsClientId=client)):
            if link in FRONT_MARKER_LINK_INDICES:
                continue
            p.changeVisualShape(
                body,
                link,
                textureUniqueId=gate_texture_id,
                rgbaColor=[1, 1, 1, 1],
                physicsClientId=client,
            )


def add_distractor(client: int, distractor: Distractor) -> int:
    """Spawn a static pole, box or cone standing on the ground and return its body id."""
    width, height = distractor.size
    rgba = [*distractor.color, 1.0]
    if distractor.kind == "pole":
        shape = {"shapeType": p.GEOM_CYLINDER, "radius": width / 2}
        visual = p.createVisualShape(length=height, rgbaColor=rgba, physicsClientId=client, **shape)
        collision = p.createCollisionShape(height=height, physicsClientId=client, **shape)
    elif distractor.kind == "box":
        half = [width / 2, width / 2, height / 2]
        visual = p.createVisualShape(
            p.GEOM_BOX, halfExtents=half, rgbaColor=rgba, physicsClientId=client
        )
        collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=half, physicsClientId=client)
    elif distractor.kind == "cone":
        vertices, indices = _cone_mesh(width / 2, height)
        visual = p.createVisualShape(
            p.GEOM_MESH,
            vertices=vertices,
            indices=indices,
            rgbaColor=rgba,
            physicsClientId=client,
        )
        collision = p.createCollisionShape(p.GEOM_MESH, vertices=vertices, physicsClientId=client)
    else:
        raise ValueError(f"unknown distractor kind {distractor.kind!r}")
    # Cylinders and boxes are centered on their origin; the cone mesh starts at its base.
    base_z = 0.0 if distractor.kind == "cone" else height / 2
    return p.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=collision,
        baseVisualShapeIndex=visual,
        basePosition=[*distractor.position, base_z],
        baseOrientation=p.getQuaternionFromEuler([0, 0, distractor.yaw]),
        physicsClientId=client,
    )


def _cone_mesh(radius: float, height: float, sides: int = 16) -> tuple[list, list]:
    angles = np.linspace(0, 2 * np.pi, sides, endpoint=False)
    rim = [[radius * np.cos(a), radius * np.sin(a), 0.0] for a in angles]
    vertices = [*rim, [0.0, 0.0, height], [0.0, 0.0, 0.0]]
    apex, center = sides, sides + 1
    indices = []
    for i in range(sides):
        j = (i + 1) % sides
        indices += [i, j, apex, j, i, center]
    return [[float(v) for v in vertex] for vertex in vertices], indices


def hsv(h: float, s: float, v: float) -> RGB:
    r, g, b = colorsys.hsv_to_rgb(h % 1.0, s, v)
    return (r, g, b)
