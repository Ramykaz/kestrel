"""Contact sheet of labeled frames for visual QA."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from kestrel.data.labels import (
    CLASS_GATE,
    CLASS_GATE_BACK,
    CLASS_NAMES,
    OCCLUDED,
    VISIBLE,
    GateLabel,
    read_yolo_label,
)

BOX_COLORS = {CLASS_GATE: (40, 110, 200), CLASS_GATE_BACK: (200, 40, 40)}
KEYPOINT_COLORS = [(230, 30, 30), (30, 200, 30), (30, 80, 230), (240, 210, 20)]
KEYPOINT_RADIUS_PX = 3


def draw_labels(rgb: np.ndarray, labels: list[GateLabel]) -> Image.Image:
    """The frame with boxes, class names and keypoints (filled visible, hollow occluded).

    Keypoint colors: top left red, top right green, bottom right blue, bottom left yellow.
    """
    image = Image.fromarray(rgb)
    draw = ImageDraw.Draw(image)
    for label in labels:
        color = BOX_COLORS[label.class_id]
        x0, y0, x1, y1 = (label.box_xyxy + 0.5).tolist()
        draw.rectangle([x0, y0, x1, y1], outline=color, width=1)
        draw.text((x0 + 2, y0 + 1), CLASS_NAMES[label.class_id], fill=color)
        for (u, v), vis, kp_color in zip(
            label.keypoints + 0.5, label.visibility, KEYPOINT_COLORS, strict=True
        ):
            if vis not in (VISIBLE, OCCLUDED):
                continue
            r = KEYPOINT_RADIUS_PX
            draw.ellipse(
                [u - r, v - r, u + r, v + r],
                fill=kp_color if vis == VISIBLE else None,
                outline=kp_color,
            )
    return image


def dataset_contact_sheet(
    dataset: Path, count: int, seed: int, columns: int = 8
) -> tuple[Image.Image, list[list[GateLabel]]]:
    """Random frames from a written dataset with their label files drawn on top."""
    images = sorted((dataset / "images").glob("*/*.jpg"))
    if not images:
        raise FileNotFoundError(f"no images under {dataset / 'images'}")
    rng = np.random.default_rng(seed)
    chosen = [images[i] for i in sorted(rng.choice(len(images), min(count, len(images)), False))]
    tiles, all_labels = [], []
    for image_path in chosen:
        rgb = np.asarray(Image.open(image_path).convert("RGB"))
        label_path = dataset / "labels" / image_path.parent.name / f"{image_path.stem}.txt"
        labels = read_yolo_label(label_path, rgb.shape[1], rgb.shape[0])
        tiles.append(draw_labels(rgb, labels))
        all_labels.append(labels)
    width, height = tiles[0].size
    rows = -(-len(tiles) // columns)
    sheet = Image.new("RGB", (columns * width, rows * height), (255, 255, 255))
    for i, tile in enumerate(tiles):
        sheet.paste(tile, ((i % columns) * width, (i // columns) * height))
    return sheet, all_labels
