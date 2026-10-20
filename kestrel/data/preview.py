"""Contact sheet of labeled frames for visual QA."""

from __future__ import annotations

from PIL import Image, ImageDraw

from kestrel.data.generate import LabeledFrame
from kestrel.data.labels import CLASS_GATE, CLASS_GATE_BACK, CLASS_NAMES, OCCLUDED, VISIBLE

BOX_COLORS = {CLASS_GATE: (40, 110, 200), CLASS_GATE_BACK: (200, 40, 40)}
KEYPOINT_COLORS = [(230, 30, 30), (30, 200, 30), (30, 80, 230), (240, 210, 20)]
KEYPOINT_RADIUS_PX = 3


def draw_labels(frame: LabeledFrame) -> Image.Image:
    """The frame with boxes, class names and keypoints (filled visible, hollow occluded).

    Keypoint colors: top left red, top right green, bottom right blue, bottom left yellow.
    """
    image = Image.fromarray(frame.rgb)
    draw = ImageDraw.Draw(image)
    for label in frame.labels:
        color = BOX_COLORS[label.class_id]
        x0, y0, x1, y1 = (label.box_xyxy + 0.5).tolist()
        draw.rectangle([x0, y0, x1, y1], outline=color, width=1)
        draw.text((x0 + 2, y0 + 1), f"{CLASS_NAMES[label.class_id]} {label.gate_index}", fill=color)
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


def contact_sheet(frames: list[LabeledFrame], columns: int = 8) -> Image.Image:
    tiles = [draw_labels(frame) for frame in frames]
    width, height = tiles[0].size
    rows = -(-len(tiles) // columns)
    sheet = Image.new("RGB", (columns * width, rows * height), (255, 255, 255))
    for i, tile in enumerate(tiles):
        sheet.paste(tile, ((i % columns) * width, (i // columns) * height))
    return sheet
