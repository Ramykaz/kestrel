"""Export trained detector weights to ONNX for the pilot and the website."""

from __future__ import annotations

import math
import shutil
from pathlib import Path

from kestrel.sim.camera import HEIGHT, WIDTH


def export_input_size(train_imgsz: int) -> tuple[int, int]:
    """(height, width) of the exported model input: the camera image scaled to the training
    resolution, rounded up to the network stride of 32."""
    scale = train_imgsz / max(WIDTH, HEIGHT)
    return (math.ceil(HEIGHT * scale / 32) * 32, math.ceil(WIDTH * scale / 32) * 32)


def export_onnx(weights: Path, out: Path, train_imgsz: int) -> Path:
    """Write an fp32 ONNX model without built-in NMS to out and return its path."""
    from ultralytics import YOLO

    exported = YOLO(str(weights)).export(
        format="onnx", imgsz=list(export_input_size(train_imgsz)), simplify=True, nms=False
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(exported, out)
    return out
