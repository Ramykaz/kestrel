"""Gate detector inference on ONNX: letterbox, decode, NMS and keypoint checks.

The model is a YOLO pose network exported without NMS. Its output is (1, 4 + classes + 12, N):
box center x, y, width, height in input pixels, one score per class, then x, y and a
visibility score for each of the 4 corners.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

CLASS_GATE = 0
CLASS_GATE_BACK = 1
CLASS_COUNT = 2
KEYPOINTS = 4
LETTERBOX_FILL = 114
NMS_IOU = 0.5
KEYPOINT_VISIBLE_SCORE = 0.5
VISIBLE = 2
NOT_VISIBLE = 0


@dataclass(frozen=True)
class Detection:
    class_id: int
    confidence: float
    box_xyxy: np.ndarray
    """Pixels, OpenCV convention."""
    keypoints: np.ndarray
    """4x2 pixels: top left, top right, bottom right, bottom left of the gate opening as seen
    from its front."""
    visibility: np.ndarray


def _iou(box: np.ndarray, others: np.ndarray) -> np.ndarray:
    top_left = np.maximum(box[:2], others[:, :2])
    bottom_right = np.minimum(box[2:], others[:, 2:])
    overlap = np.prod(np.clip(bottom_right - top_left, 0, None), axis=1)
    area = np.prod(box[2:] - box[:2])
    areas = np.prod(others[:, 2:] - others[:, :2], axis=1)
    return overlap / np.maximum(area + areas - overlap, 1e-9)


def non_max_suppression(boxes: np.ndarray, scores: np.ndarray, iou: float) -> list[int]:
    """Indices of kept boxes, best first. Class-agnostic: a gate is either front or back."""
    order = np.argsort(-scores)
    keep = []
    while len(order):
        best = order[0]
        keep.append(int(best))
        rest = order[1:]
        order = rest[_iou(boxes[best], boxes[rest]) <= iou]
    return keep


def corner_winding_matches_class(detection: Detection) -> bool:
    """Seen from the front, top left, top right, bottom right, bottom left run clockwise on
    screen (y points down); seen from behind they run the other way. A quad that is not convex
    or winds against its class would give PnP a mirrored pose."""
    corners = detection.keypoints
    edges = np.roll(corners, -1, axis=0) - corners
    next_edges = np.roll(edges, -1, axis=0)
    turns = edges[:, 0] * next_edges[:, 1] - edges[:, 1] * next_edges[:, 0]
    clockwise = bool(np.all(turns > 0))
    anticlockwise = bool(np.all(turns < 0))
    return clockwise if detection.class_id == CLASS_GATE else anticlockwise


def decode_predictions(
    raw: np.ndarray,
    scale: float,
    offset: np.ndarray,
    frame_size: tuple[int, int],
    conf_thresh: float,
) -> list[Detection]:
    """Turn raw network rows (one per candidate) into detections in frame pixels.

    scale and offset describe the letterbox: input pixel = frame pixel edge * scale + offset.
    Boxes are clipped to the frame of size (width, height), as the training labels are.
    """
    low = np.array([-0.5, -0.5])
    high = np.array(frame_size, dtype=float) - 0.5
    class_scores = raw[:, 4 : 4 + CLASS_COUNT]
    confidence = class_scores.max(axis=1)
    candidates = np.flatnonzero(confidence >= conf_thresh)
    if len(candidates) == 0:
        return []
    rows = raw[candidates]
    centers, sizes = rows[:, 0:2], rows[:, 2:4]
    boxes = np.c_[centers - sizes / 2, centers + sizes / 2]
    keep = non_max_suppression(boxes, confidence[candidates], NMS_IOU)
    detections = []
    for i in keep:
        # The network predicts in pixel-edge coordinates (top left corner of the image is
        # 0); OpenCV pixel centers sit half a pixel further in.
        box = np.clip((boxes[i].reshape(2, 2) - offset) / scale - 0.5, low, high).ravel()
        points = rows[i, 4 + CLASS_COUNT :].reshape(KEYPOINTS, 3)
        keypoints = (points[:, :2] - offset) / scale - 0.5
        visibility = np.where(points[:, 2] >= KEYPOINT_VISIBLE_SCORE, VISIBLE, NOT_VISIBLE)
        detections.append(
            Detection(
                class_id=int(class_scores[candidates[i]].argmax()),
                confidence=float(confidence[candidates[i]]),
                box_xyxy=box,
                keypoints=keypoints,
                visibility=visibility,
            )
        )
    return detections


class GateDetector:
    """Runs the exported detector on single 320x240 RGB frames on the CPU."""

    def __init__(self, onnx_path: Path, threads: int = 1) -> None:
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(
            str(onnx_path), options, providers=["CPUExecutionProvider"]
        )
        model_input = self.session.get_inputs()[0]
        self.input_name = model_input.name
        _, _, self.input_height, self.input_width = model_input.shape
        output_rows = self.session.get_outputs()[0].shape[1]
        if output_rows != 4 + CLASS_COUNT + 3 * KEYPOINTS:
            raise ValueError(
                f"{onnx_path} outputs {output_rows} rows per candidate; a 2 class, 4 keypoint "
                f"pose model exported without NMS has {4 + CLASS_COUNT + 3 * KEYPOINTS}"
            )

    def _letterbox(self, rgb: np.ndarray) -> tuple[np.ndarray, float, np.ndarray]:
        height, width = rgb.shape[:2]
        scale = min(self.input_width / width, self.input_height / height)
        resized_w, resized_h = round(width * scale), round(height * scale)
        image = rgb if scale == 1 else cv2.resize(rgb, (resized_w, resized_h), cv2.INTER_LINEAR)
        # Same split of the padding as ultralytics' LetterBox(center=True).
        pad_w = (self.input_width - resized_w) / 2
        pad_h = (self.input_height - resized_h) / 2
        left, top = round(pad_w - 0.1), round(pad_h - 0.1)
        canvas = np.full((self.input_height, self.input_width, 3), LETTERBOX_FILL, np.uint8)
        canvas[top : top + resized_h, left : left + resized_w] = image
        return canvas, scale, np.array([left, top], dtype=float)

    def detect(self, rgb: np.ndarray, conf_thresh: float) -> list[Detection]:
        """Detections above conf_thresh, best first, in the frame's pixel coordinates."""
        canvas, scale, offset = self._letterbox(rgb)
        blob = canvas.transpose(2, 0, 1)[None].astype(np.float32) / 255.0
        raw = self.session.run(None, {self.input_name: blob})[0][0].T
        frame_size = (rgb.shape[1], rgb.shape[0])
        return decode_predictions(raw, scale, offset, frame_size, conf_thresh)
