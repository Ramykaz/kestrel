"""Everything the website needs, written as static files (SPEC section 12).

Flight files (.bin) are little endian:

    bytes 0-3    b"KFL1"
    bytes 4-7    uint32 length of the header JSON in bytes
    header JSON  UTF-8, padded with spaces so the arrays start on an 8 byte boundary
    arrays       back to back, in the order and with the dtypes listed in header["arrays"]

Arrays, with n frames at header["sample_hz"] and m detections in total:

    position     float16 (n, 3)   metres, world frame
    quaternion   float16 (n, 4)   x, y, z, w
    target       float16 (n, 3)   the pilot's current target point
    state        uint8   (n,)     PilotState
    det_count    uint8   (n,)     detections in each frame; frame i owns the next det_count[i]
    detections   float16 (m, 15)  class, confidence, box x0 y0 x1 y1, then 4 keypoints x y,
                                  then visibility packed as v0 + 4 v1 + 16 v2 + 64 v3;
                                  coordinates are normalized to the image size
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from kestrel.eval.metrics import FlightMetrics
from kestrel.pilot.flight_log import FlightLog
from kestrel.sim.camera import HEIGHT, WIDTH
from kestrel.sim.course import Course, eval_suite_seeds, load_preset

SCHEMA_VERSION = 1
FLIGHT_MAGIC = b"KFL1"
MAX_FILE_BYTES = 8 * 1024 * 1024
MAX_TOTAL_BYTES = 120 * 1024 * 1024
FLIGHT_TARGET_BYTES = 150 * 1024


def _round_floats(values: list[float], places: int = 3) -> list[float]:
    return [round(float(v), places) for v in values]


def course_json(course: Course) -> dict:
    return {
        "seed": course.seed,
        "preset": course.preset,
        "lighting": load_preset(course.preset).lighting,
        "gates": [
            {"center": _round_floats(list(g.center)), "yaw": round(g.yaw, 4)} for g in course.gates
        ],
        "decoys": [],
        "start": {
            "position": _round_floats(list(course.start_pos)),
            "yaw": round(course.start_yaw, 4),
        },
    }


def _detection_rows(log: FlightLog) -> tuple[np.ndarray, np.ndarray]:
    counts = np.array([len(frame) for frame in log.detections], dtype=np.uint8)
    rows = []
    size = np.array([WIDTH, HEIGHT], dtype=float)
    for frame in log.detections:
        for det in frame:
            box = ((det.box_xyxy.reshape(2, 2) + 0.5) / size).ravel()
            keypoints = ((det.keypoints + 0.5) / size).ravel()
            packed = sum(int(v) << (2 * i) for i, v in enumerate(det.visibility))
            rows.append([det.class_id, det.confidence, *box, *keypoints, packed])
    detections = np.array(rows, dtype=np.float16).reshape(-1, 15)
    return counts, detections


def flight_summary(log: FlightLog, gate_count: int, file_name: str) -> dict:
    if log.end is None or log.end_time is None:
        raise ValueError(f"flight on course {log.course_seed} has no recorded end")
    return {
        "file": file_name,
        "end": log.end.name.lower(),
        "end_time_s": round(log.end_time, 3),
        "gates_passed": log.gates_passed,
        "gate_count": gate_count,
        "lap_time_s": None if log.lap_time_s is None else round(log.lap_time_s, 3),
        "crash_body_kind": log.crash_body_kind,
    }


def encode_flight(log: FlightLog, gate_count: int, detections_source: str) -> bytes:
    """Serialize a flight log into the binary format described in the module docstring."""
    counts, detections = _detection_rows(log)
    arrays = [
        ("position", np.array(log.positions, dtype=np.float16).reshape(-1, 3)),
        ("quaternion", np.array(log.quaternions, dtype=np.float16).reshape(-1, 4)),
        ("target", np.array(log.targets, dtype=np.float16).reshape(-1, 3)),
        ("state", np.array([int(s) for s in log.states], dtype=np.uint8)),
        ("det_count", counts),
        ("detections", detections),
    ]
    header = {
        "schema": SCHEMA_VERSION,
        "course_seed": log.course_seed,
        "course_preset": log.course_preset,
        "sample_hz": log.sample_hz,
        "frames": len(log.times),
        "image_size": [WIDTH, HEIGHT],
        "detections_source": detections_source,
        "gate_pass_times_s": _round_floats(log.gate_pass_times),
        **{k: v for k, v in flight_summary(log, gate_count, "").items() if k != "file"},
        "arrays": [
            {"name": name, "dtype": str(array.dtype), "shape": list(array.shape)}
            for name, array in arrays
        ],
    }
    header_bytes = json.dumps(header, separators=(",", ":")).encode()
    padding = (-(8 + len(header_bytes))) % 8
    header_bytes += b" " * padding
    parts = [FLIGHT_MAGIC, len(header_bytes).to_bytes(4, "little"), header_bytes]
    parts += [array.astype(array.dtype.newbyteorder("<")).tobytes() for _, array in arrays]
    return b"".join(parts)


def flight_file_name(round_index: int, course_seed: int) -> str:
    return f"flights/r{round_index:02d}-s{course_seed}.bin"


@dataclass
class BundleRound:
    round_index: int
    status: str
    label: str
    metrics: FlightMetrics
    flights: dict[int, dict]
    card: dict | None = None
    report: dict | None = None
    ablation: FlightMetrics | None = None
    budgets: list[dict] | None = None

    def to_dict(self) -> dict:
        return {
            "round": self.round_index,
            "status": self.status,
            "label": self.label,
            "card": self.card,
            "report": self.report,
            "metrics": self.metrics.to_dict(),
            "ablation": None if self.ablation is None else self.ablation.to_dict(),
            "budgets": self.budgets or [],
            "flights": {str(seed): summary for seed, summary in self.flights.items()},
        }


def write_rounds_json(out_dir: Path, source: str, rounds: list[BundleRound]) -> Path:
    seeds = eval_suite_seeds()
    payload = {
        "schema": SCHEMA_VERSION,
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "source": source,
        "eval_suite": seeds,
        "rounds": [r.to_dict() for r in rounds],
    }
    path = out_dir / "rounds.json"
    path.write_text(json.dumps(payload, separators=(",", ":")))
    return path


def write_course_files(out_dir: Path, courses: list[Course]) -> None:
    (out_dir / "courses").mkdir(parents=True, exist_ok=True)
    for course in courses:
        path = out_dir / "courses" / f"s{course.seed}.json"
        path.write_text(json.dumps(course_json(course), separators=(",", ":")))


def check_bundle_size(out_dir: Path) -> tuple[int, int]:
    """Raise if any file breaks the hosting limits; return (file count, total bytes)."""
    files = [f for f in out_dir.rglob("*") if f.is_file()]
    oversized = [f for f in files if f.stat().st_size > MAX_FILE_BYTES]
    if oversized:
        raise ValueError(f"files above {MAX_FILE_BYTES} bytes: {[str(f) for f in oversized]}")
    total = sum(f.stat().st_size for f in files)
    if total > MAX_TOTAL_BYTES:
        raise ValueError(f"bundle is {total} bytes, above the {MAX_TOTAL_BYTES} byte limit")
    return len(files), total


def flights_over_target(out_dir: Path) -> list[tuple[str, int]]:
    return [
        (str(f.relative_to(out_dir)), f.stat().st_size)
        for f in sorted((out_dir / "flights").glob("*.bin"))
        if f.stat().st_size > FLIGHT_TARGET_BYTES
    ]
