"""What one flight leaves behind: pose, pilot state, target and detections at 30 Hz."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path

import numpy as np

from kestrel.perception.detect import Detection
from kestrel.sim.camera import HEIGHT, WIDTH


class PilotState(IntEnum):
    SEARCH = 0
    APPROACH = 1
    COMMIT = 2
    EXIT = 3


class FlightEnd(IntEnum):
    FINISHED = 0
    CRASH = 1
    TIMEOUT = 2


@dataclass
class FlightLog:
    course_seed: int
    course_preset: str
    sample_hz: float
    times: list[float] = field(default_factory=list)
    positions: list[np.ndarray] = field(default_factory=list)
    quaternions: list[np.ndarray] = field(default_factory=list)
    targets: list[np.ndarray] = field(default_factory=list)
    states: list[PilotState] = field(default_factory=list)
    detections: list[list[Detection]] = field(default_factory=list)
    believed_gates: list[np.ndarray] = field(default_factory=list)
    """Where the pilot thought its target gate was; NaN when it had no target."""
    gate_pass_times: list[float] = field(default_factory=list)
    end: FlightEnd | None = None
    end_time: float | None = None
    crash_body_kind: str | None = None
    """For crashes: what was hit, for example gate_frame, gate_post or ground."""
    crash_gate_index: int | None = None
    """For crashes into a gate: which one, in course order."""
    crash_frames: list[np.ndarray] = field(default_factory=list)
    """Last 2 s of onboard frames, kept only for crashed flights."""

    def record(
        self,
        time_s: float,
        position: np.ndarray,
        quaternion: np.ndarray,
        target: np.ndarray,
        state: PilotState,
        detections: list[Detection],
        believed_gate: np.ndarray | None = None,
    ) -> None:
        self.times.append(time_s)
        self.positions.append(position.copy())
        self.quaternions.append(quaternion.copy())
        self.targets.append(target.copy())
        self.states.append(state)
        self.detections.append(detections)
        self.believed_gates.append(
            np.full(3, np.nan) if believed_gate is None else believed_gate.copy()
        )

    @property
    def gates_passed(self) -> int:
        return len(self.gate_pass_times)

    @property
    def lap_time_s(self) -> float | None:
        if self.end is not FlightEnd.FINISHED:
            return None
        return self.gate_pass_times[-1]


def save_flight_log(log: FlightLog, path: Path) -> None:
    """Write a log as a compressed .npz with a JSON header for the scalar fields."""
    counts = np.array([len(frame) for frame in log.detections], dtype=np.int32)
    rows = [
        [d.class_id, d.confidence, *d.box_xyxy, *d.keypoints.ravel(), *d.visibility]
        for frame in log.detections
        for d in frame
    ]
    header = {
        "course_seed": log.course_seed,
        "course_preset": log.course_preset,
        "sample_hz": log.sample_hz,
        "end": None if log.end is None else log.end.name,
        "end_time": log.end_time,
        "crash_body_kind": log.crash_body_kind,
        "crash_gate_index": log.crash_gate_index,
    }
    np.savez_compressed(
        path,
        header=np.array(json.dumps(header)),
        times=np.array(log.times, dtype=np.float64),
        positions=np.array(log.positions, dtype=np.float64).reshape(-1, 3),
        quaternions=np.array(log.quaternions, dtype=np.float64).reshape(-1, 4),
        targets=np.array(log.targets, dtype=np.float64).reshape(-1, 3),
        states=np.array(log.states, dtype=np.uint8),
        believed_gates=np.array(log.believed_gates, dtype=np.float64).reshape(-1, 3),
        detection_counts=counts,
        detections=np.array(rows, dtype=np.float32).reshape(-1, 18),
        gate_pass_times=np.array(log.gate_pass_times, dtype=np.float64),
        crash_frames=np.array(log.crash_frames, dtype=np.uint8).reshape(-1, HEIGHT, WIDTH, 3),
    )


def load_flight_log(path: Path) -> FlightLog:
    with np.load(path) as data:
        header = json.loads(str(data["header"]))
        log = FlightLog(header["course_seed"], header["course_preset"], header["sample_hz"])
        log.times = data["times"].tolist()
        log.positions = list(data["positions"])
        log.quaternions = list(data["quaternions"])
        log.targets = list(data["targets"])
        log.states = [PilotState(s) for s in data["states"]]
        log.believed_gates = list(data["believed_gates"])
        rows = iter(data["detections"])
        log.detections = [
            [_detection_from_row(next(rows)) for _ in range(count)]
            for count in data["detection_counts"]
        ]
        log.gate_pass_times = data["gate_pass_times"].tolist()
        log.crash_frames = list(data["crash_frames"])
    log.end = None if header["end"] is None else FlightEnd[header["end"]]
    log.end_time = header["end_time"]
    log.crash_body_kind = header["crash_body_kind"]
    log.crash_gate_index = header["crash_gate_index"]
    return log


def _detection_from_row(row: np.ndarray) -> Detection:
    return Detection(
        class_id=int(row[0]),
        confidence=float(row[1]),
        box_xyxy=row[2:6].astype(np.float64),
        keypoints=row[6:14].reshape(4, 2).astype(np.float64),
        visibility=row[14:18].astype(int),
    )
