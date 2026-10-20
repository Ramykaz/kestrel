"""What one flight leaves behind: pose, pilot state, target and detections at 30 Hz."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum

import numpy as np


class PilotState(IntEnum):
    SEARCH = 0
    APPROACH = 1
    COMMIT = 2
    EXIT = 3


class FlightEnd(IntEnum):
    FINISHED = 0
    CRASH = 1
    TIMEOUT = 2


@dataclass(frozen=True)
class Detection:
    class_id: int
    confidence: float
    box_xyxy: np.ndarray
    """Pixels, OpenCV convention."""
    keypoints: np.ndarray
    """4x2 pixels: top left, top right, bottom right, bottom left."""
    visibility: np.ndarray


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
    gate_pass_times: list[float] = field(default_factory=list)
    end: FlightEnd | None = None
    end_time: float | None = None
    crash_body_kind: str | None = None
    """For crashes: what was hit, for example gate_frame, gate_post or ground."""
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
    ) -> None:
        self.times.append(time_s)
        self.positions.append(position.copy())
        self.quaternions.append(quaternion.copy())
        self.targets.append(target.copy())
        self.states.append(state)
        self.detections.append(detections)

    @property
    def gates_passed(self) -> int:
        return len(self.gate_pass_times)

    @property
    def lap_time_s(self) -> float | None:
        if self.end is not FlightEnd.FINISHED:
            return None
        return self.gate_pass_times[-1]
