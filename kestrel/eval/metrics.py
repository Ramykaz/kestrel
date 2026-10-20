"""Flight metrics on the held-out suite (SPEC section 9.1)."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from kestrel.pilot.flight_log import FlightEnd, FlightLog


@dataclass(frozen=True)
class FlightMetrics:
    completion_rate: float
    gates_per_course: float
    lap_time_s: float | None
    crash_rate: float
    night_completion: float | None

    def to_dict(self) -> dict:
        return asdict(self)


def flight_metrics(tech_flights: list[FlightLog], night_flights: list[FlightLog]) -> FlightMetrics:
    """Completion, gates and crashes over tech flights; night completion over night flights."""
    if not tech_flights:
        raise ValueError("flight metrics need at least one tech flight")
    finished = [f for f in tech_flights if f.end is FlightEnd.FINISHED]
    lap_times = [f.lap_time_s for f in finished if f.lap_time_s is not None]
    night_finished = sum(f.end is FlightEnd.FINISHED for f in night_flights)
    return FlightMetrics(
        completion_rate=len(finished) / len(tech_flights),
        gates_per_course=float(np.mean([f.gates_passed for f in tech_flights])),
        lap_time_s=float(np.median(lap_times)) if lap_times else None,
        crash_rate=sum(f.end is FlightEnd.CRASH for f in tech_flights) / len(tech_flights),
        night_completion=night_finished / len(night_flights) if night_flights else None,
    )
