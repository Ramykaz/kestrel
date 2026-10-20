"""The pilot's tunable parameters (SPEC 8.3) and their hard bounds."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields

import yaml

from kestrel.paths import CONFIG_DIR

BOUNDS = {
    "conf_thresh": (0.05, 0.9),
    "max_reproj_px": (0.5, 10.0),
    "ema_alpha": (0.05, 1.0),
    "forget_s": (0.2, 10.0),
    # SPEC 6.1: never above 4 m/s.
    "speed_cap": (0.3, 4.0),
    "approach_offset": (0.3, 3.0),
    "commit_dist": (0.3, 3.0),
    "exit_dist": (0.2, 3.0),
    "search_yaw_rate": (0.1, 2.0),
}


@dataclass(frozen=True)
class PilotParams:
    conf_thresh: float
    max_reproj_px: float
    ema_alpha: float
    """Weight of a new observation when it updates a gate estimate."""
    forget_s: float
    speed_cap: float
    """m/s"""
    approach_offset: float
    """Metres in front of the gate for the approach point."""
    commit_dist: float
    """Within this many metres of the gate, the target stops updating."""
    exit_dist: float
    """Metres to fly past the gate before choosing the next one."""
    search_yaw_rate: float
    """rad/s"""

    def __post_init__(self) -> None:
        for field in fields(self):
            value = getattr(self, field.name)
            low, high = BOUNDS[field.name]
            if not low <= value <= high:
                raise ValueError(f"{field.name} {value} is outside the hard bounds [{low}, {high}]")

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def load_pilot_preset(name: str) -> PilotParams:
    path = CONFIG_DIR / "pilot" / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"no pilot preset named {name!r} at {path}")
    raw = yaml.safe_load(path.read_text())
    expected = set(BOUNDS)
    if set(raw) != expected:
        missing, extra = sorted(expected - set(raw)), sorted(set(raw) - expected)
        raise ValueError(f"pilot preset {name!r}: missing {missing}, unexpected {extra}")
    return PilotParams(**{key: float(value) for key, value in raw.items()})
