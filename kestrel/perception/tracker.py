"""Gate memory: a short list of gate estimates built from noisy per-frame observations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from kestrel.perception.pose import GateObservation

# SPEC 8.2: an observation within this distance of an estimate updates it.
MATCH_RADIUS_M = 0.8
# A gate counts as confirmed after this many observations.
CONFIRM_HITS = 3
# PnP on a gate a few pixels tall pins down its position but hardly its facing direction, so
# observations farther than this update the direction with weight falling as 1 / distance^2.
FORWARD_TRUST_M = 2.5


@dataclass
class GateHypothesis:
    id: int
    center: np.ndarray
    forward: np.ndarray
    hits: int
    last_seen_s: float
    passed: bool = False

    @property
    def confirmed(self) -> bool:
        return self.hits >= CONFIRM_HITS


class GateTracker:
    def __init__(self, ema_alpha: float, forget_s: float) -> None:
        self.ema_alpha = ema_alpha
        self.forget_s = forget_s
        self.hypotheses: list[GateHypothesis] = []
        self._next_id = 0

    def update(
        self, observations: list[GateObservation], time_s: float, keep_id: int | None
    ) -> None:
        """Fold this frame's observations in, then drop estimates not seen for forget_s
        seconds. The estimate with keep_id (the current target) is never dropped."""
        for observation in observations:
            match = self._nearest(observation.center)
            if match is None:
                self.hypotheses.append(
                    GateHypothesis(
                        self._next_id,
                        observation.center.copy(),
                        observation.forward.copy(),
                        1,
                        time_s,
                    )
                )
                self._next_id += 1
                continue
            a = self.ema_alpha
            match.center = (1 - a) * match.center + a * observation.center
            b = a * min(1.0, (FORWARD_TRUST_M / observation.distance_m) ** 2)
            forward = (1 - b) * match.forward + b * observation.forward
            match.forward = forward / max(np.linalg.norm(forward), 1e-9)
            match.hits += 1
            match.last_seen_s = time_s
        self.hypotheses = [
            h for h in self.hypotheses if h.id == keep_id or time_s - h.last_seen_s <= self.forget_s
        ]

    def _nearest(self, center: np.ndarray) -> GateHypothesis | None:
        best, best_distance = None, MATCH_RADIUS_M
        for hypothesis in self.hypotheses:
            distance = float(np.linalg.norm(hypothesis.center - center))
            if distance <= best_distance:
                best, best_distance = hypothesis, distance
        return best

    def get(self, hypothesis_id: int | None) -> GateHypothesis | None:
        return next((h for h in self.hypotheses if h.id == hypothesis_id), None)
