"""The camera pilot in closed loop with a detector that reports true corners."""

import numpy as np
import pytest

from kestrel.pilot.flight_log import FlightEnd
from kestrel.pilot.fly import CameraPilot
from kestrel.pilot.params import load_pilot_preset
from kestrel.sim.course import generate_course, load_preset
from kestrel.sim.flight import fly_course
from kestrel.sim.truth_detector import TruthDetector


def _fly(preset: str, seed: int):
    params = load_pilot_preset("base")
    course = generate_course(seed, load_preset(preset))

    def make_pilot(world, oracle, camera, own):
        return CameraPilot(params, TruthDetector(world, oracle, camera), camera.intrinsics, own)

    return fly_course(course, make_pilot), course


@pytest.mark.parametrize(("preset", "seed"), [("straight", 1000), ("tech", 2003)])
def test_camera_pilot_finishes_with_perfect_detections(preset, seed):
    log, course = _fly(preset, seed)
    assert log.end is FlightEnd.FINISHED, (log.end, log.gates_passed, log.crash_body_kind)
    assert log.gates_passed == len(course.gates)


def test_pilot_flies_a_passed_gate_again_when_nothing_new_is_found():
    # On this course the pilot flies the last gate right after gate 0, out of order. After
    # gate 6 nothing unflown is left in view, so after one search turn it has to fly an
    # already passed gate again instead of searching until the timeout.
    log, _ = _fly("tech", 2008)
    assert log.gates_passed >= 7, (log.end, log.gates_passed)
    targets_after_gate_6 = [
        believed
        for time_s, believed in zip(log.times, log.believed_gates, strict=True)
        if time_s > log.gate_pass_times[6] and not np.any(np.isnan(believed))
    ]
    assert targets_after_gate_6
