import numpy as np
import pytest

from kestrel.eval.crash_cause import crash_cause
from kestrel.pilot.flight_log import FlightEnd, load_flight_log, save_flight_log
from kestrel.sim.course import generate_course, load_preset
from kestrel.sim.reference_pilot import ReferencePilotParams, fly_reference


def test_reference_pilot_finishes_straight_courses():
    preset = load_preset("straight")
    for seed in range(3):
        log = fly_reference(generate_course(seed, preset), ReferencePilotParams(speed_m_s=1.5))
        assert log.end is FlightEnd.FINISHED, (seed, log.end, log.crash_body_kind)
        assert log.gates_passed == 4
        assert log.lap_time_s is not None and log.lap_time_s < 60


def test_large_gate_error_crashes_into_the_course():
    log = fly_reference(
        generate_course(3, load_preset("tech")),
        ReferencePilotParams(speed_m_s=1.5, gate_position_noise_m=0.6),
        seed=1,
    )
    assert log.end is FlightEnd.CRASH
    assert log.crash_body_kind in {"gate_frame", "gate_post", "ground"}


def test_crash_from_gate_error_is_labeled_bad_pose_and_survives_save(tmp_path):
    course = generate_course(3, load_preset("tech"))
    log = fly_reference(
        course,
        ReferencePilotParams(speed_m_s=1.5, gate_position_noise_m=0.6),
        seed=1,
        keep_crash_frames=True,
    )
    label = crash_cause(log, course)
    assert label.cause == "bad_pose", label
    assert label.pose_error_m is not None and label.pose_error_m > 0.5
    assert len(log.crash_frames) == 60

    save_flight_log(log, tmp_path / "flight.npz")
    loaded = load_flight_log(tmp_path / "flight.npz")
    assert loaded.end is log.end and loaded.crash_gate_index == log.crash_gate_index
    assert loaded.gate_pass_times == log.gate_pass_times
    np.testing.assert_allclose(np.array(loaded.positions), np.array(log.positions))
    np.testing.assert_allclose(np.array(loaded.believed_gates), np.array(log.believed_gates))
    assert [len(f) for f in loaded.detections] == [len(f) for f in log.detections]
    first = next(d for frame in log.detections for d in frame)
    again = next(d for frame in loaded.detections for d in frame)
    np.testing.assert_allclose(again.keypoints, first.keypoints, atol=1e-3)
    assert crash_cause(loaded, course) == label


def test_crash_cause_refuses_finished_flights():
    course = generate_course(0, load_preset("straight"))
    log = fly_reference(course, ReferencePilotParams(speed_m_s=1.5))
    with pytest.raises(ValueError, match="not in a crash"):
        crash_cause(log, course)
