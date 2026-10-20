from kestrel.pilot.flight_log import FlightEnd
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
