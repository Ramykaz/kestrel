import json

import pytest

from kestrel.eval.metrics import FlightMetrics, flight_metrics, round_statuses
from kestrel.export.rounds import round_dirs, write_round_bundle
from kestrel.pilot.flight_log import save_flight_log
from kestrel.sim.course import generate_course, load_preset
from kestrel.sim.reference_pilot import ReferencePilotParams, fly_reference


def _write_round(root, index, label="test round", courses="suite", noise=0.0):
    course = generate_course(900001, load_preset("tech"))
    log = fly_reference(course, ReferencePilotParams(speed_m_s=1.5, gate_position_noise_m=noise))
    path = root / f"r{index:02d}"
    (path / "flights").mkdir(parents=True)
    save_flight_log(log, path / "flights/tech-s900001.npz")
    crash = {"cause": "bad_pose"} if log.end.name == "CRASH" else None
    summary = {
        "courses": courses,
        "label": label,
        "metrics": flight_metrics([log], []).to_dict(),
        "flights": [{"seed": 900001, "file": "flights/tech-s900001.npz", "crash": crash}],
    }
    (path / "summary.json").write_text(json.dumps(summary))
    return log


def test_round_statuses_follow_the_headline_rule():
    def metrics(completion, gates):
        return FlightMetrics(completion, gates, None, 0.0, None)

    history = [metrics(0.1, 2), metrics(0.1, 3), metrics(0.05, 5), metrics(0.1, 3), metrics(0.2, 1)]
    assert round_statuses(history) == ["baseline", "promoted", "rejected", "rejected", "promoted"]


def test_round_directories_must_be_contiguous_labeled_suites(tmp_path):
    _write_round(tmp_path, 0)
    _write_round(tmp_path, 2)
    with pytest.raises(ValueError, match="r00, r01"):
        round_dirs(tmp_path)
    other = tmp_path / "other"
    _write_round(other, 0, label="")
    with pytest.raises(ValueError, match="label"):
        round_dirs(other)


def test_bundle_has_flights_crash_causes_and_statuses(tmp_path):
    root = tmp_path / "rounds"
    _write_round(root, 0, noise=0.6)
    _write_round(root, 1)
    out = tmp_path / "bundle"
    (out / "flights").mkdir(parents=True)
    (out / "flights" / "r07-s1.bin").write_bytes(b"stale")
    rounds = write_round_bundle(out, root)
    assert [r.status for r in rounds] == ["baseline", "promoted"]
    assert sorted(p.name for p in (out / "flights").iterdir()) == [
        "r00-s900001.bin",
        "r01-s900001.bin",
    ]
    assert rounds[0].flights[900001]["crash_cause"] == "bad_pose"
    assert rounds[1].flights[900001]["crash_cause"] is None
    assert len(list((out / "courses").glob("*.json"))) == 30
