import json

import numpy as np
import pytest

from kestrel.eval.metrics import flight_metrics
from kestrel.export.web_bundle import FLIGHT_MAGIC, check_bundle_size, encode_flight
from kestrel.pilot.flight_log import Detection, FlightEnd, FlightLog, PilotState


def _log(end: FlightEnd, gates: int, frames: int = 5) -> FlightLog:
    log = FlightLog(course_seed=900001, course_preset="tech", sample_hz=30.0)
    for i in range(frames):
        det = Detection(
            class_id=0,
            confidence=0.875,
            box_xyxy=np.array([10.0, 20.0, 110.0, 100.0]),
            keypoints=np.array([[30.0, 40.0], [90.0, 40.0], [90.0, 80.0], [30.0, 80.0]]),
            visibility=np.array([2, 2, 1, 0]),
        )
        log.record(
            i / 30,
            np.array([i * 0.1, 1.0, 1.5]),
            np.array([0.0, 0.0, 0.0, 1.0]),
            np.array([3.0, 1.0, 1.5]),
            PilotState.APPROACH,
            [det] if i % 2 == 0 else [],
        )
    log.gate_pass_times = [1.0 + i for i in range(gates)]
    log.end, log.end_time = end, 9.5
    return log


def _decode(blob: bytes):
    assert blob[:4] == FLIGHT_MAGIC
    header_len = int.from_bytes(blob[4:8], "little")
    header = json.loads(blob[8 : 8 + header_len])
    assert (8 + header_len) % 8 == 0
    offset = 8 + header_len
    arrays = {}
    for spec in header["arrays"]:
        dtype = np.dtype(spec["dtype"]).newbyteorder("<")
        count = int(np.prod(spec["shape"]))
        arrays[spec["name"]] = np.frombuffer(blob, dtype, count, offset).reshape(spec["shape"])
        offset += count * dtype.itemsize
    assert offset == len(blob)
    return header, arrays


def test_flight_file_round_trip():
    header, arrays = _decode(encode_flight(_log(FlightEnd.CRASH, 2), 8, "ground_truth"))
    assert header["frames"] == 5
    assert header["end"] == "crash"
    assert header["gates_passed"] == 2
    assert header["gate_count"] == 8
    assert arrays["position"].shape == (5, 3)
    assert np.allclose(arrays["position"][4], [0.4, 1.0, 1.5], atol=1e-3)
    assert list(arrays["det_count"]) == [1, 0, 1, 0, 1]
    row = arrays["detections"][0].astype(float)
    assert row[0] == 0 and row[1] == pytest.approx(0.875)
    assert row[2] == pytest.approx(10.5 / 320, abs=1e-3)
    packed = int(row[14])
    assert [(packed >> (2 * i)) & 3 for i in range(4)] == [2, 2, 1, 0]


def test_flight_metrics_follow_spec_definitions():
    tech = [_log(FlightEnd.FINISHED, 8), _log(FlightEnd.CRASH, 3), _log(FlightEnd.TIMEOUT, 1)]
    night = [_log(FlightEnd.FINISHED, 8), _log(FlightEnd.CRASH, 0)]
    metrics = flight_metrics(tech, night)
    assert metrics.completion_rate == pytest.approx(1 / 3)
    assert metrics.gates_per_course == pytest.approx(4.0)
    assert metrics.lap_time_s == pytest.approx(8.0)
    assert metrics.crash_rate == pytest.approx(1 / 3)
    assert metrics.night_completion == pytest.approx(0.5)


def test_bundle_size_check_rejects_oversized_files(tmp_path):
    (tmp_path / "big.bin").write_bytes(b"\0" * (8 * 1024 * 1024 + 1))
    with pytest.raises(ValueError, match="files above"):
        check_bundle_size(tmp_path)
