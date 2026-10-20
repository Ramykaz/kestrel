"""Evaluated rounds into the web bundle.

A round is a directory artifacts/rounds/rNN written by `kestrel fly --courses suite`.
Round numbers must run from r00 without gaps, so the site's scrubber matches the history.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from kestrel.eval.metrics import FlightMetrics, round_statuses
from kestrel.eval.suite import eval_courses
from kestrel.export.web_bundle import (
    BundleRound,
    encode_flight,
    flight_file_name,
    flight_summary,
    write_course_files,
)
from kestrel.pilot.flight_log import load_flight_log

ROUND_DIR = re.compile(r"r(\d{2})")


def round_dirs(rounds_root: Path) -> list[Path]:
    """Round directories in order. Raises on gaps or on results that are not a full suite."""
    found = sorted(
        (int(m.group(1)), path)
        for path in rounds_root.glob("r*")
        if path.is_dir() and (m := ROUND_DIR.fullmatch(path.name))
    )
    indices = [index for index, _ in found]
    if indices != list(range(len(found))):
        raise ValueError(
            f"round directories in {rounds_root} must run r00, r01, ...; found {indices}"
        )
    for _, path in found:
        summary = json.loads((path / "summary.json").read_text())
        if summary["courses"] != "suite":
            raise ValueError(f"{path} holds {summary['courses']} flights, not the eval suite")
        if not summary.get("label"):
            raise ValueError(f"{path}/summary.json has no label; pass --label to kestrel fly")
    return [path for _, path in found]


def write_round_bundle(out_dir: Path, rounds_root: Path) -> list[BundleRound]:
    """Encode every round's flights into out_dir/flights and return the rounds for
    rounds.json. Stale flight files from an earlier export are removed first."""
    courses = {course.seed: course for course in eval_courses()}
    write_course_files(out_dir, list(courses.values()))
    flights_dir = out_dir / "flights"
    flights_dir.mkdir(parents=True, exist_ok=True)
    for stale in flights_dir.glob("*.bin"):
        stale.unlink()

    summaries_by_round, metrics_by_round, labels = [], [], []
    for index, path in enumerate(round_dirs(rounds_root)):
        summary = json.loads((path / "summary.json").read_text())
        flights = {}
        for record in summary["flights"]:
            course = courses[record["seed"]]
            log = load_flight_log(path / record["file"])
            name = flight_file_name(index, course.seed)
            (out_dir / name).write_bytes(encode_flight(log, len(course.gates), "detector"))
            flights[course.seed] = flight_summary(log, len(course.gates), name) | {
                "crash_cause": record["crash"]["cause"] if record["crash"] else None
            }
        summaries_by_round.append(flights)
        metrics_by_round.append(FlightMetrics(**summary["metrics"]))
        labels.append(summary["label"])
    statuses = round_statuses(metrics_by_round)
    return [
        BundleRound(i, statuses[i], labels[i], metrics_by_round[i], summaries_by_round[i])
        for i in range(len(labels))
    ]
