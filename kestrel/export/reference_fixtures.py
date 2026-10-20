"""Development fixtures for the website: reference autopilot flights on the eval suite.

These are real simulator flights, but the pilot steers by perturbed ground truth, so they say
nothing about the project's results. They are written outside web/public and never deployed.
Each fixture round uses different autopilot settings, and rounds are marked promoted or
rejected with the same headline rule the real round loop uses.
"""

from __future__ import annotations

import multiprocessing as mp
from dataclasses import dataclass
from pathlib import Path

from kestrel.eval.metrics import flight_metrics, round_statuses
from kestrel.export.web_bundle import (
    BundleRound,
    encode_flight,
    flight_file_name,
    flight_summary,
    write_course_files,
    write_rounds_json,
)
from kestrel.pilot.flight_log import FlightLog
from kestrel.sim.course import Course, eval_suite_seeds, generate_course, load_preset
from kestrel.sim.reference_pilot import ReferencePilotParams, fly_reference

FIXTURE_ROUNDS = [
    ReferencePilotParams(speed_m_s=1.0, gate_position_noise_m=0.30),
    ReferencePilotParams(speed_m_s=1.2, gate_position_noise_m=0.22),
    ReferencePilotParams(speed_m_s=1.5, gate_position_noise_m=0.25),
    ReferencePilotParams(speed_m_s=1.5, gate_position_noise_m=0.12),
    ReferencePilotParams(speed_m_s=2.0, gate_position_noise_m=0.10),
    ReferencePilotParams(speed_m_s=1.5, gate_position_noise_m=0.05),
    ReferencePilotParams(speed_m_s=1.8, gate_position_noise_m=0.03),
    ReferencePilotParams(speed_m_s=1.5, gate_position_noise_m=0.0),
]


@dataclass(frozen=True)
class _FlightJob:
    round_index: int
    course: Course
    params: ReferencePilotParams


def _fly(job: _FlightJob) -> tuple[_FlightJob, FlightLog]:
    seed = job.round_index * 1_000_003 + job.course.seed
    return job, fly_reference(job.course, job.params, seed=seed)


def write_reference_fixtures(out_dir: Path, workers: int) -> list[BundleRound]:
    seeds = eval_suite_seeds()
    courses = {
        preset: [generate_course(seed, load_preset(preset)) for seed in seeds[preset]]
        for preset in ("tech", "night")
    }
    all_courses = courses["tech"] + courses["night"]
    jobs = [
        _FlightJob(index, course, params)
        for index, params in enumerate(FIXTURE_ROUNDS)
        for course in all_courses
    ]
    with mp.get_context("spawn").Pool(workers) as pool:
        flown = pool.map(_fly, jobs)

    (out_dir / "flights").mkdir(parents=True, exist_ok=True)
    write_course_files(out_dir, all_courses)
    metrics_by_round, summaries_by_round = [], []
    for index in range(len(FIXTURE_ROUNDS)):
        logs = {job.course.seed: log for job, log in flown if job.round_index == index}
        summaries = {}
        for course in all_courses:
            log = logs[course.seed]
            name = flight_file_name(index, course.seed)
            (out_dir / name).write_bytes(encode_flight(log, len(course.gates), "ground_truth"))
            summaries[course.seed] = flight_summary(log, len(course.gates), name)
        summaries_by_round.append(summaries)
        metrics_by_round.append(
            flight_metrics(
                [logs[c.seed] for c in courses["tech"]], [logs[c.seed] for c in courses["night"]]
            )
        )
    rounds = []
    statuses = round_statuses(metrics_by_round)
    for index, params in enumerate(FIXTURE_ROUNDS):
        label = (
            f"Reference autopilot at {params.speed_m_s:.1f} m/s, "
            f"gate position error {params.gate_position_noise_m:.2f} m"
        )
        rounds.append(
            BundleRound(
                index, statuses[index], label, metrics_by_round[index], summaries_by_round[index]
            )
        )
    write_rounds_json(out_dir, "reference_fixtures", rounds)
    return rounds
