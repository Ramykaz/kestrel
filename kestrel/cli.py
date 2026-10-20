"""Command line entry point."""

from __future__ import annotations

import logging
import os
from pathlib import Path

import typer
from PIL import Image

app = typer.Typer(no_args_is_help=True)
sim_app = typer.Typer(no_args_is_help=True, help="Simulator checks and measurements.")
app.add_typer(sim_app, name="sim")
data_app = typer.Typer(no_args_is_help=True, help="Synthetic data generation and QA.")
app.add_typer(data_app, name="data")


@app.callback()
def main(verbose: bool = typer.Option(False, "--verbose", "-v")) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO, format="%(levelname)s %(message)s"
    )


@sim_app.command("smoke")
def sim_smoke(
    seconds: float = typer.Option(5.0, help="Simulated hover time in seconds."),
    egl: bool = typer.Option(False, help="Try the EGL renderer, fall back to TinyRenderer."),
    save_frame: Path | None = typer.Option(None, help="Write the last onboard frame as PNG."),
) -> None:
    """Hover the drone at 1 m and render the onboard camera."""
    from kestrel.sim.smoke import HOVER_TOLERANCE_M, run_hover

    result = run_hover(sim_seconds=seconds, use_egl=egl)
    typer.echo(f"renderer: {result.renderer}")
    typer.echo(f"simulated {result.sim_seconds:.1f} s in {result.wall_seconds:.2f} s wall")
    typer.echo(f"rendered {result.frames_rendered} frames at {result.render_fps:.1f} frames/s")
    typer.echo(f"hover error after {result.sim_seconds:.1f} s: {result.final_error_m:.3f} m")
    if save_frame is not None:
        Image.fromarray(result.last_frame).save(save_frame)
        typer.echo(f"wrote {save_frame}")
    if not result.passed:
        typer.echo(f"FAIL: hover error above {HOVER_TOLERANCE_M} m", err=True)
        raise typer.Exit(code=1)
    typer.echo("PASS")


@sim_app.command("bench")
def sim_bench(
    frames: int = typer.Option(300, help="Frames rendered per worker."),
    workers: int = typer.Option(1, help="Worker processes, one PyBullet client each."),
    egl: bool = typer.Option(False, help="Try the EGL renderer, fall back to TinyRenderer."),
) -> None:
    """Measure onboard camera render throughput in frames per second."""
    from kestrel.sim.smoke import run_render_bench

    result = run_render_bench(frames_per_worker=frames, workers=workers, use_egl=egl)
    typer.echo(
        f"renderer {result.renderer}, {result.workers} workers of {os.cpu_count()} CPUs: "
        f"{result.frames} frames in {result.wall_seconds:.2f} s, {result.fps:.1f} frames/s"
    )


@app.command("gen")
def gen(
    out: Path = typer.Option(..., help="New dataset directory; must not exist or be empty."),
    frames: int = typer.Option(8000, help="Total frames, split into train and val by course."),
    dr: str = typer.Option("base", help="Domain randomization preset in configs/dr."),
    seed: int = typer.Option(0, help="Seed for courses, looks and camera poses."),
    workers: int = typer.Option(os.cpu_count() or 1, help="Worker processes."),
    egl: bool = typer.Option(False, help="Render with EGL when available."),
) -> None:
    """Generate a labeled YOLO pose dataset of randomized scatter frames."""
    from kestrel.data.generate import write_dataset
    from kestrel.data.randomize import load_dr_preset

    manifest = write_dataset(out, load_dr_preset(dr), frames, seed, workers, egl)
    counts = manifest["counts"]
    typer.echo(
        f"wrote {out}: {counts['frames']['train']} train and {counts['frames']['val']} val "
        f"frames from {counts['courses']} courses in {manifest['render_seconds']} s"
    )
    typer.echo(f"labels {counts['labels']}, frames without labels {counts['empty_frames']}")


@app.command("train")
def train(
    dataset: Path = typer.Option(..., help="Dataset directory written by kestrel gen."),
    out: Path = typer.Option(..., help="New directory for weights, ONNX model and report."),
    preset: str = typer.Option("base", help="Training preset in configs/train."),
    device: str = typer.Option("0", help="CUDA device index, or cpu."),
    seed: int = typer.Option(0, help="Training seed."),
) -> None:
    """Train the gate detector and export it to ONNX."""
    from kestrel.perception.train import load_train_preset, train_detector

    result = train_detector(dataset, out, load_train_preset(preset), device, seed)
    pose_map = result.metrics.get("metrics/mAP50-95(P)")
    typer.echo(f"trained {result.epochs_run} epochs in {result.minutes:.1f} min")
    typer.echo(f"pose mAP50-95 on val: {pose_map}")
    typer.echo(f"wrote {result.onnx} and {out / 'train_report.json'}")


@app.command("fly")
def fly(
    onnx: Path = typer.Option(..., help="Detector exported by kestrel train."),
    out: Path = typer.Option(..., help="New directory for flight logs and summary.json."),
    courses: str = typer.Option("suite", help="suite (30 held-out courses) or straight (10)."),
    pilot: str = typer.Option("base", help="Pilot preset in configs/pilot."),
    label: str = typer.Option("", help="One line describing the round, shown on the site."),
    workers: int = typer.Option(os.cpu_count() or 1, help="Worker processes."),
) -> None:
    """Fly the camera-only pilot and score it with the evaluator."""
    import hashlib
    import json

    from kestrel.eval.crash_cause import crash_cause
    from kestrel.eval.metrics import flight_metrics
    from kestrel.eval.suite import eval_courses, run_flights, straight_check_courses
    from kestrel.pilot.flight_log import FlightEnd, save_flight_log
    from kestrel.pilot.params import load_pilot_preset

    if courses not in ("suite", "straight"):
        raise typer.BadParameter(f"courses must be suite or straight, got {courses!r}")
    if out.exists() and any(out.iterdir()):
        raise typer.BadParameter(f"{out} already holds files; flight results are never overwritten")
    course_list = eval_courses() if courses == "suite" else straight_check_courses()
    params = load_pilot_preset(pilot)
    logs = run_flights(course_list, onnx, params, workers)

    (out / "flights").mkdir(parents=True, exist_ok=True)
    flights = []
    for course, log in zip(course_list, logs, strict=True):
        name = f"flights/{course.preset}-s{course.seed}.npz"
        save_flight_log(log, out / name)
        record = {
            "seed": course.seed,
            "preset": course.preset,
            "file": name,
            "end": None if log.end is None else log.end.name.lower(),
            "end_time_s": round(log.end_time or 0.0, 2),
            "gates_passed": log.gates_passed,
            "gate_count": len(course.gates),
            "lap_time_s": None if log.lap_time_s is None else round(log.lap_time_s, 2),
            "crash_body_kind": log.crash_body_kind,
            "crash": crash_cause(log, course).to_dict() if log.end is FlightEnd.CRASH else None,
        }
        flights.append(record)
        cause = record["crash"]["cause"] if record["crash"] else ""
        typer.echo(
            f"{course.preset:8s} {course.seed}  {record['end']:8s} "
            f"{log.gates_passed}/{len(course.gates)} gates  {cause}"
        )
    summary = {
        "courses": courses,
        "label": label,
        "onnx": str(onnx),
        "onnx_sha256": hashlib.sha256(onnx.read_bytes()).hexdigest(),
        "pilot": params.to_dict(),
        "flights": flights,
    }
    if courses == "suite":
        tech = [log for c, log in zip(course_list, logs, strict=True) if c.preset == "tech"]
        night = [log for c, log in zip(course_list, logs, strict=True) if c.preset == "night"]
        summary["metrics"] = flight_metrics(tech, night).to_dict()
        typer.echo(json.dumps(summary["metrics"]))
    finished = sum(f["end"] == "finished" for f in flights)
    summary["finished"] = finished
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    typer.echo(f"finished {finished} of {len(flights)}; wrote {out / 'summary.json'}")


@data_app.command("preview")
def data_preview(
    dataset: Path = typer.Argument(..., help="Dataset directory written by kestrel gen."),
    out: Path = typer.Option(Path("artifacts/contact_sheet.png"), help="Output image path."),
    frames: int = typer.Option(64, help="Number of frames on the sheet."),
    seed: int = typer.Option(0, help="Seed for choosing frames."),
) -> None:
    """Write a contact sheet of random frames from a dataset, labels drawn from its files."""
    from kestrel.data.labels import CLASS_GATE_BACK
    from kestrel.data.preview import dataset_contact_sheet

    sheet, labels = dataset_contact_sheet(dataset, frames, seed)
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out)
    gates = sum(len(frame) for frame in labels)
    backs = sum(label.class_id == CLASS_GATE_BACK for frame in labels for label in frame)
    empty = sum(not frame for frame in labels)
    typer.echo(f"wrote {out}: {len(labels)} frames, {gates} gate labels ({backs} gate_back)")
    typer.echo(f"frames without any gate label: {empty}")


@app.command("export-web")
def export_web(
    out: Path = typer.Option(Path("web/public/data"), help="Bundle directory."),
    reference_fixtures: bool = typer.Option(
        False, help="Write reference autopilot fixtures for development instead of rounds."
    ),
    workers: int = typer.Option(os.cpu_count() or 1, help="Worker processes for fixtures."),
    rounds_root: Path = typer.Option(Path("artifacts/rounds"), help="Evaluated round directories."),
) -> None:
    """Write the static data bundle the website reads."""
    from kestrel.export.web_bundle import check_bundle_size, flights_over_target

    out.mkdir(parents=True, exist_ok=True)
    if reference_fixtures:
        if out.resolve() == (Path("web/public/data")).resolve():
            raise typer.BadParameter("fixtures must not be written into the deployed bundle")
        from kestrel.export.reference_fixtures import write_reference_fixtures

        for r in write_reference_fixtures(out, workers):
            m = r.metrics
            typer.echo(
                f"fixture round {r.round_index} {r.status:8s} completion {m.completion_rate:.2f} "
                f"gates {m.gates_per_course:.2f} crash {m.crash_rate:.2f}  {r.label}"
            )
    else:
        from kestrel.export.rounds import write_round_bundle
        from kestrel.export.web_bundle import write_rounds_json

        rounds = write_round_bundle(out, rounds_root)
        write_rounds_json(out, "rounds", rounds)
        for r in rounds:
            m = r.metrics
            typer.echo(
                f"round {r.round_index} {r.status:8s} completion {m.completion_rate:.2f} "
                f"gates {m.gates_per_course:.2f} crash {m.crash_rate:.2f}  {r.label}"
            )
        if not rounds:
            typer.echo("no rounds have run yet: wrote eval suite courses and an empty rounds.json")
    files, total = check_bundle_size(out)
    typer.echo(f"{out}: {files} files, {total / 1024:.0f} KiB")
    for name, size in flights_over_target(out):
        typer.echo(f"over the 150 KiB flight target: {name} {size / 1024:.0f} KiB")
