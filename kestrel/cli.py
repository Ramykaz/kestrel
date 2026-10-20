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
