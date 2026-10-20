# Kestrel

A camera-only racing drone that a team of AI agents teaches to fly, in simulation, with no human labeling.

The drone has one forward camera and its own motion estimate. It does not know where the gates are. Agents generate synthetic training data, train a gate detector, tune the flight controller, review crash footage, and decide what to try next. Every claim an agent makes is checked by code.

Status: phase 0 (setup and measurement).

## Rules of the world

These rules define what camera-only means here.

| The pilot knows | The pilot does not know |
| :- | :- |
| Its own position, velocity, attitude (as if from visual inertial odometry) | Where any gate is |
| Onboard camera frames, 320x240, 30 Hz | How many gates there are |
| The physical size of a gate (0.60 m inner square, 0.80 m outer) | The course layout or seed |
| | Which objects are decoys |

The simulator knows everything and uses it only for labels, scoring, and crash cause ground truth. Only `kestrel/sim/oracle.py` may read gate ground truth, and a test asserts that nothing under `kestrel/pilot/` imports it.

## Development setup

PyBullet has no Windows wheels, so development runs on Linux (WSL Ubuntu on Windows), the same as Kaggle.

```sh
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync
uv run kestrel sim smoke
uv run pytest
uv run ruff check .
```

When the repo lives on the Windows drive and is used from WSL, keep the virtual environment on the Linux side for speed:

```sh
export UV_PROJECT_ENVIRONMENT=$HOME/.venvs/kestrel
```

## Commands

| Command | What it does |
| :- | :- |
| `kestrel sim smoke` | Hovers the drone at 1 m for 5 s and renders the onboard camera. Fails if the hover error is above 0.10 m |
| `kestrel sim bench --workers N [--egl]` | Measures onboard render throughput in frames per second |
