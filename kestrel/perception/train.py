"""Detector training: presets from configs/train and a thin wrapper around ultralytics.

Every value in a preset must sit inside the hard limits below; SPEC 8.1 caps a round at
25 GPU minutes and imgsz at 288 to 416.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import yaml

from kestrel.paths import CONFIG_DIR

MAX_GPU_MINUTES = 25.0
IMGSZ_RANGE = (288, 416)
EPOCH_RANGE = (1, 300)
BATCH_RANGE = (8, 128)
# The first 10 modules of YOLO11n-pose are its backbone.
MAX_FREEZE = 10
AUGMENT_LIMITS = {
    "hsv_h": (0.0, 0.1),
    "hsv_s": (0.0, 0.9),
    "hsv_v": (0.0, 0.9),
    "degrees": (0.0, 30.0),
    "translate": (0.0, 0.3),
    "scale": (0.0, 0.9),
    "fliplr": (0.0, 0.5),
    "mosaic": (0.0, 1.0),
}


@dataclass(frozen=True)
class TrainConfig:
    name: str
    base_weights: str
    """An ultralytics model name such as yolo11n-pose.pt, or a path to earlier weights."""
    epochs: int
    gpu_minutes: float
    imgsz: int
    batch: int
    freeze: int
    augment: dict[str, float]

    def to_dict(self) -> dict:
        return asdict(self)


def _in_range(name: str, value: float, bounds: tuple[float, float]) -> None:
    if not bounds[0] <= value <= bounds[1]:
        raise ValueError(f"{name} {value} is outside the hard limits {list(bounds)}")


def parse_train_config(name: str, raw: dict) -> TrainConfig:
    expected = set(TrainConfig.__dataclass_fields__) - {"name"}
    if set(raw) != expected:
        missing, extra = sorted(expected - set(raw)), sorted(set(raw) - expected)
        raise ValueError(f"train preset {name!r}: missing {missing}, unexpected {extra}")
    _in_range("epochs", raw["epochs"], EPOCH_RANGE)
    _in_range("gpu_minutes", raw["gpu_minutes"], (1.0, MAX_GPU_MINUTES))
    _in_range("imgsz", raw["imgsz"], IMGSZ_RANGE)
    if raw["imgsz"] % 32:
        raise ValueError(f"imgsz {raw['imgsz']} must be a multiple of 32")
    _in_range("batch", raw["batch"], BATCH_RANGE)
    _in_range("freeze", raw["freeze"], (0, MAX_FREEZE))
    unknown = sorted(set(raw["augment"]) - set(AUGMENT_LIMITS))
    if unknown:
        raise ValueError(f"unknown augmentations {unknown}; allowed {sorted(AUGMENT_LIMITS)}")
    for key, value in raw["augment"].items():
        _in_range(f"augment.{key}", value, AUGMENT_LIMITS[key])
    return TrainConfig(
        name=name,
        base_weights=str(raw["base_weights"]),
        epochs=int(raw["epochs"]),
        gpu_minutes=float(raw["gpu_minutes"]),
        imgsz=int(raw["imgsz"]),
        batch=int(raw["batch"]),
        freeze=int(raw["freeze"]),
        augment={k: float(v) for k, v in raw["augment"].items()},
    )


def load_train_preset(name: str) -> TrainConfig:
    path = CONFIG_DIR / "train" / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"no train preset named {name!r} at {path}")
    return parse_train_config(name, yaml.safe_load(path.read_text()))


@dataclass(frozen=True)
class TrainResult:
    weights: Path
    onnx: Path
    epochs_run: int
    minutes: float
    metrics: dict[str, float]


def train_detector(
    dataset: Path, out: Path, config: TrainConfig, device: str, seed: int = 0
) -> TrainResult:
    """Train on a dataset written by kestrel gen, export out/detector.onnx and write
    out/train_report.json. Training stops after config.epochs, or at the end of the first
    epoch that finishes past config.gpu_minutes."""
    # Notebook kernels export an inline matplotlib backend that this environment does not
    # have, and ultralytics imports matplotlib while checking the dataset. Nothing is shown.
    os.environ["MPLBACKEND"] = "Agg"
    from ultralytics import YOLO

    from kestrel.export.onnx_export import export_input_size, export_onnx

    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"{out} already holds files; training runs are never overwritten")
    manifest = json.loads((dataset / "manifest.json").read_text())
    started = time.monotonic()
    model = YOLO(config.base_weights)

    # Ultralytics' own time argument re-plans the epoch count to fill the time, which would
    # ignore config.epochs, so the budget is enforced here as a cap instead.
    def stop_at_budget(trainer) -> None:
        if time.monotonic() - started > config.gpu_minutes * 60:
            trainer.stop = True

    model.add_callback("on_fit_epoch_end", stop_at_budget)
    model.train(
        data=str(dataset / "data.yaml"),
        epochs=config.epochs,
        imgsz=config.imgsz,
        batch=config.batch,
        freeze=config.freeze,
        device=device,
        seed=seed,
        deterministic=False,
        project=str(out.resolve()),
        name="run",
        exist_ok=False,
        plots=False,
        verbose=False,
        **config.augment,
    )
    minutes = (time.monotonic() - started) / 60
    trainer = model.trainer
    metrics = {k: round(float(v), 5) for k, v in trainer.metrics.items()}
    onnx = export_onnx(Path(trainer.best), out / "detector.onnx", config.imgsz)
    result = TrainResult(Path(trainer.best), onnx, int(trainer.epoch) + 1, minutes, metrics)
    report = {
        "config": config.to_dict(),
        "dataset": str(dataset),
        "dataset_commit": manifest["git_commit"],
        "dataset_counts": manifest["counts"],
        "device": device,
        "seed": seed,
        "epochs_run": result.epochs_run,
        "minutes": round(minutes, 2),
        "metrics": metrics,
        "weights": str(result.weights),
        "onnx": str(onnx),
        "onnx_input_hw": list(export_input_size(config.imgsz)),
    }
    (out / "train_report.json").write_text(json.dumps(report, indent=2) + "\n")
    return result
