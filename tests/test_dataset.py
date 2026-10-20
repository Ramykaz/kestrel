import json

import numpy as np
import pytest
import yaml

from kestrel.data.generate import plan_jobs, write_dataset
from kestrel.data.labels import read_yolo_label
from kestrel.data.randomize import load_dr_preset
from kestrel.sim.camera import HEIGHT, WIDTH
from kestrel.sim.course import eval_suite_seeds


def test_jobs_never_use_eval_seeds_and_split_by_course():
    dr = load_dr_preset("base")
    jobs = plan_jobs(dr, 1000, seed=3)
    held_out = {s for seeds in eval_suite_seeds().values() for s in seeds}
    assert not {j.course_seed for j in jobs} & held_out
    assert sum(j.frames for j in jobs) == 1000
    assert {j.split for j in jobs} == {"train", "val"}


def test_small_dataset_round_trip(tmp_path):
    out = tmp_path / "ds"
    manifest = write_dataset(out, load_dr_preset("base"), 20, seed=5, workers=1)
    data = yaml.safe_load((out / "data.yaml").read_text())
    assert data["kpt_shape"] == [4, 3] and data["names"] == ["gate", "gate_back"]
    assert data["flip_idx"] == [1, 0, 3, 2]
    images = sorted((out / "images").glob("*/*.jpg"))
    assert len(images) == 20
    assert manifest["counts"]["frames"]["train"] + manifest["counts"]["frames"]["val"] == 20
    assert json.loads((out / "manifest.json").read_text())["git_commit"] == manifest["git_commit"]
    labels = 0
    for image in images:
        for label in read_yolo_label(
            out / "labels" / image.parent.name / f"{image.stem}.txt", WIDTH, HEIGHT
        ):
            labels += 1
            assert label.class_id in (0, 1)
            assert np.all(label.box_xyxy[:2] <= label.box_xyxy[2:])
    assert labels == sum(manifest["counts"]["labels"].values())
    with pytest.raises(FileExistsError):
        write_dataset(out, load_dr_preset("base"), 20, seed=5, workers=1)
