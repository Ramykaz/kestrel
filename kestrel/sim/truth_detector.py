"""A stand-in detector that reports the true gate corners, for diagnosing the pilot.

Flying the camera pilot with it separates control and state machine problems from detector
problems. Its flights are never results; the real pilot only ever gets the ONNX detector.
"""

from __future__ import annotations

import numpy as np

from kestrel.data.labels import compute_gate_labels
from kestrel.perception.detect import Detection
from kestrel.sim.camera import OnboardCamera
from kestrel.sim.oracle import Oracle
from kestrel.sim.world import World


class TruthDetector:
    def __init__(self, world: World, oracle: Oracle, camera: OnboardCamera) -> None:
        self.world = world
        self.oracle = oracle
        self.camera = camera

    def detect(self, rgb: np.ndarray, conf_thresh: float) -> list[Detection]:
        """True labels for the drone's current pose; the frame itself is ignored."""
        drone = self.world.drone_state()
        view = self.camera.view_matrix(drone.pos, drone.quat)
        labels = compute_gate_labels(
            self.oracle, self.world.client, drone.pos, view, self.camera.intrinsics
        )
        return [
            Detection(label.class_id, 1.0, label.box_xyxy, label.keypoints, label.visibility)
            for label in labels
        ]
