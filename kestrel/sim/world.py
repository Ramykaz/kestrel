"""PyBullet world: client, ground, the CF2X drone and its motor physics."""

from __future__ import annotations

import importlib.util
import logging
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from importlib.resources import files

import numpy as np
import pybullet as p
import pybullet_data

log = logging.getLogger(__name__)

PHYSICS_HZ = 240
CONTROL_HZ = 48
CAMERA_HZ = 30
GRAVITY = 9.8

CF2X_URDF = str(files("gym_pybullet_drones") / "assets" / "cf2x.urdf")


@dataclass(frozen=True)
class DroneState:
    pos: np.ndarray
    quat: np.ndarray
    vel: np.ndarray
    ang_vel: np.ndarray


def _cf2x_motor_constants() -> tuple[float, float]:
    properties = ET.parse(CF2X_URDF).getroot().find("properties")
    if properties is None:
        raise ValueError(f"no <properties> element in {CF2X_URDF}")
    return float(properties.attrib["kf"]), float(properties.attrib["km"])


class World:
    """One headless PyBullet client stepped at PHYSICS_HZ with the CF2X drone."""

    def __init__(self, use_egl: bool = False) -> None:
        self.client = p.connect(p.DIRECT)
        self.renderer = p.ER_TINY_RENDERER
        if use_egl:
            self._try_load_egl()
        self.kf, self.km = _cf2x_motor_constants()
        p.setGravity(0, 0, -GRAVITY, physicsClientId=self.client)
        p.setTimeStep(1.0 / PHYSICS_HZ, physicsClientId=self.client)
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.client)
        self.ground_id = p.loadURDF("plane.urdf", physicsClientId=self.client)
        self.drone_id = p.loadURDF(
            CF2X_URDF,
            [0.0, 0.0, 0.05],
            [0.0, 0.0, 0.0, 1.0],
            flags=p.URDF_USE_INERTIA_FROM_FILE,
            physicsClientId=self.client,
        )
        # Rendering from inside the drone mesh costs TinyRenderer about 4x per frame,
        # and no PyBullet camera ever needs to show the drone.
        for link in range(-1, p.getNumJoints(self.drone_id, physicsClientId=self.client)):
            p.changeVisualShape(
                self.drone_id, link, rgbaColor=[0, 0, 0, 0], physicsClientId=self.client
            )
        self.step_count = 0

    def _try_load_egl(self) -> None:
        spec = importlib.util.find_spec("eglRenderer")
        if spec is None or spec.origin is None:
            log.warning("EGL requested but the eglRenderer module is missing; using TinyRenderer")
            return
        plugin_id = p.loadPlugin(spec.origin, "_eglRendererPlugin", physicsClientId=self.client)
        if plugin_id < 0:
            log.warning("EGL plugin failed to load; using TinyRenderer")
            return
        self.renderer = p.ER_BULLET_HARDWARE_OPENGL

    @property
    def uses_egl(self) -> bool:
        return self.renderer == p.ER_BULLET_HARDWARE_OPENGL

    @property
    def time_s(self) -> float:
        return self.step_count / PHYSICS_HZ

    def reset_drone(self, pos: np.ndarray, yaw: float = 0.0) -> None:
        quat = p.getQuaternionFromEuler([0.0, 0.0, yaw])
        p.resetBasePositionAndOrientation(self.drone_id, pos.tolist(), quat, self.client)
        p.resetBaseVelocity(self.drone_id, [0, 0, 0], [0, 0, 0], self.client)

    def drone_state(self) -> DroneState:
        pos, quat = p.getBasePositionAndOrientation(self.drone_id, physicsClientId=self.client)
        vel, ang_vel = p.getBaseVelocity(self.drone_id, physicsClientId=self.client)
        return DroneState(np.array(pos), np.array(quat), np.array(vel), np.array(ang_vel))

    def step(self, rpm: np.ndarray) -> None:
        """Apply motor thrust and yaw torque, then advance one physics step.

        Same force model as gym-pybullet-drones Physics.PYB for the CF2X.
        """
        forces = rpm**2 * self.kf
        torques = rpm**2 * self.km
        z_torque = -torques[0] + torques[1] - torques[2] + torques[3]
        for motor in range(4):
            p.applyExternalForce(
                self.drone_id,
                motor,
                forceObj=[0, 0, float(forces[motor])],
                posObj=[0, 0, 0],
                flags=p.LINK_FRAME,
                physicsClientId=self.client,
            )
        p.applyExternalTorque(
            self.drone_id,
            4,
            torqueObj=[0, 0, float(z_torque)],
            flags=p.LINK_FRAME,
            physicsClientId=self.client,
        )
        p.stepSimulation(physicsClientId=self.client)
        self.step_count += 1

    def close(self) -> None:
        p.disconnect(self.client)
