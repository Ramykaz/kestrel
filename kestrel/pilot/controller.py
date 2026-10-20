"""Low level flight control shared by every pilot: a smoothed reference and DSLPIDControl."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.utils.enums import DroneModel

MAX_ACCEL_M_S2 = 2.5
# Speed kept when arriving at a point the drone must turn at, such as a gate's approach point.
ARRIVAL_SPEED_M_S = 0.6
MAX_YAW_RATE_RAD_S = 2.0


@dataclass(frozen=True)
class OwnState:
    """What the drone knows about itself, as if from visual inertial odometry."""

    pos: np.ndarray
    quat: np.ndarray
    """x, y, z, w"""
    vel: np.ndarray
    ang_vel: np.ndarray


class ReferenceTrajectory:
    """A point the drone tracks, moved toward the goal with limited acceleration.

    Handing DSLPIDControl a far-away goal directly asks for more tilt than the CF2X can hold
    altitude at, and the drone falls out of the sky.
    """

    def __init__(self, start: np.ndarray) -> None:
        self.pos = start.astype(float).copy()
        self.vel = np.zeros(3)

    def distance_to(self, goal: np.ndarray) -> float:
        return float(np.linalg.norm(goal - self.pos))

    def advance(self, goal: np.ndarray, speed: float, arrive: bool, dt: float) -> None:
        """Move one step toward goal at up to speed. With arrive, slow down to
        ARRIVAL_SPEED_M_S by the time the goal is reached."""
        distance = self.distance_to(goal)
        direction = (goal - self.pos) / max(distance, 1e-6)
        if arrive:
            speed = min(speed, ARRIVAL_SPEED_M_S + math.sqrt(2 * MAX_ACCEL_M_S2 * distance))
        change = direction * speed - self.vel
        max_change = MAX_ACCEL_M_S2 * dt
        norm = float(np.linalg.norm(change))
        if norm > max_change:
            change *= max_change / norm
        self.vel += change
        self.pos += self.vel * dt

    def brake(self, dt: float) -> None:
        """Slow toward a stop with limited deceleration."""
        speed = float(np.linalg.norm(self.vel))
        self.vel *= max(0.0, speed - MAX_ACCEL_M_S2 * dt) / max(speed, 1e-9)
        self.pos += self.vel * dt


def turn_toward(current: float, desired: float, dt: float) -> float:
    """Yaw angle moved toward desired at no more than MAX_YAW_RATE_RAD_S."""
    error = math.remainder(desired - current, math.tau)
    step = MAX_YAW_RATE_RAD_S * dt
    return current + max(-step, min(step, error))


class LowLevelController:
    def __init__(self) -> None:
        self._pid = DSLPIDControl(drone_model=DroneModel.CF2X)

    def rpm(
        self, own: OwnState, reference: ReferenceTrajectory, yaw: float, dt: float
    ) -> np.ndarray:
        rpm, _, _ = self._pid.computeControl(
            control_timestep=dt,
            cur_pos=own.pos,
            cur_quat=own.quat,
            cur_vel=own.vel,
            cur_ang_vel=own.ang_vel,
            target_pos=reference.pos,
            target_rpy=np.array([0.0, 0.0, yaw]),
            target_vel=reference.vel,
        )
        return rpm
