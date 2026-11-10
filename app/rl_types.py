from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class RocketKinematicState:
    position: np.ndarray
    target_position: np.ndarray
    linear_velocity: np.ndarray
    quaternion_wxyz: np.ndarray
    angular_velocity: np.ndarray
    linear_acceleration: np.ndarray
    altitude_to_ground: float
    main_thrust: float
    gimbal_x: float
    gimbal_y: float
    out_of_bounds: bool = False
    invalid: bool = False


@dataclass(frozen=True)
class TerrainSafetyState:
    local_slope_deg: float
    local_roughness_m: float
    safe_zone_score: float
    terrain_height_at_target: float
    slope_norm: float = 0.0
    roughness_norm: float = 0.0


@dataclass(frozen=True)
class ContactState:
    leg_contact: np.ndarray = field(default_factory=lambda: np.zeros(4, dtype=bool))
    leg_contact_force: np.ndarray = field(default_factory=lambda: np.zeros(4, dtype=np.float32))
    body_contact: bool = False

    @property
    def any_contact(self) -> bool:
        return bool(self.body_contact or np.any(self.leg_contact))

    @property
    def all_legs_contact(self) -> bool:
        return bool(np.all(self.leg_contact))

    @property
    def max_force(self) -> float:
        if self.leg_contact_force.size == 0:
            return 0.0
        return float(np.max(self.leg_contact_force))


@dataclass(frozen=True)
class EpisodeTimeState:
    step_count: int
    elapsed_s: float
    max_steps: int
    max_episode_s: float


@dataclass(frozen=True)
class RewardMemory:
    target_distance_xy: float


def normalize_quaternion_wxyz(quaternion: np.ndarray) -> np.ndarray:
    q = np.asarray(quaternion, dtype=np.float32)
    norm = float(np.linalg.norm(q))
    if not np.isfinite(norm) or norm < 1e-8:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    return q / norm


def target_delta(state: RocketKinematicState) -> np.ndarray:
    return np.asarray(state.target_position, dtype=np.float32) - np.asarray(state.position, dtype=np.float32)


def target_distance_xy(state: RocketKinematicState) -> float:
    delta = target_delta(state)
    return float(np.linalg.norm(delta[:2]))


def horizontal_speed(state: RocketKinematicState) -> float:
    velocity = np.asarray(state.linear_velocity, dtype=np.float32)
    return float(np.linalg.norm(velocity[:2]))


def vertical_speed(state: RocketKinematicState) -> float:
    return float(abs(np.asarray(state.linear_velocity, dtype=np.float32)[2]))


def angular_speed(state: RocketKinematicState) -> float:
    return float(np.linalg.norm(np.asarray(state.angular_velocity, dtype=np.float32)))


def rocket_up_vector_from_quaternion(quaternion_wxyz: np.ndarray) -> np.ndarray:
    q = normalize_quaternion_wxyz(quaternion_wxyz)
    w, x, y, z = [float(v) for v in q]
    return np.array(
        [
            2.0 * (x * z + w * y),
            2.0 * (y * z - w * x),
            1.0 - 2.0 * (x * x + y * y),
        ],
        dtype=np.float32,
    )


def up_alignment_from_quaternion(quaternion_wxyz: np.ndarray) -> float:
    up = rocket_up_vector_from_quaternion(quaternion_wxyz)
    return float(np.clip(up[2], -1.0, 1.0))


def tilt_angle_deg_from_quaternion(quaternion_wxyz: np.ndarray) -> float:
    alignment = up_alignment_from_quaternion(quaternion_wxyz)
    return float(np.degrees(np.arccos(np.clip(alignment, -1.0, 1.0))))
