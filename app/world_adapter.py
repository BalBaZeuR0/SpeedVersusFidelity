from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from app.rl_types import ContactState, RocketKinematicState, TerrainSafetyState
from app.world_builder import LunarLandingWorld


@dataclass
class AdapterMemory:
    previous_position: np.ndarray
    previous_velocity: np.ndarray
    previous_time_s: float
    previous_action: np.ndarray


class IsaacWorldAdapter:
    """Bridge LunarLandingWorld to the Gymnasium RL wrapper.

    This adapter intentionally keeps Isaac-specific sensor/contact details in
    small methods. Some values are currently estimated from terrain/pose until
    dedicated Isaac contact, LiDAR, and depth sensors are wired in.
    """

    def __init__(self, env: LunarLandingWorld, config: dict[str, Any] | None = None):
        self.env = env
        self.config = config or {}
        self.control_dt = float(self.config.get("control_dt", 0.05))
        self.max_lidar_distance = float(self.config.get("max_lidar_distance_m", 30.0))
        self.memory: AdapterMemory | None = None

    def reset(self) -> tuple[RocketKinematicState, dict[str, np.ndarray], TerrainSafetyState, ContactState]:
        self.env.reset()
        state = self._build_state(np.zeros(3, dtype=np.float32))
        self.memory = AdapterMemory(
            previous_position=state.position.copy(),
            previous_velocity=np.zeros(3, dtype=np.float32),
            previous_time_s=0.0,
            previous_action=np.zeros(3, dtype=np.float32),
        )
        sensors = self._build_sensor_dict(state)
        terrain = self._build_terrain_state(state)
        contact = self._estimate_contact(state)
        return state, sensors, terrain, contact

    def step(
        self, action: np.ndarray
    ) -> tuple[RocketKinematicState, dict[str, np.ndarray], TerrainSafetyState, ContactState]:
        action = np.asarray(action, dtype=np.float32)
        throttle = float(np.clip(action[0], 0.0, 1.0)) if action.size else 0.0
        gimbal_x = float(np.clip(action[1], -1.0, 1.0)) if action.size > 1 else 0.0
        gimbal_y = float(np.clip(action[2], -1.0, 1.0)) if action.size > 2 else 0.0
        self.env.step(throttle=throttle, gimbal_x=gimbal_x, gimbal_y=gimbal_y)
        state = self._build_state(action)
        sensors = self._build_sensor_dict(state)
        terrain = self._build_terrain_state(state)
        contact = self._estimate_contact(state)
        if self.memory is not None:
            self.memory.previous_position = state.position.copy()
            self.memory.previous_velocity = state.linear_velocity.copy()
            self.memory.previous_action = action.copy()
            self.memory.previous_time_s += self.control_dt
        return state, sensors, terrain, contact

    def _build_state(self, action: np.ndarray) -> RocketKinematicState:
        if self.env.state is None:
            raise RuntimeError("LunarLandingWorld must be reset before adapter state can be built")
        position = self._rocket_position()
        velocity = self._rocket_linear_velocity(position)
        acceleration = self._estimate_acceleration(velocity)
        quaternion = self._rocket_quaternion()
        angular_velocity = self._rocket_angular_velocity()
        target = self._target_position()
        altitude = self._altitude_to_ground(position)
        action = np.asarray(action, dtype=np.float32)
        return RocketKinematicState(
            position=position,
            target_position=target,
            linear_velocity=velocity,
            quaternion_wxyz=quaternion,
            angular_velocity=angular_velocity,
            linear_acceleration=acceleration,
            altitude_to_ground=altitude,
            main_thrust=float(np.clip(action[0], 0.0, 1.0)) if action.size else 0.0,
            gimbal_x=float(np.clip(action[1], -1.0, 1.0)) if action.size > 1 else 0.0,
            gimbal_y=float(np.clip(action[2], -1.0, 1.0)) if action.size > 2 else 0.0,
            out_of_bounds=self._is_out_of_bounds(position),
        )

    def _rocket_position(self) -> np.ndarray:
        assert self.env.state is not None
        pose = self.env.rocket.get_world_pose()
        if pose is not None:
            return pose[0]
        return np.asarray(self.env.state.rocket.position, dtype=np.float32)

    def _rocket_quaternion(self) -> np.ndarray:
        assert self.env.state is not None
        pose = self.env.rocket.get_world_pose()
        if pose is not None:
            return pose[1]
        euler = np.deg2rad(np.asarray(self.env.state.rocket.euler_deg, dtype=np.float32))
        roll, pitch, yaw = [float(v) for v in euler]
        cr, sr = np.cos(roll / 2.0), np.sin(roll / 2.0)
        cp, sp = np.cos(pitch / 2.0), np.sin(pitch / 2.0)
        cy, sy = np.cos(yaw / 2.0), np.sin(yaw / 2.0)
        return np.array(
            [
                cr * cp * cy + sr * sp * sy,
                sr * cp * cy - cr * sp * sy,
                cr * sp * cy + sr * cp * sy,
                cr * cp * sy - sr * sp * cy,
            ],
            dtype=np.float32,
        )

    def _target_position(self) -> np.ndarray:
        if self.env.state is None:
            return np.zeros(3, dtype=np.float32)
        x, y, _ = self.env.state.reset_sample.target_position
        z = self._terrain_height_at(float(x), float(y))
        return np.array([x, y, z], dtype=np.float32)

    def _rocket_linear_velocity(self, position: np.ndarray) -> np.ndarray:
        """Prefer the rigid body's true linear velocity; fall back to finite differences."""
        v = self.env.rocket.get_linear_velocity()
        if v is not None and v.shape == (3,) and bool(np.all(np.isfinite(v))):
            return v.astype(np.float32)
        return self._estimate_velocity(position)

    def _rocket_angular_velocity(self) -> np.ndarray:
        w = self.env.rocket.get_angular_velocity()
        if w is not None and w.shape == (3,) and bool(np.all(np.isfinite(w))):
            return w.astype(np.float32)
        return np.zeros(3, dtype=np.float32)

    def _estimate_velocity(self, position: np.ndarray) -> np.ndarray:
        if self.memory is None:
            return np.zeros(3, dtype=np.float32)
        return ((position - self.memory.previous_position) / max(self.control_dt, 1e-6)).astype(np.float32)

    def _estimate_acceleration(self, velocity: np.ndarray) -> np.ndarray:
        if self.memory is None:
            return np.zeros(3, dtype=np.float32)
        return ((velocity - self.memory.previous_velocity) / max(self.control_dt, 1e-6)).astype(np.float32)

    def _build_sensor_dict(self, state: RocketKinematicState) -> dict[str, np.ndarray]:
        return {
            "lidar": self._synthetic_lidar(state),
            "depth": np.ones((1, 64, 64), dtype=np.float32) * min(state.altitude_to_ground, self.max_lidar_distance),
            "rgb": np.zeros((3, 84, 84), dtype=np.float32),
        }

    def _synthetic_lidar(self, state: RocketKinematicState) -> np.ndarray:
        angles = np.linspace(0.0, 2.0 * np.pi, 32, endpoint=False)
        radius = float(self.config.get("lidar_ring_radius_m", 2.0))
        distances = []
        for angle in angles:
            x = float(state.position[0] + radius * np.cos(angle))
            y = float(state.position[1] + radius * np.sin(angle))
            ground = self._terrain_height_at(x, y)
            distances.append(np.clip(float(state.position[2] - ground), 0.0, self.max_lidar_distance))
        return np.asarray(distances, dtype=np.float32)

    def _build_terrain_state(self, state: RocketKinematicState) -> TerrainSafetyState:
        target = state.target_position
        slope, roughness = self._terrain_slope_roughness_at(float(target[0]), float(target[1]))
        slope_norm = float(np.clip(slope / 25.0, 0.0, 1.0))
        roughness_norm = float(np.clip(roughness / 0.25, 0.0, 1.0))
        safe_zone_score = float(np.clip(1.0 - 0.55 * slope_norm - 0.45 * roughness_norm, 0.0, 1.0))
        return TerrainSafetyState(
            local_slope_deg=slope,
            local_roughness_m=roughness,
            safe_zone_score=safe_zone_score,
            terrain_height_at_target=float(target[2]),
            slope_norm=slope_norm,
            roughness_norm=roughness_norm,
        )

    def _estimate_contact(self, state: RocketKinematicState) -> ContactState:
        in_contact = bool(state.altitude_to_ground < float(self.config.get("contact_altitude_m", 0.15)))
        return ContactState(
            leg_contact=np.array([in_contact, in_contact, in_contact, in_contact], dtype=bool),
            leg_contact_force=np.zeros(4, dtype=np.float32),
            body_contact=False,
        )

    def _altitude_to_ground(self, position: np.ndarray) -> float:
        return max(0.0, float(position[2]) - self._terrain_height_at(float(position[0]), float(position[1])))

    def _terrain_height_at(self, x: float, y: float) -> float:
        if self.env.state is None:
            return 0.0
        terrain = self._mesh_for_xy(x, y)
        return _height_at_mesh(terrain.heights, terrain.size_m, x, y, terrain.vertices)

    def _terrain_slope_roughness_at(self, x: float, y: float) -> tuple[float, float]:
        if self.env.state is None:
            return 0.0, 0.0
        terrain = self._mesh_for_xy(x, y)
        rows, cols = terrain.heights.shape
        min_x, max_x, min_y, max_y = _mesh_bounds(terrain.vertices)
        col = int(np.clip((x - min_x) / max(max_x - min_x, 1e-6) * (cols - 1), 0, cols - 1))
        row = int(np.clip((y - min_y) / max(max_y - min_y, 1e-6) * (rows - 1), 0, rows - 1))
        radius = 4
        patch = terrain.heights[max(0, row - radius) : min(rows, row + radius + 1), max(0, col - radius) : min(cols, col + radius + 1)]
        if patch.size < 4:
            return 0.0, 0.0
        spacing_x = terrain.size_m[0] / max(1, cols - 1)
        spacing_y = terrain.size_m[1] / max(1, rows - 1)
        dy, dx = np.gradient(patch, spacing_y, spacing_x)
        slope = float(np.degrees(np.arctan(np.mean(np.sqrt(dx * dx + dy * dy)))))
        roughness = float(np.std(patch))
        return slope, roughness

    def _mesh_for_xy(self, x: float, y: float):
        assert self.env.state is not None
        local = self.env.state.terrain.local_detail_mesh
        if local is not None:
            min_x, max_x, min_y, max_y = _mesh_bounds(local.vertices)
            if min_x <= x <= max_x and min_y <= y <= max_y:
                return local
        return self.env.state.terrain

    def _is_out_of_bounds(self, position: np.ndarray) -> bool:
        if self.env.state is None:
            return False
        size_x, size_y = self.env.state.terrain.size_m
        return bool(abs(float(position[0])) > size_x * 0.5 or abs(float(position[1])) > size_y * 0.5)


def _height_at_mesh(
    heights: np.ndarray,
    size_m: tuple[float, float],
    x: float,
    y: float,
    vertices: list[tuple[float, float, float]],
) -> float:
    rows, cols = heights.shape
    min_x, max_x, min_y, max_y = _mesh_bounds(vertices)
    col_f = np.clip((x - min_x) / max(max_x - min_x, 1e-6) * (cols - 1), 0, cols - 1)
    row_f = np.clip((y - min_y) / max(max_y - min_y, 1e-6) * (rows - 1), 0, rows - 1)
    c0 = int(np.floor(col_f))
    r0 = int(np.floor(row_f))
    c1 = min(cols - 1, c0 + 1)
    r1 = min(rows - 1, r0 + 1)
    cf = float(col_f - c0)
    rf = float(row_f - r0)
    top = float(heights[r0, c0]) * (1.0 - cf) + float(heights[r0, c1]) * cf
    bottom = float(heights[r1, c0]) * (1.0 - cf) + float(heights[r1, c1]) * cf
    return top * (1.0 - rf) + bottom * rf


_BOUNDS_CACHE: dict[int, tuple[float, float, float, float]] = {}


def _mesh_bounds(vertices: list[tuple[float, float, float]]) -> tuple[float, float, float, float]:
    vid = id(vertices)
    if vid in _BOUNDS_CACHE:
        return _BOUNDS_CACHE[vid]
    xs = [v[0] for v in vertices]
    ys = [v[1] for v in vertices]
    bounds = (min(xs), max(xs), min(ys), max(ys))
    _BOUNDS_CACHE[vid] = bounds
    return bounds
