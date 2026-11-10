from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from app.randomization import ResetSample, sample_reset
from app.rl_types import ContactState, RocketKinematicState, TerrainSafetyState
from app.terrain_generator import MoonTerrainGenerator, TerrainMeshData


@dataclass
class AnalyticRocketState:
    position: np.ndarray
    velocity: np.ndarray
    acceleration: np.ndarray
    quaternion_wxyz: np.ndarray
    angular_velocity: np.ndarray
    previous_action: np.ndarray


@dataclass
class MeshSampler:
    heights: np.ndarray
    min_x: float
    max_x: float
    min_y: float
    max_y: float
    size_m: tuple[float, float]


class AnalyticLunarLandingAdapter:
    """Fast training adapter that keeps high-detail terrain out of the Isaac stage.

    The adapter preserves high-resolution terrain and rock information for
    observation/reward, while using a simple rocket dynamics model. It is meant
    for RL throughput; Isaac-rendered scenes remain the evaluation path.
    """

    def __init__(self, configs: dict[str, dict[str, Any]], config: dict[str, Any] | None = None):
        self.configs = configs
        self.config = config or {}
        self.sim_config = configs["sim"]
        self.terrain_config = configs["terrain"]
        self.rocket_config = configs["rocket"]
        self.terrain_generator = MoonTerrainGenerator(self.terrain_config)
        self.rng = np.random.default_rng(int(self.sim_config.get("seed", 1234)))
        self.control_dt = float(self.config.get("control_dt", 0.05))
        self.max_lidar_distance = float(self.config.get("max_lidar_distance_m", 30.0))
        self.mass_kg = float(self.rocket_config.get("mass_kg", 1.63))
        self.max_thrust_n = float(self.rocket_config.get("thrust", {}).get("max_newtons", 30.0))
        self.max_gimbal_deg = float(self.config.get("max_gimbal_deg", 12.0))
        self.reset_sample: ResetSample | None = None
        self.terrain: TerrainMeshData | None = None
        self.base_sampler: MeshSampler | None = None
        self.local_sampler: MeshSampler | None = None
        self.rocks: np.ndarray = np.zeros((0, 3), dtype=np.float32)
        self.state: AnalyticRocketState | None = None

    def reset(self) -> tuple[RocketKinematicState, dict[str, np.ndarray], TerrainSafetyState, ContactState]:
        self.terrain_generator.load_dem()
        self.reset_sample = sample_reset(self.terrain_config, self.rocket_config, self.rng)
        self.terrain = self.terrain_generator.generate(self.reset_sample)
        self.base_sampler = _sampler_from_mesh(self.terrain)
        self.local_sampler = (
            _sampler_from_mesh(self.terrain.local_detail_mesh) if self.terrain.local_detail_mesh is not None else None
        )
        self.rocks = self._sample_rocks(self.reset_sample)

        x, y, altitude = self.reset_sample.rocket_position
        ground = self._terrain_height_at(float(x), float(y))
        position = np.array([x, y, ground + altitude], dtype=np.float32)
        self.state = AnalyticRocketState(
            position=position,
            velocity=np.zeros(3, dtype=np.float32),
            acceleration=np.zeros(3, dtype=np.float32),
            quaternion_wxyz=_euler_xyz_to_quat_wxyz(self.reset_sample.rocket_euler_deg),
            angular_velocity=np.zeros(3, dtype=np.float32),
            previous_action=np.zeros(3, dtype=np.float32),
        )
        return self._outputs(np.zeros(3, dtype=np.float32))

    def step(
        self, action: np.ndarray
    ) -> tuple[RocketKinematicState, dict[str, np.ndarray], TerrainSafetyState, ContactState]:
        if self.state is None:
            raise RuntimeError("AnalyticLunarLandingAdapter.reset() must be called before step()")
        action = np.asarray(action, dtype=np.float32)
        if action.size < 3:
            action = np.pad(action, (0, 3 - action.size), constant_values=0.0).astype(np.float32)

        throttle = float(np.clip(action[0], 0.0, 1.0))
        gx = float(np.clip(action[1], -1.0, 1.0))
        gy = float(np.clip(action[2], -1.0, 1.0))
        max_gimbal_rad = np.deg2rad(self.max_gimbal_deg)
        direction = np.array(
            [
                np.sin(gx * max_gimbal_rad),
                np.sin(gy * max_gimbal_rad),
                np.cos(gx * max_gimbal_rad) * np.cos(gy * max_gimbal_rad),
            ],
            dtype=np.float32,
        )
        direction /= max(float(np.linalg.norm(direction)), 1e-6)

        gravity = np.array([0.0, 0.0, -float(self.sim_config.get("gravity_mps2", 1.62))], dtype=np.float32)
        thrust_acc = direction * (self.max_thrust_n * throttle / max(self.mass_kg, 1e-6))
        acceleration = gravity + thrust_acc
        self.state.velocity = self.state.velocity + acceleration * self.control_dt
        self.state.position = self.state.position + self.state.velocity * self.control_dt
        self.state.acceleration = acceleration.astype(np.float32)
        previous_quat = self.state.quaternion_wxyz.copy()
        self.state.quaternion_wxyz = _direction_to_quat(direction)
        # Kinematic body rate: finite-difference of the thrust-aligned attitude.
        # The analytic model has no true rotational state, so this is the rate at
        # which its (instantaneously thrust-aligned) body frame turns.
        self.state.angular_velocity = (
            _quat_finite_diff_omega(previous_quat, self.state.quaternion_wxyz, self.control_dt)
        ).astype(np.float32)
        self.state.previous_action = action.copy()

        ground = self._terrain_height_at(float(self.state.position[0]), float(self.state.position[1]))
        if float(self.state.position[2]) < ground:
            self.state.position[2] = ground

        return self._outputs(action)

    def _outputs(
        self, action: np.ndarray
    ) -> tuple[RocketKinematicState, dict[str, np.ndarray], TerrainSafetyState, ContactState]:
        assert self.state is not None
        target = self._target_position()
        altitude = self._altitude_to_ground(self.state.position)
        contact = self._estimate_contact(altitude)
        state = RocketKinematicState(
            position=self.state.position.copy(),
            target_position=target,
            linear_velocity=self.state.velocity.copy(),
            quaternion_wxyz=self.state.quaternion_wxyz.copy(),
            angular_velocity=self.state.angular_velocity.copy(),
            linear_acceleration=self.state.acceleration.copy(),
            altitude_to_ground=altitude,
            main_thrust=float(np.clip(action[0], 0.0, 1.0)) if action.size else 0.0,
            gimbal_x=float(np.clip(action[1], -1.0, 1.0)) if action.size > 1 else 0.0,
            gimbal_y=float(np.clip(action[2], -1.0, 1.0)) if action.size > 2 else 0.0,
            out_of_bounds=self._is_out_of_bounds(self.state.position),
        )
        sensors = {
            "lidar": self._synthetic_lidar(state),
            "depth": np.ones((1, 64, 64), dtype=np.float32) * min(altitude, self.max_lidar_distance),
            "rgb": np.zeros((3, 84, 84), dtype=np.float32),
        }
        terrain = self._build_terrain_state(state)
        return state, sensors, terrain, contact

    def _target_position(self) -> np.ndarray:
        assert self.reset_sample is not None
        x, y, _ = self.reset_sample.target_position
        return np.array([x, y, self._terrain_height_at(float(x), float(y))], dtype=np.float32)

    def _synthetic_lidar(self, state: RocketKinematicState) -> np.ndarray:
        angles = np.linspace(0.0, 2.0 * np.pi, 32, endpoint=False)
        radius = float(self.config.get("lidar_ring_radius_m", 2.0))
        distances = []
        for angle in angles:
            x = float(state.position[0] + radius * np.cos(angle))
            y = float(state.position[1] + radius * np.sin(angle))
            ground = self._terrain_height_at(x, y)
            rock_height = self._rock_height_at(x, y)
            distances.append(np.clip(float(state.position[2] - max(ground, rock_height)), 0.0, self.max_lidar_distance))
        return np.asarray(distances, dtype=np.float32)

    def _build_terrain_state(self, state: RocketKinematicState) -> TerrainSafetyState:
        target = state.target_position
        slope, roughness = self._terrain_slope_roughness_at(float(target[0]), float(target[1]))
        rock_penalty = self._rock_penalty_at(float(target[0]), float(target[1]))
        slope_norm = float(np.clip(slope / 25.0, 0.0, 1.0))
        roughness_norm = float(np.clip(roughness / 0.25, 0.0, 1.0))
        safe_zone_score = float(np.clip(1.0 - 0.50 * slope_norm - 0.35 * roughness_norm - 0.15 * rock_penalty, 0.0, 1.0))
        return TerrainSafetyState(
            local_slope_deg=slope,
            local_roughness_m=roughness,
            safe_zone_score=safe_zone_score,
            terrain_height_at_target=float(target[2]),
            slope_norm=slope_norm,
            roughness_norm=roughness_norm,
        )

    def _estimate_contact(self, altitude: float) -> ContactState:
        in_contact = bool(altitude < float(self.config.get("contact_altitude_m", 0.15)))
        force = np.zeros(4, dtype=np.float32)
        if in_contact and self.state is not None:
            force[:] = abs(float(self.state.velocity[2])) * self.mass_kg / max(self.control_dt, 1e-6) / 4.0
        return ContactState(
            leg_contact=np.array([in_contact, in_contact, in_contact, in_contact], dtype=bool),
            leg_contact_force=force,
            body_contact=False,
        )

    def _sample_rocks(self, reset_sample: ResetSample) -> np.ndarray:
        rocks_config = self.terrain_config.get("rocks", {})
        if not bool(rocks_config.get("enabled", True)):
            return np.zeros((0, 3), dtype=np.float32)
        total = int(rocks_config.get("total_count", 35))
        lz_percentage = float(rocks_config.get("landing_zone_percentage", 0.75))
        min_dist = float(rocks_config.get("min_distance_from_center_m", 2.0))
        lz_radius = float(rocks_config.get("landing_zone_radius_m", 12.0))
        outer_radius = float(rocks_config.get("outer_radius_m", 35.0))
        scale_min, scale_max = [float(v) for v in rocks_config.get("scale_range", [0.12, 0.55])]
        center_x, center_y = reset_sample.target_position[:2]
        rocks = []
        for _ in range(total):
            if float(self.rng.random()) < lz_percentage:
                radius = float(self.rng.uniform(min_dist, lz_radius))
            else:
                radius = float(self.rng.uniform(lz_radius, outer_radius))
            angle = float(self.rng.uniform(0.0, 2.0 * np.pi))
            scale = float(self.rng.uniform(scale_min, scale_max))
            rocks.append((center_x + radius * np.cos(angle), center_y + radius * np.sin(angle), scale))
        return np.asarray(rocks, dtype=np.float32)

    def _altitude_to_ground(self, position: np.ndarray) -> float:
        return max(0.0, float(position[2]) - self._terrain_height_at(float(position[0]), float(position[1])))

    def _terrain_height_at(self, x: float, y: float) -> float:
        sampler = self._sampler_for_xy(x, y)
        return _height_at_sampler(sampler, x, y)

    def _terrain_slope_roughness_at(self, x: float, y: float) -> tuple[float, float]:
        sampler = self._sampler_for_xy(x, y)
        rows, cols = sampler.heights.shape
        col = int(np.clip((x - sampler.min_x) / max(sampler.max_x - sampler.min_x, 1e-6) * (cols - 1), 0, cols - 1))
        row = int(np.clip((y - sampler.min_y) / max(sampler.max_y - sampler.min_y, 1e-6) * (rows - 1), 0, rows - 1))
        radius = 4
        patch = sampler.heights[
            max(0, row - radius) : min(rows, row + radius + 1),
            max(0, col - radius) : min(cols, col + radius + 1),
        ]
        if patch.size < 4:
            return 0.0, 0.0
        spacing_x = sampler.size_m[0] / max(1, cols - 1)
        spacing_y = sampler.size_m[1] / max(1, rows - 1)
        dy, dx = np.gradient(patch, spacing_y, spacing_x)
        slope = float(np.degrees(np.arctan(np.mean(np.sqrt(dx * dx + dy * dy)))))
        roughness = float(np.std(patch))
        return slope, roughness

    def _sampler_for_xy(self, x: float, y: float) -> MeshSampler:
        assert self.base_sampler is not None
        local = self.local_sampler
        if local is not None and local.min_x <= x <= local.max_x and local.min_y <= y <= local.max_y:
            return local
        return self.base_sampler

    def _rock_height_at(self, x: float, y: float) -> float:
        if self.rocks.size == 0:
            return -np.inf
        dx = self.rocks[:, 0] - x
        dy = self.rocks[:, 1] - y
        radius = np.maximum(self.rocks[:, 2] * 0.7, 1e-3)
        inside = dx * dx + dy * dy <= radius * radius
        if not bool(np.any(inside)):
            return -np.inf
        return self._terrain_height_at(x, y) + float(np.max(self.rocks[inside, 2]))

    def _rock_penalty_at(self, x: float, y: float) -> float:
        if self.rocks.size == 0:
            return 0.0
        distance = np.sqrt((self.rocks[:, 0] - x) ** 2 + (self.rocks[:, 1] - y) ** 2)
        clearance = distance - np.maximum(self.rocks[:, 2] * 0.8, 0.1)
        return float(np.clip(np.exp(-max(float(np.min(clearance)), 0.0) / 1.5), 0.0, 1.0))

    def _is_out_of_bounds(self, position: np.ndarray) -> bool:
        size_x, size_y = self.terrain_config.get("terrain_size_m", [80.0, 80.0])
        return bool(abs(float(position[0])) > float(size_x) * 0.5 or abs(float(position[1])) > float(size_y) * 0.5)


def _sampler_from_mesh(mesh: TerrainMeshData) -> MeshSampler:
    xs = [v[0] for v in mesh.vertices]
    ys = [v[1] for v in mesh.vertices]
    return MeshSampler(
        heights=np.asarray(mesh.heights, dtype=np.float32),
        min_x=float(min(xs)),
        max_x=float(max(xs)),
        min_y=float(min(ys)),
        max_y=float(max(ys)),
        size_m=mesh.size_m,
    )


def _height_at_sampler(sampler: MeshSampler, x: float, y: float) -> float:
    rows, cols = sampler.heights.shape
    col_f = np.clip((x - sampler.min_x) / max(sampler.max_x - sampler.min_x, 1e-6) * (cols - 1), 0, cols - 1)
    row_f = np.clip((y - sampler.min_y) / max(sampler.max_y - sampler.min_y, 1e-6) * (rows - 1), 0, rows - 1)
    c0 = int(np.floor(col_f))
    r0 = int(np.floor(row_f))
    c1 = min(cols - 1, c0 + 1)
    r1 = min(rows - 1, r0 + 1)
    cf = float(col_f - c0)
    rf = float(row_f - r0)
    top = float(sampler.heights[r0, c0]) * (1.0 - cf) + float(sampler.heights[r0, c1]) * cf
    bottom = float(sampler.heights[r1, c0]) * (1.0 - cf) + float(sampler.heights[r1, c1]) * cf
    return top * (1.0 - rf) + bottom * rf


def _euler_xyz_to_quat_wxyz(euler_deg: tuple[float, float, float]) -> np.ndarray:
    roll, pitch, yaw = np.deg2rad(np.array(euler_deg, dtype=np.float32))
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


def _quat_finite_diff_omega(q_prev: np.ndarray, q_curr: np.ndarray, dt: float) -> np.ndarray:
    """World-frame angular velocity (rad/s) from two successive wxyz quaternions."""
    qp = np.asarray(q_prev, dtype=np.float64)
    qc = np.asarray(q_curr, dtype=np.float64)
    qp = qp / max(float(np.linalg.norm(qp)), 1e-12)
    qc = qc / max(float(np.linalg.norm(qc)), 1e-12)
    # relative rotation q_rel = q_curr * conj(q_prev)
    w0, x0, y0, z0 = qc
    w1, x1, y1, z1 = qp[0], -qp[1], -qp[2], -qp[3]
    rel = np.array([
        w0 * w1 - x0 * x1 - y0 * y1 - z0 * z1,
        w0 * x1 + x0 * w1 + y0 * z1 - z0 * y1,
        w0 * y1 - x0 * z1 + y0 * w1 + z0 * x1,
        w0 * z1 + x0 * y1 - y0 * x1 + z0 * w1,
    ])
    if rel[0] < 0.0:
        rel = -rel
    angle = 2.0 * np.arccos(np.clip(rel[0], -1.0, 1.0))
    axis_norm = float(np.linalg.norm(rel[1:4]))
    if axis_norm < 1e-9 or dt <= 0.0:
        return np.zeros(3, dtype=np.float64)
    axis = rel[1:4] / axis_norm
    return axis * (angle / dt)


def _direction_to_quat(direction: np.ndarray) -> np.ndarray:
    direction = np.asarray(direction, dtype=np.float32)
    direction /= max(float(np.linalg.norm(direction)), 1e-6)
    z_axis = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    dot = float(np.clip(np.dot(z_axis, direction), -1.0, 1.0))
    if dot > 0.9999:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    axis = np.cross(z_axis, direction)
    axis /= max(float(np.linalg.norm(axis)), 1e-6)
    angle = float(np.arccos(dot))
    half = angle * 0.5
    return np.array([np.cos(half), *(axis * np.sin(half))], dtype=np.float32)
