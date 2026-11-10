from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ResetSample:
    seed: int
    terrain_origin: tuple[int, int]
    rocket_position: tuple[float, float, float]
    target_position: tuple[float, float, float]
    rocket_euler_deg: tuple[float, float, float]
    crater_count: int
    slope: tuple[float, float]
    roughness_scale: float


def _range_pair(config: dict[str, Any], key: str, default: tuple[float, float]) -> tuple[float, float]:
    value = config.get(key, default)
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{key} must be a two-value range")
    low, high = float(value[0]), float(value[1])
    if low > high:
        raise ValueError(f"{key} lower bound must be <= upper bound")
    return low, high


def sample_reset(
    terrain_config: dict[str, Any],
    rocket_config: dict[str, Any],
    rng: np.random.Generator | None = None,
) -> ResetSample:
    rng = rng or np.random.default_rng()
    seed = int(rng.integers(0, np.iinfo(np.int32).max))

    patch_shape = terrain_config.get("patch_shape_pixels", [256, 256])
    if len(patch_shape) != 2:
        raise ValueError("terrain.patch_shape_pixels must contain [height, width]")

    max_y = max(0, int(terrain_config.get("dem_height_pixels", patch_shape[0])) - int(patch_shape[0]))
    max_x = max(0, int(terrain_config.get("dem_width_pixels", patch_shape[1])) - int(patch_shape[1]))
    terrain_origin = (int(rng.integers(0, max_y + 1)), int(rng.integers(0, max_x + 1)))

    crater_min, crater_max = terrain_config.get("crater_count_range", [12, 36])
    crater_count = int(rng.integers(int(crater_min), int(crater_max) + 1))

    if "base_slope_deg_range" in terrain_config:
        slope_deg_range = _range_pair(terrain_config, "base_slope_deg_range", (0.0, 2.5))
        slope_magnitude = float(np.tan(np.deg2rad(rng.uniform(*slope_deg_range))))
        slope_direction = float(rng.uniform(0.0, 2.0 * np.pi))
        sampled_slope = (
            slope_magnitude * float(np.cos(slope_direction)),
            slope_magnitude * float(np.sin(slope_direction)),
        )
    else:
        slope_x = _range_pair(terrain_config, "slope_x_range", (-0.05, 0.05))
        slope_y = _range_pair(terrain_config, "slope_y_range", (-0.05, 0.05))
        sampled_slope = (float(rng.uniform(*slope_x)), float(rng.uniform(*slope_y)))
    roughness = _range_pair(terrain_config, "roughness_scale_range", (0.1, 0.7))

    spawn = rocket_config.get("spawn", {})
    if bool(spawn.get("sample_xy_from_terrain", False)):
        terrain_size = terrain_config.get("terrain_size_m", [80.0, 80.0])
        if len(terrain_size) != 2:
            raise ValueError("terrain.terrain_size_m must contain [x, y]")
        margin = float(spawn.get("terrain_margin_m", 8.0))
        half_x = max(0.0, float(terrain_size[0]) * 0.5 - margin)
        half_y = max(0.0, float(terrain_size[1]) * 0.5 - margin)
        xy_range_x = (-half_x, half_x)
        xy_range_y = (-half_y, half_y)
    else:
        xy_range = _range_pair(spawn, "xy_range_m", (-4.0, 4.0))
        xy_range_x = xy_range
        xy_range_y = xy_range
    attitude_range = _range_pair(spawn, "attitude_deg_range", (-3.0, 3.0))
    altitude = float(spawn.get("altitude_m", 30.0))

    target_x = float(rng.uniform(*xy_range_x))
    target_y = float(rng.uniform(*xy_range_y))
    target_position = (target_x, target_y, 0.0)

    # Offset rocket's spawn position laterally from the target landing spot
    offset_range = _range_pair(spawn, "xy_offset_range_m", (-5.0, 5.0))
    offset_x = float(rng.uniform(*offset_range))
    offset_y = float(rng.uniform(*offset_range))
    
    rocket_position = (
        target_x + offset_x,
        target_y + offset_y,
        altitude,
    )
    
    rocket_euler_deg = (
        float(rng.uniform(*attitude_range)),
        float(rng.uniform(*attitude_range)),
        float(rng.uniform(-180.0, 180.0)),
    )

    return ResetSample(
        seed=seed,
        terrain_origin=terrain_origin,
        rocket_position=rocket_position,
        target_position=target_position,
        rocket_euler_deg=rocket_euler_deg,
        crater_count=crater_count,
        slope=sampled_slope,
        roughness_scale=float(rng.uniform(*roughness)),
    )
