from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class CameraObservation:
    rgb: Any


class SensorSuite:
    def __init__(self, config: dict[str, Any]):
        self.config = config
        camera_cfg = config.get("camera", {})
        self.prim_path = str(camera_cfg.get("prim_path", "/World/Rocket/Camera"))
        self.resolution = tuple(camera_cfg.get("resolution", [640, 480]))
        self.frequency = float(camera_cfg.get("frequency_hz", 30.0))
        self.translation = tuple(camera_cfg.get("translation", [0.0, -8.0, 4.0]))
        self.rotation = tuple(camera_cfg.get("rotation_deg", [60.0, 0.0, 0.0]))
        self.focal_length = float(camera_cfg.get("focal_length_mm", 24.0))
        self._camera: Any | None = None

    def build(self) -> Any:
        camera_cls = self._camera_class()
        self._camera = camera_cls(
            prim_path=self.prim_path,
            name="rocket_rgb_camera",
            frequency=self.frequency,
            resolution=self.resolution,
            translation=self.translation,
            orientation=self._euler_xyz_to_quat_wxyz(self.rotation),
        )
        try:
            self._camera.initialize()
            self._camera.set_focal_length(self.focal_length)
        except Exception:
            pass
        return self._camera

    def get_observation(self) -> CameraObservation:
        if self._camera is None:
            raise RuntimeError("SensorSuite.build() must be called before observations are available")
        frame = self._camera.get_current_frame()
        rgb = frame.get("rgba")
        if rgb is None:
            rgb = frame.get("rgb")
        return CameraObservation(rgb=rgb)

    def _camera_class(self) -> Any:
        candidates = (
            "isaacsim.sensors.camera",
            "omni.isaac.sensor",
        )
        for module_name in candidates:
            try:
                module = __import__(module_name, fromlist=["Camera"])
                return module.Camera
            except (ImportError, AttributeError):
                continue
        raise RuntimeError("Isaac Sim Camera API not found")

    def _euler_xyz_to_quat_wxyz(self, euler_deg: tuple[float, float, float]) -> tuple[float, float, float, float]:
        import numpy as np

        roll, pitch, yaw = np.deg2rad(np.array(euler_deg, dtype=float))
        cr, sr = np.cos(roll / 2.0), np.sin(roll / 2.0)
        cp, sp = np.cos(pitch / 2.0), np.sin(pitch / 2.0)
        cy, sy = np.cos(yaw / 2.0), np.sin(yaw / 2.0)
        return (
            float(cr * cp * cy + sr * sp * sy),
            float(sr * cp * cy - cr * sp * sy),
            float(cr * sp * cy + sr * cp * sy),
            float(cr * cp * sy - sr * sp * cy),
        )
