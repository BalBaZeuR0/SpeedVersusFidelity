from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import shutil

import numpy as np

from app.config import resolve_project_path


def _quat_rotate_wxyz(quat_wxyz: np.ndarray, vec: np.ndarray) -> np.ndarray:
    """Rotate ``vec`` by the wxyz quaternion (same convention as Isaac / isaaclab)."""
    q = np.asarray(quat_wxyz, dtype=float)
    w, xyz = float(q[0]), q[1:4]
    v = np.asarray(vec, dtype=float)
    t = 2.0 * np.cross(xyz, v)
    return v + w * t + np.cross(xyz, t)


@dataclass
class RocketState:
    prim_path: str
    position: tuple[float, float, float]
    euler_deg: tuple[float, float, float]


class RocketAsset:
    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.prim_path = str(config.get("prim_path", "/World/Rocket"))
        self.rigid_body_path = str(config.get("rigid_body_path", self.prim_path))
        self.mjcf_path = resolve_project_path(config.get("mjcf_path", "assets/rocket/rocket.xml"))
        self.usd_path = resolve_project_path(config.get("usd_path", "assets/rocket/rocket.usd"))
        self.mass_kg = float(config.get("mass_kg", 1200.0))
        self.visual_scale = float(config.get("visual_scale", 1.0))
        self.thrust_config = config.get("thrust", {})
        self._rigid_prim: Any | None = None
        self._prim: Any | None = None
        self._converted_usd_path: Path | None = None

    def prepare_usd(self) -> Path:
        if self._converted_usd_path and self._converted_usd_path.exists():
            return self._converted_usd_path
        if self.usd_path.exists() and self.usd_path.stat().st_size > 0:
            print(f"[LunarRocket] using cached rocket USD: {self.usd_path}", flush=True)
            return self.usd_path
        if not self.mjcf_path.exists():
            raise FileNotFoundError(
                "Rocket USD is missing and no MuJoCo model was found. "
                f"Place the MJCF/XML model at {self.mjcf_path} or preconvert it to {self.usd_path}."
            )
        self.usd_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"[LunarRocket] converting MJCF to USD: {self.mjcf_path} -> {self.usd_path}", flush=True)
        converted_path = self._convert_mjcf_to_usd()
        if not converted_path.exists():
            raise RuntimeError(f"MJCF conversion completed without creating {converted_path}")
        self._converted_usd_path = converted_path
        return converted_path

    def spawn(self, stage: Any, world: Any | None = None) -> Any:
        usd_path = self.prepare_usd()
        try:
            from pxr import Gf, Sdf, UsdPhysics
            try:
                from isaacsim.core.utils.stage import add_reference_to_stage
            except ImportError:
                from omni.isaac.core.utils.stage import add_reference_to_stage
        except ImportError as exc:
            raise RuntimeError("Isaac Sim Python modules are required to spawn the rocket") from exc

        if stage.GetPrimAtPath(self.prim_path):
            stage.RemovePrim(self.prim_path)

        add_reference_to_stage(str(usd_path), self.prim_path)
        print(f"[LunarRocket] rocket referenced at {self.prim_path}", flush=True)
        self._prim = stage.GetPrimAtPath(self.prim_path)
        if not self._prim:
            raise RuntimeError(f"Rocket prim was not created at {self.prim_path}")

        physics_prim = stage.GetPrimAtPath(self.rigid_body_path) or self._prim
        UsdPhysics.RigidBodyAPI.Apply(physics_prim)
        mass_api = UsdPhysics.MassAPI.Apply(physics_prim)
        mass_api.CreateMassAttr(self.mass_kg)
        physics_prim.CreateAttribute("physxRigidBody:disableGravity", Sdf.ValueTypeNames.Bool).Set(False)
        self._rigid_prim = self._create_rigid_view(str(physics_prim.GetPath()), world)

        return physics_prim

    def reset_pose(
        self,
        position: tuple[float, float, float],
        euler_deg: tuple[float, float, float],
        log: bool = True,
    ) -> RocketState:
        if self._rigid_prim is not None:
            quat = self._euler_xyz_to_quat_wxyz(euler_deg)
            self._set_rigid_world_pose(position, quat)
            self._set_rigid_velocities(np.zeros(3), np.zeros(3))
        elif self._prim is not None:
            self._set_xform_pose(self._prim, position, euler_deg)
        if log:
            print(
                "[LunarRocket] rocket pose "
                f"position=({position[0]:.3f}, {position[1]:.3f}, {position[2]:.3f}) "
                f"euler_deg=({euler_deg[0]:.2f}, {euler_deg[1]:.2f}, {euler_deg[2]:.2f}) "
                f"visual_scale={self.visual_scale:.2f}",
                flush=True,
            )
        return RocketState(self.prim_path, position, euler_deg)

    def apply_thrust(
        self,
        throttle: float,
        direction: tuple[float, float, float] | None = None,
    ) -> np.ndarray:
        """Apply the engine wrench for this step.

        ``direction`` is the gimbal-deflected thrust unit vector **in the rocket
        body frame** (``[sin(gx.theta), sin(gy.theta), cos.cos]``). With
        ``thrust.six_dof`` enabled (default) the force is rotated into the world
        frame by the current body attitude and the gimbal moment
        ``tau = lever x F`` (lever = nozzle offset below the CoM) is applied as a
        couple, so a sustained gimbal produces real angular acceleration - the
        behaviour the kinematic analytic adapter cannot reproduce. With
        ``six_dof: false`` it falls back to the legacy world-frame point force at
        the CoM (no torque).
        """
        throttle = float(np.clip(throttle, 0.0, 1.0))
        max_thrust = float(self.thrust_config.get("max_newtons", 18000.0))
        vector = np.array(direction or self.thrust_config.get("default_direction", [0.0, 0.0, 1.0]), dtype=float)
        norm = float(np.linalg.norm(vector))
        if norm == 0.0:
            raise ValueError("Thrust direction must be non-zero")
        unit = vector / norm
        force_body = unit * max_thrust * throttle
        six_dof = bool(self.thrust_config.get("six_dof", True))

        if self._rigid_prim is None:
            return force_body

        if not six_dof:
            self._apply_wrench(force_body, np.zeros(3))
            return force_body

        pose = self.get_world_pose()
        quat = np.asarray(pose[1], dtype=float) if pose is not None else np.array([1.0, 0.0, 0.0, 0.0])
        lever_arm = float(self.thrust_config.get("lever_arm_m", 0.25))
        torque_body = np.cross(np.array([0.0, 0.0, -lever_arm]), force_body)
        force_world = _quat_rotate_wxyz(quat, force_body)
        torque_world = _quat_rotate_wxyz(quat, torque_body)
        self._apply_wrench(force_world, torque_world)
        return force_world

    def _apply_wrench(self, force: np.ndarray, torque: np.ndarray) -> None:
        rp = self._rigid_prim
        f = np.asarray([force], dtype=np.float32)
        t = np.asarray([torque], dtype=np.float32)
        try:
            if hasattr(rp, "apply_forces_and_torques_at_pos"):
                rp.apply_forces_and_torques_at_pos(forces=f, torques=t, is_global=True)
                return
        except Exception as exc:  # noqa: BLE001 - Isaac API surface varies by version
            print(f"[LunarRocket] warning: apply_forces_and_torques_at_pos failed ({exc}); falling back", flush=True)
        try:
            if hasattr(rp, "apply_forces"):
                rp.apply_forces(f, is_global=True)
            elif hasattr(rp, "apply_force"):
                rp.apply_force(np.asarray(force, dtype=np.float32))
            if float(np.linalg.norm(torque)) > 0.0:
                if hasattr(rp, "apply_torques"):
                    rp.apply_torques(t)
                elif hasattr(rp, "apply_torque"):
                    rp.apply_torque(np.asarray(torque, dtype=np.float32))
        except Exception as exc:  # noqa: BLE001
            print(f"[LunarRocket] warning: could not apply thrust wrench ({exc})", flush=True)

    def get_linear_velocity(self) -> np.ndarray | None:
        return self._read_rigid_vector(("get_linear_velocities", "get_linear_velocity"))

    def get_angular_velocity(self) -> np.ndarray | None:
        return self._read_rigid_vector(("get_angular_velocities", "get_angular_velocity"))

    def _read_rigid_vector(self, method_names: tuple[str, ...]) -> np.ndarray | None:
        if self._rigid_prim is None:
            return None
        for name in method_names:
            fn = getattr(self._rigid_prim, name, None)
            if fn is None:
                continue
            try:
                value = fn()
            except Exception:
                return None
            if value is None:
                return None
            arr = np.asarray(value, dtype=np.float32)
            if arr.ndim == 2:
                arr = arr[0]
            return arr if arr.shape == (3,) else None
        return None

    def get_world_pose(self) -> tuple[np.ndarray, np.ndarray] | None:
        if self._rigid_prim is None:
            return None
        try:
            if hasattr(self._rigid_prim, "get_world_pose"):
                position, orientation = self._rigid_prim.get_world_pose()
                return np.asarray(position, dtype=np.float32), np.asarray(orientation, dtype=np.float32)
            if hasattr(self._rigid_prim, "get_world_poses"):
                positions, orientations = self._rigid_prim.get_world_poses()
                return np.asarray(positions[0], dtype=np.float32), np.asarray(orientations[0], dtype=np.float32)
        except Exception:
            return None
        return None

    def _create_rigid_view(self, prim_path: str, world: Any | None) -> Any | None:
        try:
            try:
                from isaacsim.core.prims import RigidPrim
            except ImportError:
                from omni.isaac.core.prims import RigidPrim
            rigid_view = RigidPrim(prim_paths_expr=prim_path, name="lunar_rocket_body")
            if world is not None and getattr(world, "scene", None) is not None:
                try:
                    return world.scene.add(rigid_view)
                except Exception:
                    pass
            return rigid_view
        except Exception as exc:
            print(f"[LunarRocket] warning: could not create rigid body view for thrust control: {exc}", flush=True)
            return None

    def _set_rigid_world_pose(
        self,
        position: tuple[float, float, float],
        quaternion_wxyz: tuple[float, float, float, float],
    ) -> None:
        assert self._rigid_prim is not None
        if hasattr(self._rigid_prim, "set_world_pose"):
            self._rigid_prim.set_world_pose(position=np.array(position), orientation=np.array(quaternion_wxyz))
        elif hasattr(self._rigid_prim, "set_world_poses"):
            self._rigid_prim.set_world_poses(
                positions=np.asarray([position], dtype=np.float32),
                orientations=np.asarray([quaternion_wxyz], dtype=np.float32),
            )

    def _set_rigid_velocities(self, linear: np.ndarray, angular: np.ndarray) -> None:
        assert self._rigid_prim is not None
        if hasattr(self._rigid_prim, "set_linear_velocity"):
            self._rigid_prim.set_linear_velocity(linear)
        elif hasattr(self._rigid_prim, "set_linear_velocities"):
            self._rigid_prim.set_linear_velocities(np.asarray([linear], dtype=np.float32))
        if hasattr(self._rigid_prim, "set_angular_velocity"):
            self._rigid_prim.set_angular_velocity(angular)
        elif hasattr(self._rigid_prim, "set_angular_velocities"):
            self._rigid_prim.set_angular_velocities(np.asarray([angular], dtype=np.float32))

    def _convert_mjcf_to_usd(self) -> Path:
        try:
            import omni.kit.commands
            import omni.kit.app
        except ImportError as exc:
            raise RuntimeError("MJCF conversion must run inside Isaac Sim Python") from exc

        manager = omni.kit.app.get_app().get_extension_manager()
        for extension in ("isaacsim.asset.importer.mjcf", "omni.importer.mjcf"):
            try:
                manager.set_extension_enabled_immediate(extension, True)
                break
            except Exception:
                continue

        try:
            from isaacsim.asset.importer.mjcf import MJCFImporter, MJCFImporterConfig

            config = MJCFImporterConfig(
                mjcf_path=str(self.mjcf_path),
                usd_path=str(self.usd_path.parent),
                import_scene=False,
                merge_mesh=False,
                collision_from_visuals=False,
                allow_self_collision=False,
                fix_base=False,
                run_asset_transformer=False,
                run_multi_physics_conversion=True,
            )
            output_path = Path(MJCFImporter(config).import_mjcf())
            if output_path != self.usd_path:
                shutil.copyfile(output_path, self.usd_path)
                return self.usd_path
            return output_path
        except ImportError:
            pass

        status, import_config = omni.kit.commands.execute("MJCFCreateImportConfig")
        if not status:
            raise RuntimeError("Failed to create MJCF import config")

        for attr, value in {
            "set_fix_base": False,
            "set_import_sites": True,
            "set_make_default_prim": True,
            "set_create_physics_scene": False,
        }.items():
            if hasattr(import_config, attr):
                getattr(import_config, attr)(value)

        command_variants = [
            {
                "mjcf_path": str(self.mjcf_path),
                "import_config": import_config,
                "dest_path": str(self.usd_path),
            },
            {
                "mjcf_path": str(self.mjcf_path),
                "import_config": import_config,
                "usd_path": str(self.usd_path),
            },
        ]
        last_error: Exception | None = None
        for kwargs in command_variants:
            try:
                status, _ = omni.kit.commands.execute("MJCFCreateAsset", **kwargs)
                if status:
                    return self.usd_path
            except Exception as exc:
                last_error = exc
        raise RuntimeError(f"Failed to convert MJCF to USD: {last_error}")

    def _set_xform_pose(
        self,
        prim: Any,
        position: tuple[float, float, float],
        euler_deg: tuple[float, float, float],
    ) -> None:
        from pxr import Gf, UsdGeom

        xform = UsdGeom.Xformable(prim)
        translate_attr = prim.GetAttribute("xformOp:translate")
        if translate_attr and translate_attr.IsValid():
            translate_op = UsdGeom.XformOp(translate_attr)
        else:
            translate_op = xform.AddTranslateOp(precision=UsdGeom.XformOp.PrecisionFloat)

        rotate_attr = prim.GetAttribute("xformOp:rotateXYZ")
        if rotate_attr and rotate_attr.IsValid():
            rotate_op = UsdGeom.XformOp(rotate_attr)
        else:
            rotate_op = xform.AddRotateXYZOp(precision=UsdGeom.XformOp.PrecisionFloat)

        scale_attr = prim.GetAttribute("xformOp:scale")
        if scale_attr and scale_attr.IsValid():
            scale_op = UsdGeom.XformOp(scale_attr)
        else:
            scale_op = xform.AddScaleOp(precision=UsdGeom.XformOp.PrecisionFloat)

        translate_op.Set(Gf.Vec3f(*position))
        rotate_op.Set(Gf.Vec3f(*euler_deg))
        scale_op.Set(Gf.Vec3f(self.visual_scale, self.visual_scale, self.visual_scale))
        xform.SetXformOpOrder([translate_op, rotate_op, scale_op])

    def _euler_xyz_to_quat_wxyz(self, euler_deg: tuple[float, float, float]) -> tuple[float, float, float, float]:
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
