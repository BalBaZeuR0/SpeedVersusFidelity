from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import numpy as np
import requests
from PIL import Image

from app.config import resolve_project_path
from app.randomization import ResetSample


@dataclass(frozen=True)
class TerrainMeshData:
    heights: np.ndarray
    vertices: list[tuple[float, float, float]]
    faces: list[int]
    face_counts: list[int]
    size_m: tuple[float, float]
    min_height_m: float
    max_height_m: float
    vertex_count: int = 0
    triangle_count: int = 0
    grid_spacing_m: float = 0.0
    heightmap_resolution: int = 0
    collision_resolution: int = 0
    visual_resolution: int = 0
    mean_slope_deg: float = 0.0
    max_slope_deg: float = 0.0
    local_detail_mesh: TerrainMeshData | None = None


class MoonTerrainGenerator:
    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.dem_config = config.get("dem", {})
        self.cache_dir = resolve_project_path(self.dem_config.get("cache_dir", "assets/moon_heightmaps"))
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.rng = np.random.default_rng(int(config.get("seed", 7)))
        self._dem: np.ndarray | None = None
        self._quality_preset = self._get_quality_preset()

    def _get_quality_preset(self) -> dict[str, Any]:
        """Get the current terrain quality preset based on config."""
        quality_name = str(self.config.get("terrain_quality", "medium_train"))
        presets = self.config.get("terrain_quality_presets", {})
        if not presets:
            resolution = int(self.config.get("mesh_resolution", self.config.get("heightmap_resolution", 128)))
            return {
                "heightmap_resolution": resolution,
                "collision_resolution": resolution,
                "visual_resolution": resolution,
            }
        if quality_name not in presets:
            raise ValueError(f"Unknown terrain quality: {quality_name}. Available: {list(presets.keys())}")
        return presets[quality_name]

    def get_mesh_resolution(self) -> int:
        """Get heightmap/mesh resolution for the current quality preset."""
        return int(self._quality_preset.get("heightmap_resolution", 128))

    def ensure_dem(self) -> Path:
        source_url = str(self.dem_config.get("source_url", "")).strip()
        filename = self.dem_config.get("filename") or Path(urlparse(source_url).path).name
        if not filename:
            raise ValueError("terrain.dem.filename or terrain.dem.source_url must define a filename")

        target = self.cache_dir / filename
        if target.exists() and target.stat().st_size > 0:
            print(f"[LunarRocket] using cached DEM: {target}", flush=True)
            return target
        if not source_url:
            raise FileNotFoundError(f"DEM file missing and no source_url configured: {target}")

        print(f"[LunarRocket] downloading DEM: {source_url}", flush=True)
        response = requests.get(source_url, stream=True, timeout=float(self.dem_config.get("timeout_s", 120.0)))
        response.raise_for_status()
        tmp = target.with_suffix(target.suffix + ".part")
        with tmp.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    handle.write(chunk)
        tmp.replace(target)
        print(f"[LunarRocket] DEM cached: {target}", flush=True)
        return target

    def load_dem(self) -> np.ndarray:
        if self._dem is not None:
            return self._dem

        dem_path = self.ensure_dem()
        if dem_path.suffix.lower() == ".npy":
            array = np.load(dem_path).astype(np.float32)
        else:
            with Image.open(dem_path) as image:
                array = np.asarray(image, dtype=np.float32).copy()

        if array.ndim == 3:
            array = array[..., 0]

        units = self.dem_config.get("units", "kilometers_relative")
        if units == "kilometers_relative":
            array *= 1000.0
        elif units == "half_meter_offset":
            array = (array - float(self.dem_config.get("uint_offset", 20000.0))) * 0.5
        elif units == "meters":
            pass
        else:
            raise ValueError(f"Unsupported DEM units: {units}")

        if bool(self.dem_config.get("remove_patch_mean", True)):
            array -= float(np.nanmean(array))

        self._dem = np.nan_to_num(array, copy=False).astype(np.float32)
        self.config["dem_height_pixels"] = int(self._dem.shape[0])
        self.config["dem_width_pixels"] = int(self._dem.shape[1])
        print(f"[LunarRocket] DEM loaded shape={self._dem.shape}", flush=True)
        return self._dem

    def generate(self, reset: ResetSample | None = None) -> TerrainMeshData:
        dem = self.load_dem()
        rng = np.random.default_rng(reset.seed if reset else int(self.rng.integers(0, np.iinfo(np.int32).max)))
        raw_patch = self._sample_patch(dem, reset)
        
        # Get mesh resolution from quality preset
        mesh_resolution = self.get_mesh_resolution()
        patch = self._resample_patch(raw_patch, mesh_resolution)

        patch = self._condition_dem_patch(patch)
        slope = reset.slope if reset else (0.0, 0.0)
        crater_count = reset.crater_count if reset else int(self.config.get("crater_count_range", [12, 36])[0])
        roughness = reset.roughness_scale if reset else float(self.config.get("roughness_scale_range", [0.1, 0.7])[0])
        patch = self._add_slope(patch, slope)
        patch = self._add_craters(patch, crater_count, rng)
        patch = self._add_roughness(patch, roughness, rng)
        patch = self._smooth_field(patch, int(self.config.get("dem_final_smoothing_passes", 0)))
        mesh_data = self._build_mesh(patch.astype(np.float32))

        local_config = self.config.get("local_detail_patch", {})
        if reset and bool(local_config.get("enabled", False)):
            local_mesh = self.generate_local_detail_mesh(
                raw_patch=raw_patch,
                base_heights=patch,
                center_pos_m=reset.target_position[:2],
                slope=slope,
                crater_count=max(2, crater_count // 2),
                roughness_scale=roughness,
                rng=rng,
            )
            if local_mesh is not None:
                mesh_data = TerrainMeshData(
                    heights=mesh_data.heights,
                    vertices=mesh_data.vertices,
                    faces=mesh_data.faces,
                    face_counts=mesh_data.face_counts,
                    size_m=mesh_data.size_m,
                    min_height_m=min(mesh_data.min_height_m, local_mesh.min_height_m),
                    max_height_m=max(mesh_data.max_height_m, local_mesh.max_height_m),
                    vertex_count=mesh_data.vertex_count,
                    triangle_count=mesh_data.triangle_count,
                    grid_spacing_m=mesh_data.grid_spacing_m,
                    heightmap_resolution=mesh_data.heightmap_resolution,
                    collision_resolution=local_mesh.heightmap_resolution,
                    visual_resolution=local_mesh.heightmap_resolution,
                    mean_slope_deg=mesh_data.mean_slope_deg,
                    max_slope_deg=max(mesh_data.max_slope_deg, local_mesh.max_slope_deg),
                    local_detail_mesh=local_mesh,
                )
        
        return mesh_data

    def create_or_update_usd_mesh(
        self,
        stage: Any,
        prim_path: str,
        mesh_data: TerrainMeshData,
        display_color: tuple[float, float, float] = (0.42, 0.41, 0.38),
        material_suffix: str = "Base",
        collision_enabled: bool = True,
    ) -> Any:
        try:
            from pxr import Gf, Sdf, UsdGeom, UsdPhysics
        except ImportError as exc:
            raise RuntimeError("USD/Isaac Python modules are required to create terrain prims") from exc

        prim = stage.GetPrimAtPath(prim_path)
        is_new = True
        if prim and prim.IsValid():
            if prim.IsA(UsdGeom.Mesh):
                mesh = UsdGeom.Mesh(prim)
                is_new = False
            else:
                stage.RemovePrim(prim_path)
                mesh = UsdGeom.Mesh.Define(stage, prim_path)
        else:
            mesh = UsdGeom.Mesh.Define(stage, prim_path)

        if is_new:
            mesh.CreatePointsAttr(mesh_data.vertices)
            mesh.CreateFaceVertexCountsAttr(mesh_data.face_counts)
            mesh.CreateFaceVertexIndicesAttr(mesh_data.faces)
            mesh.CreateSubdivisionSchemeAttr("none")
            mesh.CreateDisplayColorAttr([Gf.Vec3f(*display_color)])
            mesh.CreateDoubleSidedAttr(True)
            mesh.CreateVisibilityAttr("inherited")
            if collision_enabled:
                UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
                mesh.GetPrim().CreateAttribute("physxCollision:approximation", Sdf.ValueTypeNames.Token).Set("none")
                mesh.GetPrim().CreateAttribute("physxCollision:collisionEnabled", Sdf.ValueTypeNames.Bool).Set(True)
            self._bind_lunar_material(stage, mesh.GetPrim(), display_color, material_suffix)
        else:
            mesh.GetPointsAttr().Set(mesh_data.vertices)

        # Update Extent using NumPy for speed
        vertices_arr = np.asarray(mesh_data.vertices, dtype=np.float32)
        min_vals = vertices_arr.min(axis=0)
        max_vals = vertices_arr.max(axis=0)
        mesh.GetExtentAttr().Set(
            [
                Gf.Vec3f(float(min_vals[0]), float(min_vals[1]), float(min_vals[2])),
                Gf.Vec3f(float(max_vals[0]), float(max_vals[1]), float(max_vals[2])),
            ]
        )

        smooth_normals = bool(self.config.get("smooth_visual_normals", True))
        if material_suffix == "LandingDetail":
            smooth_normals = bool(self.config.get("local_detail_patch", {}).get("smooth_normals", smooth_normals))
        if smooth_normals:
            normals_attr = mesh.GetNormalsAttr()
            normals_list = self._calculate_vertex_normals(mesh_data.heights)
            if not normals_attr.IsValid():
                mesh.CreateNormalsAttr(normals_list)
                mesh.SetNormalsInterpolation("vertex")
            else:
                normals_attr.Set(normals_list)

        return mesh.GetPrim()

    def create_or_update_usd_meshes(self, stage: Any, prim_path: str, mesh_data: TerrainMeshData) -> Any:
        prim = self.create_or_update_usd_mesh(
            stage,
            prim_path,
            mesh_data,
            display_color=(0.24, 0.235, 0.215),
            material_suffix="Base",
            collision_enabled=bool(self.config.get("base_collision_enabled", True)),
        )
        local_path = f"{prim_path}_LandingDetail"
        if mesh_data.local_detail_mesh is not None:
            self.create_or_update_usd_mesh(
                stage,
                local_path,
                mesh_data.local_detail_mesh,
                display_color=(0.26, 0.25, 0.23),
                material_suffix="LandingDetail",
                collision_enabled=bool(self.config.get("local_detail_patch", {}).get("collision_enabled", True)),
            )
        else:
            if stage.GetPrimAtPath(local_path):
                stage.RemovePrim(local_path)
        return prim

    def _bind_lunar_material(
        self,
        stage: Any,
        prim: Any,
        color: tuple[float, float, float],
        suffix: str,
    ) -> None:
        try:
            from pxr import Gf, Sdf, UsdPhysics, UsdShade
            from app.config import resolve_project_path

            material_path = f"/World/Looks/LunarRegolith_{suffix}"
            material = UsdShade.Material.Define(stage, material_path)
            
            # 1. Bind UsdPreviewSurface as a reliable fallback
            shader = UsdShade.Shader.Define(stage, f"{material_path}/PreviewSurface")
            shader.CreateIdAttr("UsdPreviewSurface")
            shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
            shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.92)
            shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
            material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
            
            # 2. Try to bind the OmniPBR MDL material from OmniLRS if available
            mdl_rel = "assets/moon_texture/LunarRegolith8k.mdl"
            mdl_path = resolve_project_path(mdl_rel)
            if mdl_path.exists():
                mdl_shader = UsdShade.Shader.Define(stage, f"{material_path}/Shader")
                mdl_shader.CreateIdAttr("mdlMaterial")
                mdl_shader.SetSourceAsset(Sdf.AssetPath(str(mdl_path)), "mdl")
                mdl_shader.SetSourceAssetSubIdentifier("LunarRegolith8k", "mdl")
                material.CreateSurfaceOutput("mdl").ConnectToSource(mdl_shader.ConnectableAPI(), "out")
            
            # 3. Apply physics material settings
            physics_config = self.config.get("physics_material", {})
            physics_material = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
            physics_material.CreateStaticFrictionAttr(float(physics_config.get("static_friction", 0.85)))
            physics_material.CreateDynamicFrictionAttr(float(physics_config.get("dynamic_friction", 0.65)))
            physics_material.CreateRestitutionAttr(float(physics_config.get("restitution", 0.02)))
            
            UsdShade.MaterialBindingAPI.Apply(prim).Bind(material)
        except Exception as e:
            print(f"[LunarRocket] error binding material: {e}", flush=True)

    def _condition_dem_patch(self, patch: np.ndarray) -> np.ndarray:
        shaped = patch.astype(np.float32, copy=True)
        shaped = self._smooth_field(shaped, int(self.config.get("dem_base_smoothing_passes", 0)))

        if bool(self.config.get("remove_patch_mean", True)):
            shaped -= float(np.mean(shaped))

        shaped *= float(self.config.get("dem_vertical_scale", 0.001))

        clip = self.config.get("dem_percentile_clip", None)
        if isinstance(clip, (list, tuple)) and len(clip) == 2:
            low, high = np.percentile(shaped, [float(clip[0]), float(clip[1])])
            shaped = np.clip(shaped, float(low), float(high))
            shaped -= float(np.mean(shaped))

        max_relief = float(self.config.get("dem_max_relief_m", 0.0))
        if max_relief > 0.0:
            peak = float(np.max(np.abs(shaped)))
            if peak > max_relief:
                shaped *= max_relief / peak

        feature_gain = float(self.config.get("dem_feature_gain", 1.0))
        if feature_gain != 1.0:
            macro = self._smooth_field(shaped, 8)
            detail = shaped - macro
            shaped = macro + detail * feature_gain

        return shaped.astype(np.float32)

    def _smooth_field(self, field: np.ndarray, passes: int) -> np.ndarray:
        smoothed = field.astype(np.float32, copy=True)
        for _ in range(max(0, int(passes))):
            smoothed = (
                smoothed
                + np.roll(smoothed, 1, axis=0)
                + np.roll(smoothed, -1, axis=0)
                + np.roll(smoothed, 1, axis=1)
                + np.roll(smoothed, -1, axis=1)
            ) / 5.0
        return smoothed

    def _sample_patch(self, dem: np.ndarray, reset: ResetSample | None) -> np.ndarray:
        patch_h, patch_w = [int(v) for v in self.config.get("patch_shape_pixels", [256, 256])]
        if patch_h > dem.shape[0] or patch_w > dem.shape[1]:
            raise ValueError(f"Patch {patch_h}x{patch_w} is larger than DEM {dem.shape}")

        if reset:
            y0, x0 = reset.terrain_origin
            y0 = min(max(0, y0), dem.shape[0] - patch_h)
            x0 = min(max(0, x0), dem.shape[1] - patch_w)
        else:
            y0 = int(self.rng.integers(0, dem.shape[0] - patch_h + 1))
            x0 = int(self.rng.integers(0, dem.shape[1] - patch_w + 1))
        return dem[y0 : y0 + patch_h, x0 : x0 + patch_w].copy()

    def _resample_patch(self, patch: np.ndarray, resolution: int) -> np.ndarray:
        if resolution <= 1:
            raise ValueError("mesh_resolution must be > 1")
        y_idx = np.linspace(0, patch.shape[0] - 1, resolution).round().astype(np.int32)
        x_idx = np.linspace(0, patch.shape[1] - 1, resolution).round().astype(np.int32)
        return patch[np.ix_(y_idx, x_idx)]

    def _add_slope(self, patch: np.ndarray, slope: tuple[float, float]) -> np.ndarray:
        size_x, size_y = self._terrain_size()
        y = np.linspace(-size_y / 2.0, size_y / 2.0, patch.shape[0], dtype=np.float32)
        x = np.linspace(-size_x / 2.0, size_x / 2.0, patch.shape[1], dtype=np.float32)
        xx, yy = np.meshgrid(x, y)
        return patch + xx * float(slope[0]) + yy * float(slope[1])

    def _add_craters(self, patch: np.ndarray, count: int, rng: np.random.Generator) -> np.ndarray:
        result = patch.copy()
        radius_min, radius_max = self.config.get("crater_radius_m_range", [2.0, 14.0])
        depth_min, depth_max = self.config.get("crater_depth_m_range", [0.25, 3.0])
        depth_ratio = self.config.get("crater_depth_to_radius_ratio")
        rim_ratio = self.config.get("crater_rim_height_ratio", [0.02, 0.08])
        floor_flattening = float(self.config.get("crater_floor_flattening", 0.0))
        size_x, size_y = self._terrain_size()
        y = np.linspace(-size_y / 2.0, size_y / 2.0, patch.shape[0], dtype=np.float32)
        x = np.linspace(-size_x / 2.0, size_x / 2.0, patch.shape[1], dtype=np.float32)
        xx, yy = np.meshgrid(x, y)

        for _ in range(max(0, count)):
            cx = float(rng.uniform(-size_x / 2.0, size_x / 2.0))
            cy = float(rng.uniform(-size_y / 2.0, size_y / 2.0))
            radius = float(rng.uniform(radius_min, radius_max))
            if isinstance(depth_ratio, (list, tuple)) and len(depth_ratio) == 2:
                depth = radius * float(rng.uniform(float(depth_ratio[0]), float(depth_ratio[1])))
            else:
                depth = float(rng.uniform(depth_min, depth_max))
            rim_height = radius * float(rng.uniform(float(rim_ratio[0]), float(rim_ratio[1])))
            distance = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
            normalized = np.clip(distance / radius, 0.0, 1.0)
            bowl_profile = (1.0 + np.cos(np.pi * normalized)) * 0.5
            if floor_flattening > 0.0:
                floor = np.clip(normalized / max(floor_flattening, 1e-6), 0.0, 1.0)
                bowl_profile *= floor * floor * (3.0 - 2.0 * floor)
            bowl = -depth * bowl_profile
            rim = rim_height * np.exp(-((normalized - 1.04) ** 2) / 0.018)
            mask = distance <= radius * 1.45
            result[mask] += (bowl + rim)[mask]
        return result

    def _add_roughness(self, patch: np.ndarray, scale: float, rng: np.random.Generator) -> np.ndarray:
        low_scale = float(self.config.get("low_frequency_noise_scale", 0.0))
        low_noise = rng.normal(0.0, low_scale, patch.shape).astype(np.float32)
        for _ in range(int(self.config.get("low_frequency_noise_smoothing_passes", 0))):
            low_noise = (
                low_noise
                + np.roll(low_noise, 1, axis=0)
                + np.roll(low_noise, -1, axis=0)
                + np.roll(low_noise, 1, axis=1)
                + np.roll(low_noise, -1, axis=1)
            ) / 5.0

        noise = rng.normal(0.0, float(scale), patch.shape).astype(np.float32)
        for _ in range(int(self.config.get("roughness_smoothing_passes", 2))):
            noise = (
                noise
                + np.roll(noise, 1, axis=0)
                + np.roll(noise, -1, axis=0)
                + np.roll(noise, 1, axis=1)
                + np.roll(noise, -1, axis=1)
            ) / 5.0
        return patch + low_noise + noise

    def _calculate_vertex_normals(self, heights: np.ndarray) -> list[list[float]]:
        rows, cols = heights.shape
        size_x, size_y = self._terrain_size()
        dy, dx = np.gradient(heights, size_y / max(1, rows - 1), size_x / max(1, cols - 1))
        normals = np.dstack((-dx, -dy, np.ones_like(heights, dtype=np.float32)))
        lengths = np.linalg.norm(normals, axis=2, keepdims=True)
        normals /= np.maximum(lengths, 1e-6)
        return normals.reshape(-1, 3).tolist()

    def _build_mesh(
        self,
        heights: np.ndarray,
        size_m: tuple[float, float] | None = None,
        center_m: tuple[float, float] = (0.0, 0.0),
    ) -> TerrainMeshData:
        rows, cols = heights.shape
        size_x, size_y = size_m if size_m is not None else self._terrain_size()
        vertical_scale = float(self.config.get("vertical_scale", 1.0))
        center_x, center_y = center_m
        x_values = np.linspace(center_x - size_x / 2.0, center_x + size_x / 2.0, cols, dtype=np.float32)
        y_values = np.linspace(center_y - size_y / 2.0, center_y + size_y / 2.0, rows, dtype=np.float32)

        scaled = heights * vertical_scale
        
        # Fast vectorized mesh generation using NumPy
        xx, yy = np.meshgrid(x_values, y_values)
        vertices_np = np.stack((xx, yy, scaled), axis=-1).reshape(-1, 3)
        vertices = vertices_np.tolist()

        faces: list[int] = []
        face_counts: list[int] = []
        for row in range(rows - 1):
            for col in range(cols - 1):
                a = row * cols + col
                faces.extend([a, a + 1, a + cols + 1, a + cols])
                face_counts.append(4)

        # Calculate statistics
        vertex_count = len(vertices)
        triangle_count = sum(fc - 2 for fc in face_counts)  # Convert quad/face counts to triangles
        grid_spacing_m = size_x / (cols - 1)  # Spacing between vertices in meters
        
        # Calculate slope statistics
        mean_slope_deg, max_slope_deg = self._calculate_slope_stats(scaled, size_x, size_y)

        return TerrainMeshData(
            heights=scaled,
            vertices=vertices,
            faces=faces,
            face_counts=face_counts,
            size_m=(size_x, size_y),
            min_height_m=float(np.min(scaled)),
            max_height_m=float(np.max(scaled)),
            vertex_count=vertex_count,
            triangle_count=triangle_count,
            grid_spacing_m=grid_spacing_m,
            heightmap_resolution=rows,
            collision_resolution=int(self._quality_preset.get("collision_resolution", rows)),
            visual_resolution=int(self._quality_preset.get("visual_resolution", rows)),
            mean_slope_deg=mean_slope_deg,
            max_slope_deg=max_slope_deg,
        )

    def _calculate_slope_stats(self, heights: np.ndarray, size_x: float, size_y: float) -> tuple[float, float]:
        """Calculate mean and max slope in degrees."""
        # Compute height gradients
        dy, dx = np.gradient(heights, size_y / (heights.shape[0] - 1), size_x / (heights.shape[1] - 1))
        
        # Slope in radians: atan(sqrt(dx^2 + dy^2))
        slope_rad = np.arctan(np.sqrt(dx**2 + dy**2))
        slope_deg = np.degrees(slope_rad)
        
        mean_slope = float(np.mean(slope_deg))
        max_slope = float(np.max(slope_deg))
        
        return mean_slope, max_slope

    def _terrain_size(self) -> tuple[float, float]:
        size = self.config.get("terrain_size_m", [80.0, 80.0])
        if len(size) != 2:
            raise ValueError("terrain_size_m must contain [x, y]")
        return float(size[0]), float(size[1])

    def generate_local_detail_mesh(
        self,
        raw_patch: np.ndarray,
        base_heights: np.ndarray,
        center_pos_m: tuple[float, float],
        slope: tuple[float, float],
        crater_count: int,
        roughness_scale: float,
        rng: np.random.Generator,
    ) -> TerrainMeshData | None:
        local_config = self.config.get("local_detail_patch", {})
        if not bool(local_config.get("enabled", False)):
            return None

        local_size_m = tuple(float(v) for v in local_config.get("patch_size_m", [24.0, 24.0]))
        local_resolution = int(local_config.get("resolution", 512))
        if local_resolution <= base_heights.shape[0]:
            return None

        local_base = self._sample_base_heights_to_local(base_heights, center_pos_m, local_size_m, local_resolution)
        cratered = self._add_local_craters(local_base, crater_count, rng, local_size_m)

        local_noise = self._generate_local_roughness(local_resolution, local_size_m, rng, local_config)
        regolith_grain = self._generate_regolith_grain(local_resolution, local_size_m, rng, local_config)
        micro_features = self._generate_local_micro_features(local_resolution, local_size_m, rng, local_config)
        footprint_features = self._generate_footprint_micro_features(local_resolution, local_size_m, rng, local_config)
        mask = self._local_detail_mask(
            resolution=local_resolution,
            size_m=local_size_m,
            fade_width_m=float(local_config.get("edge_fade_width_m", 4.0)),
        )
        local_patch = local_base + (
            (cratered - local_base) + local_noise + regolith_grain + micro_features + footprint_features
        ) * mask
        local_patch = self._smooth_field(local_patch, int(local_config.get("final_smoothing_passes", 0)))
        local_patch = self._limit_extreme_slopes(
            local_patch,
            local_size_m,
            max_slope_deg=float(local_config.get("max_slope_deg", 45.0)),
            passes=int(local_config.get("slope_limit_passes", 5)),
        )

        # A tiny lift avoids viewport z-fighting where the dense landing mesh overlaps the base mesh.
        local_patch = local_patch + float(local_config.get("surface_lift_m", 0.003)) * mask
        return self._build_mesh(local_patch.astype(np.float32), size_m=local_size_m, center_m=center_pos_m)

    def _local_detail_mask(
        self,
        resolution: int,
        size_m: tuple[float, float],
        fade_width_m: float,
    ) -> np.ndarray:
        x = np.linspace(-size_m[0] / 2.0, size_m[0] / 2.0, resolution, dtype=np.float32)
        y = np.linspace(-size_m[1] / 2.0, size_m[1] / 2.0, resolution, dtype=np.float32)
        xx, yy = np.meshgrid(x, y)
        distance_to_edge = np.minimum(size_m[0] / 2.0 - np.abs(xx), size_m[1] / 2.0 - np.abs(yy))
        t = np.clip(distance_to_edge / max(fade_width_m, 1e-6), 0.0, 1.0)
        return (t * t * (3.0 - 2.0 * t)).astype(np.float32)

    def _sample_base_heights_to_local(
        self,
        base_heights: np.ndarray,
        center_pos_m: tuple[float, float],
        local_size_m: tuple[float, float],
        resolution: int,
    ) -> np.ndarray:
        terrain_size = self._terrain_size()
        rows, cols = base_heights.shape
        x_world = np.linspace(
            center_pos_m[0] - local_size_m[0] / 2.0,
            center_pos_m[0] + local_size_m[0] / 2.0,
            resolution,
        )
        y_world = np.linspace(
            center_pos_m[1] - local_size_m[1] / 2.0,
            center_pos_m[1] + local_size_m[1] / 2.0,
            resolution,
        )
        x_values = (x_world + terrain_size[0] / 2.0) / terrain_size[0] * (cols - 1)
        y_values = (y_world + terrain_size[1] / 2.0) / terrain_size[1] * (rows - 1)
        return self._bilinear_sample_array(base_heights, x_values, y_values)

    def _crop_resample_raw_patch(
        self,
        raw_patch: np.ndarray,
        center_pos_m: tuple[float, float],
        local_size_m: tuple[float, float],
        resolution: int,
    ) -> np.ndarray:
        terrain_size = self._terrain_size()
        raw_rows, raw_cols = raw_patch.shape
        center_x_norm = (center_pos_m[0] / terrain_size[0]) + 0.5
        center_y_norm = (center_pos_m[1] / terrain_size[1]) + 0.5
        span_x_norm = local_size_m[0] / terrain_size[0]
        span_y_norm = local_size_m[1] / terrain_size[1]

        x_values = np.linspace(
            (center_x_norm - span_x_norm / 2.0) * (raw_cols - 1),
            (center_x_norm + span_x_norm / 2.0) * (raw_cols - 1),
            resolution,
        )
        y_values = np.linspace(
            (center_y_norm - span_y_norm / 2.0) * (raw_rows - 1),
            (center_y_norm + span_y_norm / 2.0) * (raw_rows - 1),
            resolution,
        )
        return self._bilinear_sample_array(raw_patch, x_values, y_values)

    def _bilinear_sample_array(self, array: np.ndarray, x_values: np.ndarray, y_values: np.ndarray) -> np.ndarray:
        rows, cols = array.shape
        x = np.clip(x_values, 0.0, cols - 1.0)
        y = np.clip(y_values, 0.0, rows - 1.0)
        x0 = np.floor(x).astype(np.int32)
        y0 = np.floor(y).astype(np.int32)
        x1 = np.clip(x0 + 1, 0, cols - 1)
        y1 = np.clip(y0 + 1, 0, rows - 1)
        xf = (x - x0).astype(np.float32)
        yf = (y - y0).astype(np.float32)

        top = array[np.ix_(y0, x0)] * (1.0 - xf)[None, :] + array[np.ix_(y0, x1)] * xf[None, :]
        bottom = array[np.ix_(y1, x0)] * (1.0 - xf)[None, :] + array[np.ix_(y1, x1)] * xf[None, :]
        return (top * (1.0 - yf)[:, None] + bottom * yf[:, None]).astype(np.float32)

    def _match_patch_mean_to_base(
        self,
        local_patch: np.ndarray,
        base_heights: np.ndarray,
        center_pos_m: tuple[float, float],
    ) -> np.ndarray:
        terrain_size = self._terrain_size()
        rows, cols = base_heights.shape
        col = int(np.clip((center_pos_m[0] + terrain_size[0] / 2.0) / terrain_size[0] * (cols - 1), 0, cols - 1))
        row = int(np.clip((center_pos_m[1] + terrain_size[1] / 2.0) / terrain_size[1] * (rows - 1), 0, rows - 1))
        target = float(base_heights[row, col])
        return local_patch + (target - float(np.mean(local_patch)))

    def _add_local_craters(
        self,
        patch: np.ndarray,
        count: int,
        rng: np.random.Generator,
        size_m: tuple[float, float],
    ) -> np.ndarray:
        result = patch.copy()
        radius_min, radius_max = self.config.get("crater_radius_m_range", [0.8, 6.5])
        depth_min, depth_max = self.config.get("crater_depth_m_range", [0.06, 0.38])
        depth_ratio = self.config.get("crater_depth_to_radius_ratio")
        rim_ratio = self.config.get("crater_rim_height_ratio", [0.02, 0.08])
        floor_flattening = float(self.config.get("crater_floor_flattening", 0.0))
        radius_min = min(float(radius_min), min(size_m) * 0.18)
        radius_max = min(float(radius_max), min(size_m) * 0.34)
        y = np.linspace(-size_m[1] / 2.0, size_m[1] / 2.0, patch.shape[0], dtype=np.float32)
        x = np.linspace(-size_m[0] / 2.0, size_m[0] / 2.0, patch.shape[1], dtype=np.float32)
        xx, yy = np.meshgrid(x, y)

        for _ in range(max(0, count)):
            radius = float(rng.uniform(radius_min, max(radius_min, radius_max)))
            cx = float(rng.uniform(-size_m[0] / 2.0 + radius, size_m[0] / 2.0 - radius))
            cy = float(rng.uniform(-size_m[1] / 2.0 + radius, size_m[1] / 2.0 - radius))
            if isinstance(depth_ratio, (list, tuple)) and len(depth_ratio) == 2:
                depth = radius * float(rng.uniform(float(depth_ratio[0]), float(depth_ratio[1]))) * 0.75
            else:
                depth = float(rng.uniform(depth_min, depth_max)) * 0.75
            rim_height = radius * float(rng.uniform(float(rim_ratio[0]), float(rim_ratio[1]))) * 0.65
            distance = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
            normalized = np.clip(distance / radius, 0.0, 1.0)
            bowl_profile = (1.0 + np.cos(np.pi * normalized)) * 0.5
            if floor_flattening > 0.0:
                floor = np.clip(normalized / max(floor_flattening, 1e-6), 0.0, 1.0)
                bowl_profile *= floor * floor * (3.0 - 2.0 * floor)
            bowl = -depth * bowl_profile
            rim = rim_height * np.exp(-((normalized - 1.04) ** 2) / 0.018)
            mask = distance <= radius * 1.35
            result[mask] += (bowl + rim)[mask]
        return result

    def _limit_extreme_slopes(
        self,
        patch: np.ndarray,
        size_m: tuple[float, float],
        max_slope_deg: float,
        passes: int,
    ) -> np.ndarray:
        limited = patch.astype(np.float32, copy=True)
        max_gradient = float(np.tan(np.radians(max_slope_deg)))
        for _ in range(max(0, passes)):
            dy, dx = np.gradient(
                limited,
                size_m[1] / max(1, limited.shape[0] - 1),
                size_m[0] / max(1, limited.shape[1] - 1),
            )
            steep = np.sqrt(dx**2 + dy**2) > max_gradient
            if not np.any(steep):
                break
            smoothed = self._smooth_field(limited, 1)
            limited[steep] = smoothed[steep]
        return limited

    def generate_local_detail_patch(
        self, global_heights: np.ndarray, center_pos_m: tuple[float, float], rng: np.random.Generator
    ) -> np.ndarray:
        """Generate a high-resolution local detail patch around a landing zone.
        
        Two-tier terrain system:
        - Global patch: Lower resolution for surrounding area (e.g. 128x128)
        - Local patch: Higher resolution for landing zone (e.g. 256x256)
        - Result: ~160-200k triangles total with detail where it matters
        
        Args:
            global_heights: The global terrain heightmap (rows, cols)
            center_pos_m: Center position of detail patch in meters (x, y) within terrain
            rng: Random number generator for consistency
            
        Returns:
            Modified heights array with local detail blended in
        """
        local_config = self.config.get("local_detail_patch", {})
        quality_name = str(self.config.get("terrain_quality", "medium_train"))
        
        # Only apply local detail for two-tier terrain systems
        if not (bool(local_config.get("enabled", False)) or quality_name == "high_detail_landing"):
            return global_heights
        
        result = global_heights.copy()
        local_size_m = tuple(float(v) for v in local_config.get("patch_size_m", [30.0, 30.0]))
        local_resolution = int(local_config.get("resolution", 256))
        blend_radius_m = float(local_config.get("blend_radius_m", 8.0))
        blend_scale = float(local_config.get("blend_scale", 2.0))
        
        global_size_m = self._terrain_size()
        rows, cols = result.shape
        
        # Convert center position from world coords to grid indices
        cx_idx = cols / 2.0 + (center_pos_m[0] / global_size_m[0]) * (cols / 2.0)
        cy_idx = rows / 2.0 + (center_pos_m[1] / global_size_m[1]) * (rows / 2.0)
        
        # Generate high-resolution local detail patch
        local_heights = self._generate_local_roughness(
            int(local_resolution), local_size_m, rng, local_config
        )
        
        # Blend into global terrain with multi-scale blending
        blend_idx = int(blend_radius_m / (global_size_m[0] / cols))
        result = self._blend_local_patch_multiscale(
            result, local_heights, cx_idx, cy_idx, blend_idx, blend_scale
        )
        
        return result

    def _generate_local_roughness(
        self, resolution: int, size_m: tuple[float, float], rng: np.random.Generator, config: dict[str, Any] = None
    ) -> np.ndarray:
        """Generate high-frequency roughness for local detail patch.
        
        Creates multi-scale noise for realistic microtopography:
        - Coarse roughness (craters, rocks)
        - Fine roughness (sand grains, dust)
        - Very fine detail (erosion patterns)
        """
        if config is None:
            config = {}
        
        noise_scale = float(config.get("noise_scale", 0.015))
        smoothing_passes = int(config.get("noise_smoothing_passes", 1))
        
        # Multi-scale noise: coarse + medium + fine detail
        noise_coarse = rng.normal(0.0, noise_scale * 1.0, (resolution, resolution)).astype(np.float32)
        noise_medium = rng.normal(0.0, noise_scale * 0.5, (resolution, resolution)).astype(np.float32)
        noise_fine = rng.normal(0.0, noise_scale * 0.2, (resolution, resolution)).astype(np.float32)
        
        # Combine scales: each is less smooth than the previous
        noise = noise_coarse + noise_medium * 0.6 + noise_fine * 0.3
        
        # Light smoothing to preserve detail
        for _ in range(smoothing_passes):
            noise = (
                noise
                + np.roll(noise, 1, axis=0)
                + np.roll(noise, -1, axis=0)
                + np.roll(noise, 1, axis=1)
                + np.roll(noise, -1, axis=1)
            ) / 5.0
        
        return noise

    def _generate_regolith_grain(
        self,
        resolution: int,
        size_m: tuple[float, float],
        rng: np.random.Generator,
        config: dict[str, Any],
    ) -> np.ndarray:
        grain_scale = float(config.get("regolith_grain_scale", 0.0))
        ripple_scale = float(config.get("regolith_ripple_scale", 0.0))
        if grain_scale <= 0.0 and ripple_scale <= 0.0:
            return np.zeros((resolution, resolution), dtype=np.float32)

        grain = rng.normal(0.0, grain_scale, (resolution, resolution)).astype(np.float32)
        for _ in range(int(config.get("regolith_grain_smoothing_passes", 1))):
            grain = self._smooth_field(grain, 1)

        wavelengths = config.get("regolith_ripple_wavelength_m", [0.18, 0.42])
        min_wave, max_wave = float(wavelengths[0]), float(wavelengths[1])
        x = np.linspace(-size_m[0] / 2.0, size_m[0] / 2.0, resolution, dtype=np.float32)
        y = np.linspace(-size_m[1] / 2.0, size_m[1] / 2.0, resolution, dtype=np.float32)
        xx, yy = np.meshgrid(x, y)
        ripples = np.zeros_like(grain)
        for _ in range(4):
            angle = float(rng.uniform(0.0, 2.0 * np.pi))
            wavelength = float(rng.uniform(min_wave, max_wave))
            phase = float(rng.uniform(0.0, 2.0 * np.pi))
            direction = np.cos(angle) * xx + np.sin(angle) * yy
            ripples += np.sin((2.0 * np.pi / max(wavelength, 1e-6)) * direction + phase).astype(np.float32)
        ripples *= ripple_scale / 4.0

        return grain + ripples

    def _generate_local_micro_features(
        self,
        resolution: int,
        size_m: tuple[float, float],
        rng: np.random.Generator,
        config: dict[str, Any],
    ) -> np.ndarray:
        count = int(config.get("micro_feature_count", 0))
        if count <= 0:
            return np.zeros((resolution, resolution), dtype=np.float32)

        radius_min, radius_max = config.get("micro_feature_radius_m_range", [0.035, 0.12])
        height_min, height_max = config.get("micro_feature_height_m_range", [-0.012, 0.028])
        sharpness = float(config.get("micro_feature_sharpness", 2.0))
        pit_probability = float(config.get("micro_pit_probability", 0.5))
        pit_rim_ratio = float(config.get("micro_pit_rim_ratio", 0.25))
        radius_min = float(radius_min)
        radius_max = float(radius_max)
        height_min = float(height_min)
        height_max = float(height_max)

        y = np.linspace(-size_m[1] / 2.0, size_m[1] / 2.0, resolution, dtype=np.float32)
        x = np.linspace(-size_m[0] / 2.0, size_m[0] / 2.0, resolution, dtype=np.float32)
        xx, yy = np.meshgrid(x, y)
        features = np.zeros((resolution, resolution), dtype=np.float32)

        for _ in range(count):
            radius = float(rng.uniform(radius_min, radius_max))
            cx = float(rng.uniform(-size_m[0] / 2.0 + radius, size_m[0] / 2.0 - radius))
            cy = float(rng.uniform(-size_m[1] / 2.0 + radius, size_m[1] / 2.0 - radius))
            distance = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
            normalized = np.clip(distance / max(radius, 1e-6), 0.0, 2.0)
            if rng.random() < pit_probability:
                depth = abs(float(rng.uniform(height_min, min(height_max, 0.0))))
                bowl = -depth * np.exp(-(normalized**2) / max(sharpness, 1e-6))
                rim = depth * pit_rim_ratio * np.exp(-((normalized - 1.05) ** 2) / 0.035)
                features += (bowl + rim).astype(np.float32)
            else:
                height = max(0.0, float(rng.uniform(max(0.0, height_min), height_max)))
                profile = np.exp(-(normalized**2) / max(sharpness, 1e-6))
                features += (height * profile).astype(np.float32)

        return features

    def _generate_footprint_micro_features(
        self,
        resolution: int,
        size_m: tuple[float, float],
        rng: np.random.Generator,
        config: dict[str, Any],
    ) -> np.ndarray:
        count = int(config.get("footprint_micro_feature_count", 0))
        radius_limit = float(config.get("footprint_detail_radius_m", 0.0))
        if count <= 0 or radius_limit <= 0.0:
            return np.zeros((resolution, resolution), dtype=np.float32)

        radius_min, radius_max = config.get("footprint_micro_feature_radius_m_range", [0.015, 0.060])
        height_min, height_max = config.get("footprint_micro_feature_height_m_range", [-0.020, 0.040])
        sharpness = float(config.get("micro_feature_sharpness", 0.55))
        pit_probability = float(config.get("micro_pit_probability", 0.62))
        pit_rim_ratio = float(config.get("micro_pit_rim_ratio", 0.28))

        y = np.linspace(-size_m[1] / 2.0, size_m[1] / 2.0, resolution, dtype=np.float32)
        x = np.linspace(-size_m[0] / 2.0, size_m[0] / 2.0, resolution, dtype=np.float32)
        xx, yy = np.meshgrid(x, y)
        features = np.zeros((resolution, resolution), dtype=np.float32)

        for _ in range(count):
            angle = float(rng.uniform(0.0, 2.0 * np.pi))
            radial = radius_limit * float(np.sqrt(rng.uniform(0.0, 1.0)))
            cx = radial * float(np.cos(angle))
            cy = radial * float(np.sin(angle))
            radius = float(rng.uniform(float(radius_min), float(radius_max)))
            height = float(rng.uniform(float(height_min), float(height_max)))
            distance = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
            normalized = np.clip(distance / max(radius, 1e-6), 0.0, 2.0)
            if height < 0.0 or rng.random() < pit_probability:
                depth = abs(height)
                bowl = -depth * np.exp(-(normalized**2) / max(sharpness, 1e-6))
                rim = depth * pit_rim_ratio * np.exp(-((normalized - 1.05) ** 2) / 0.035)
                features += (bowl + rim).astype(np.float32)
            else:
                profile = np.exp(-(normalized**2) / max(sharpness, 1e-6))
                features += (height * profile).astype(np.float32)

        footprint = np.sqrt(xx**2 + yy**2)
        fade = np.clip((radius_limit - footprint) / max(radius_limit * 0.25, 1e-6), 0.0, 1.0)
        fade = fade * fade * (3.0 - 2.0 * fade)
        return features * fade.astype(np.float32)

    def _blend_local_patch_multiscale(
        self, 
        global_h: np.ndarray, 
        local_h: np.ndarray, 
        cx: float, 
        cy: float, 
        blend_idx: int,
        blend_scale: float = 2.0
    ) -> np.ndarray:
        """Blend local detail patch into global terrain with smooth multi-scale transition.
        
        Uses Gaussian-like falloff for seamless blending:
        - Inner region: 100% local detail
        - Blend zone: Smooth transition (Gaussian falloff)
        - Outer region: 0% local detail (pure global)
        """
        result = global_h.copy()
        
        # Map local patch back to global grid with multi-scale blending
        local_rows, local_cols = local_h.shape
        global_rows, global_cols = global_h.shape
        
        # Process blending with distance-based Gaussian falloff
        blend_range = int(blend_idx * 1.5)  # Slightly larger range for smooth transition
        
        for i in range(max(0, int(cy) - blend_range), min(global_rows, int(cy) + blend_range)):
            for j in range(max(0, int(cx) - blend_range), min(global_cols, int(cx) + blend_range)):
                # Euclidean distance from center in grid cells
                di = i - cy
                dj = j - cx
                dist_grid = np.sqrt(di * di + dj * dj)
                
                # Gaussian-like blend factor
                # blend_idx is the radius, blend_scale controls sharpness
                if blend_idx > 0:
                    # Gaussian: exp(-(dist/sigma)^blend_scale)
                    normalized_dist = dist_grid / blend_idx
                    blend_factor = np.exp(-(normalized_dist ** blend_scale))
                else:
                    blend_factor = 0.0
                
                if blend_factor > 0.001:  # Only blend if factor is significant
                    # Map global grid coordinates to local patch coordinates
                    # The local patch is centered and sized differently
                    local_i = (i - cy + blend_idx) / (2 * blend_idx) * local_rows
                    local_j = (j - cx + blend_idx) / (2 * blend_idx) * local_cols
                    
                    if -0.5 < local_i < local_rows - 0.5 and -0.5 < local_j < local_cols - 0.5:
                        # Clamp to valid range
                        li = max(0, min(local_rows - 2, int(np.floor(local_i))))
                        lj = max(0, min(local_cols - 2, int(np.floor(local_j))))
                        
                        li_frac = np.clip(local_i - li, 0.0, 1.0)
                        lj_frac = np.clip(local_j - lj, 0.0, 1.0)
                        
                        # Bilinear interpolation from local patch
                        if li + 1 < local_rows and lj + 1 < local_cols:
                            local_value = (
                                local_h[li, lj] * (1 - li_frac) * (1 - lj_frac)
                                + local_h[li + 1, lj] * li_frac * (1 - lj_frac)
                                + local_h[li, lj + 1] * (1 - li_frac) * lj_frac
                                + local_h[li + 1, lj + 1] * li_frac * lj_frac
                            )
                            
                            # Apply blended local detail
                            result[i, j] += blend_factor * local_value
        
        return result
