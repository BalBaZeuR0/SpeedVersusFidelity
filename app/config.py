from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def resolve_project_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return (PROJECT_ROOT / candidate).resolve()


def load_yaml(path: str | Path) -> dict[str, Any]:
    config_path = resolve_project_path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}

    if not isinstance(data, dict):
        raise ValueError(f"Config must be a YAML mapping: {config_path}")
    return data


def load_configs(config_dir: str | Path = "configs") -> dict[str, dict[str, Any]]:
    base = resolve_project_path(config_dir)
    configs = {
        "sim": load_yaml(base / "sim_config.yaml"),
        "terrain": load_yaml(base / "terrain_config.yaml"),
        "rocket": load_yaml(base / "rocket_config.yaml"),
    }
    rl_path = base / "rl_config.yaml"
    if rl_path.exists():
        configs["rl"] = load_yaml(rl_path)
    return configs
