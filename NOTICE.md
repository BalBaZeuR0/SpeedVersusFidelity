# Third-party code

The simulator layer in `app/` and `configs/` comes from
[LunarRocket](https://github.com/GulogluHaktan/LunarRocket) at commit `9af2744` (MIT License,
Copyright (c) 2026 Cem Özkurt, Muhammed Emin Onay, Eyüp Altuğ Tunç, Ahmet Kutay Küçükler,
Haktan Yusuf Güloğlu). Only the modules this study imports are included.

Unchanged from LunarRocket `9af2744`: `app/config.py`, `app/randomization.py`, `app/rl_types.py`,
`app/sensors.py`, `app/terrain_generator.py`, `configs/sim_config.yaml`, `configs/terrain_config.yaml`,
`configs/rl_config.yaml`, `assets/rocket/hopper_lunar.xml` (MuJoCo model of the 1.63 kg
gimbal-TVC hopper; converted to USD on first PhysX run).

Modified for this study: `app/rocket_asset.py` (6-DOF thrust: body-frame force + gimbal moment,
ported from LunarRocket's Isaac Lab task), `app/world_adapter.py` (reads real linear/angular velocity),
`app/world_builder.py`, `app/analytic_world_adapter.py` (kinematic angular rate),
`configs/rocket_config.yaml` (`thrust.six_dof`, `thrust.lever_arm_m`).

New in this repository: `app/adapter_fidelity_*.py`, `app/benchmark_adapter_fidelity.py`,
`app/run_adapter_fidelity_matrix.py`, `analysis/`, `scripts/`, `tests/`.
