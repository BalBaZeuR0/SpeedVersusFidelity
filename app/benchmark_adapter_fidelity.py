"""Run one open-loop scenario through one world adapter and log the trajectory.

This is the measurement core of the analytic-vs-PhysX speed/accuracy study
(see D:/obsidianVault/IsaacSim_Adapter/01_Plan.md).

Design decisions:

* The adapters are driven **directly** (adapter.reset / adapter.step), not through
  ``LunarLandingIsaacEnv``. That bypasses actuator noise/lag, reward and observation
  building, so the only moving part is the physics integration and the speed number
  is a clean per-step dynamics cost.
* The initial condition is **forced identical** across adapters: both adapters seed
  their RNG from ``sim.seed`` and call ``sample_reset`` first, so the drawn
  ``ResetSample`` matches; on top of that we overwrite pose + velocity right after
  ``reset()`` from an explicit canonical initial state and assert agreement.
* One process = one adapter. The PhysX adapter needs ``SimulationApp`` booted before
  Isaac imports, so we boot it here when ``--adapter physx``.

Output: a Parquet file per (scenario, repeat, adapter) under
``outputs/adapter_fidelity/`` plus a JSON sidecar with run metadata.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import load_configs, resolve_project_path
from app.adapter_fidelity_scenarios import (
    build_action_sequence,
    hover_throttle,
    rollout_scripted_descent,
)


# --------------------------------------------------------------------------- utils


def euler_xyz_deg_to_quat_wxyz(euler_deg) -> np.ndarray:
    roll, pitch, yaw = np.deg2rad(np.asarray(euler_deg, dtype=np.float64))
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
        dtype=np.float64,
    )


def git_commit(root: Path) -> str:
    try:
        import subprocess

        out = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        return out.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


# ----------------------------------------------------------------- adapter builders


def _apply_dt(configs: dict, effective_dt: float) -> None:
    configs["sim"]["physics_dt"] = float(effective_dt)
    configs["sim"]["rendering_dt"] = float(effective_dt)
    configs["sim"]["seed"] = int(configs["sim"].get("seed", 1234))


def build_analytic(configs: dict, effective_dt: float):
    from app.analytic_world_adapter import AnalyticLunarLandingAdapter

    adapter = AnalyticLunarLandingAdapter(
        configs, {"control_dt": float(effective_dt),
                  "max_lidar_distance_m": 30.0}
    )
    adapter.reset()
    return adapter


def build_physx(configs: dict, effective_dt: float, headless: bool):
    # SimulationApp must exist before any isaac import.
    try:
        from isaacsim import SimulationApp
    except ImportError:  # older layout
        from omni.isaac.kit import SimulationApp

    # We never render in this study; disabling RTX avoids the Vulkan init that
    # fails under Docker Desktop / WSL2 and cuts startup + VRAM. PhysX runs on CUDA.
    app = SimulationApp(
        {
            "headless": bool(headless),
            "renderer": "None",
            "disable_viewport_updates": True,
        }
    )
    configs["sim"]["headless"] = bool(headless)
    configs["sim"]["renderer"] = "None"
    configs["sim"]["render_on_step"] = False
    configs["sim"].setdefault("camera", {})["enabled"] = False

    from app.world_adapter import IsaacWorldAdapter
    from app.world_builder import LunarLandingWorld

    world = LunarLandingWorld(configs)
    world.build()
    adapter = IsaacWorldAdapter(world, {"control_dt": float(effective_dt),
                                        "max_lidar_distance_m": 30.0})
    adapter.reset()
    return adapter, app


# ------------------------------------------------------------- initial-condition sync


def read_state(adapter, kind: str):
    """Return (pos[3], vel[3], quat_wxyz[4]) as float64 from either adapter."""
    if kind == "analytic":
        s = adapter.state
        return (
            np.asarray(s.position, dtype=np.float64),
            np.asarray(s.velocity, dtype=np.float64),
            np.asarray(s.quaternion_wxyz, dtype=np.float64),
        )
    st = adapter._build_state(np.zeros(3, dtype=np.float32))
    return (
        np.asarray(st.position, dtype=np.float64),
        np.asarray(st.linear_velocity, dtype=np.float64),
        np.asarray(st.quaternion_wxyz, dtype=np.float64),
    )


def _force_reset_sample(adapter, kind: str, rs_dict: dict) -> None:
    """Overwrite the adapter's drawn ResetSample with the shared canonical one."""
    from dataclasses import replace as _dc_replace

    from app.randomization import ResetSample

    shared = ResetSample(
        seed=int(rs_dict["seed"]),
        terrain_origin=tuple(rs_dict["terrain_origin"]),
        rocket_position=tuple(rs_dict["rocket_position"]),
        target_position=tuple(rs_dict["target_position"]),
        rocket_euler_deg=tuple(rs_dict["rocket_euler_deg"]),
        crater_count=int(rs_dict["crater_count"]),
        slope=tuple(rs_dict["slope"]),
        roughness_scale=float(rs_dict["roughness_scale"]),
    )
    if kind == "analytic":
        adapter.reset_sample = shared
    else:
        adapter.env.state = _dc_replace(adapter.env.state, reset_sample=shared)


def force_initial_condition(adapter, kind: str, pos, euler_deg) -> None:
    pos = np.asarray(pos, dtype=np.float64)
    quat = euler_xyz_deg_to_quat_wxyz(euler_deg)
    if kind == "analytic":
        adapter.state.position = pos.astype(np.float32)
        adapter.state.velocity = np.zeros(3, dtype=np.float32)
        adapter.state.acceleration = np.zeros(3, dtype=np.float32)
        adapter.state.quaternion_wxyz = quat.astype(np.float32)
        adapter.state.angular_velocity = np.zeros(3, dtype=np.float32)
        adapter.state.previous_action = np.zeros(3, dtype=np.float32)
        return
    # PhysX: reset the rigid body pose + zero velocities, then rebuild adapter memory.
    adapter.env.rocket.reset_pose(tuple(float(v) for v in pos),
                                  tuple(float(v) for v in euler_deg), log=False)
    if hasattr(adapter.env, "world") and adapter.env.world is not None:
        adapter.env.world.step(render=False)  # settle the kinematic write
    st = adapter._build_state(np.zeros(3, dtype=np.float32))
    from app.world_adapter import AdapterMemory

    adapter.memory = AdapterMemory(
        previous_position=np.asarray(st.position, dtype=np.float32).copy(),
        previous_velocity=np.zeros(3, dtype=np.float32),
        previous_time_s=0.0,
        previous_action=np.zeros(3, dtype=np.float32),
    )


# ----------------------------------------------------------------------- run loop


def emit_init(args: argparse.Namespace) -> int:
    """Build the analytic adapter (terrain-aware, cheap) and dump its post-reset pose.

    This canonical initial condition is then forced onto *both* adapters via
    ``--init-json`` so the two runs start bit-for-bit from the same state.
    """
    configs = load_configs(args.config_dir)
    _apply_dt(configs, args.effective_dt)
    configs["sim"]["seed"] = int(args.seed)
    adapter = build_analytic(configs, args.effective_dt)
    pos, _vel, _quat = read_state(adapter, "analytic")
    rs = adapter.reset_sample
    # The repo draws initial heading uniformly in [-180, 180]. The analytic model
    # carries no heading state (its body is always thrust-aligned, yaw = 0), so a
    # large initial yaw inflates the divergence for a reason unrelated to the
    # translational / attitude-dynamics fidelity we want to isolate. Clamp it to a
    # heading-aligned approach; yaw sensitivity is studied separately.
    euler = list(rs.rocket_euler_deg)
    if args.init_yaw_deg is not None:
        euler[2] = float(args.init_yaw_deg)          # exact heading (for the sweep)
    else:
        ymax = float(args.init_yaw_clamp_deg)
        if ymax >= 0.0:
            euler[2] = float(np.clip(euler[2], -ymax, ymax))
    rs_dict = _reset_sample_dict(rs)
    rs_dict["rocket_euler_deg"] = list(euler)
    payload = {
        "position": pos.tolist(),
        "euler_deg": euler,
        "seed": int(args.seed),
        "reset_sample": rs_dict,
    }
    out = Path(args.emit_init)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    print(f"[fidelity] wrote canonical init -> {out} pos={pos.tolist()}", flush=True)
    return 0


def _build_analytic_at_init(args: argparse.Namespace):
    """Analytic adapter, reset and forced onto the canonical initial condition."""
    configs = load_configs(args.config_dir)
    _apply_dt(configs, args.effective_dt)
    configs["sim"]["seed"] = int(args.seed)
    adapter = build_analytic(configs, args.effective_dt)
    if args.init_json:
        payload = json.loads(Path(args.init_json).read_text())
        init_pos = np.asarray(payload["position"], dtype=np.float64)
        init_euler = np.asarray(payload["euler_deg"], dtype=np.float64)
        if payload.get("reset_sample"):
            _force_reset_sample(adapter, "analytic", payload["reset_sample"])
    else:
        rs = adapter.reset_sample
        init_pos = np.asarray(rs.rocket_position, dtype=np.float64)
        init_euler = np.asarray(rs.rocket_euler_deg, dtype=np.float64)
    force_initial_condition(adapter, "analytic", init_pos, init_euler)
    return adapter, configs


def emit_actions(args: argparse.Namespace) -> int:
    """Fly the scripted PD descent on the analytic adapter and dump the applied actions.

    The resulting open-loop sequence is what scenario S6 replays through *both*
    adapters (see adapter_fidelity_scenarios.rollout_scripted_descent).
    """
    adapter, configs = _build_analytic_at_init(args)
    horizon = int(args.horizon)
    actions = rollout_scripted_descent(adapter, configs, horizon)
    tgt = np.asarray(adapter.reset_sample.target_position, dtype=np.float64)[:2]
    out = Path(args.emit_actions)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "actions": np.asarray(actions, dtype=np.float64).tolist(),
        "scenario": "s6_scripted_descent",
        "horizon": horizon,
        "seed": int(args.seed),
        "target_xy": tgt.tolist(),
    }, indent=2))
    arr = np.asarray(actions, dtype=np.float64)
    nonzero = np.flatnonzero(np.any(arr != 0.0, axis=1))
    last_active = int(nonzero[-1]) + 1 if nonzero.size else 0
    burning = arr[:, 0] > 1e-3
    print(f"[fidelity] wrote scripted-descent actions -> {out} "
          f"(last active step {last_active}/{horizon}, {int(burning.sum())} burn steps, "
          f"throttle max {float(arr[:, 0].max()):.3f}, |gimbal| max {float(np.abs(arr[:, 1:]).max()):.3f})",
          flush=True)
    return 0


def _current_state(adapter, kind: str):
    """Full RocketKinematicState for the controller, without advancing the sim."""
    if kind == "analytic":
        return adapter._outputs(np.zeros(3, dtype=np.float32))[0]
    return adapter._build_state(np.zeros(3, dtype=np.float32))


def run(args: argparse.Namespace) -> int:
    root = resolve_project_path(".")
    configs = load_configs(args.config_dir)
    _apply_dt(configs, args.effective_dt)
    configs["sim"]["seed"] = int(args.seed)
    if args.no_six_dof:
        configs["rocket"].setdefault("thrust", {})["six_dof"] = False

    horizon = int(args.horizon)
    closed_loop = bool(args.closed_loop)

    controller = None
    init_tilt_deg = np.zeros(3)
    if closed_loop:
        from app.adapter_fidelity_controller import (
            CLOSED_LOOP_INIT_TILT_DEG,
            CLOSED_LOOP_SCENARIOS,
            DescentController,
        )

        if args.scenario not in CLOSED_LOOP_SCENARIOS:
            raise KeyError(f"unknown closed-loop scenario {args.scenario!r}; "
                           f"known: {sorted(CLOSED_LOOP_SCENARIOS)}")
        # PhysX gets the over-damped PD inner loop; the analytic model (and the
        # six_dof=false ablation, which has no rotational dynamics to damp) get the
        # feed-forward pointing law.
        inner_law = "pd" if (args.adapter == "physx" and not args.no_six_dof) else "ff"
        controller = DescentController(configs, CLOSED_LOOP_SCENARIOS[args.scenario],
                                       inner_law=inner_law)
        init_tilt_deg = np.asarray(CLOSED_LOOP_INIT_TILT_DEG.get(args.scenario, (0.0, 0.0, 0.0)))
        actions = np.zeros((horizon, 3), dtype=np.float32)  # filled in the loop
    else:
        actions = build_action_sequence(
            args.scenario, horizon, configs,
            seed=int(args.action_seed), actions_json=args.actions_json,
        )

    kind = "analytic" if args.adapter == "analytic" else "physx"
    app = None
    if kind == "analytic":
        adapter = build_analytic(configs, args.effective_dt)
    else:
        adapter, app = build_physx(configs, args.effective_dt, headless=not args.gui)

    # canonical initial condition: derived from the (shared) ResetSample unless overridden
    if args.init_json:
        payload = json.loads(Path(args.init_json).read_text())
        init_pos = np.asarray(payload["position"], dtype=np.float64)
        init_euler = np.asarray(payload["euler_deg"], dtype=np.float64)
        # both adapters redraw their own ResetSample on reset() -> the target
        # position and euler can diverge. Force the canonical one onto the adapter
        # so target_position / bounds are identical too (terrain is already shared).
        if payload.get("reset_sample"):
            _force_reset_sample(adapter, kind, payload["reset_sample"])
    else:
        rs = getattr(adapter, "reset_sample", None) or getattr(adapter.env.state, "reset_sample", None)
        init_pos = np.asarray(rs.rocket_position, dtype=np.float64)
        init_euler = np.asarray(rs.rocket_euler_deg, dtype=np.float64)

    rs = getattr(adapter, "reset_sample", None) or getattr(adapter.env.state, "reset_sample", None)

    pos0, vel0, quat0 = read_state(adapter, kind)
    force_initial_condition(adapter, kind, init_pos + 0.0, init_euler + init_tilt_deg)
    pos1, vel1, quat1 = read_state(adapter, kind)

    rows = np.zeros(
        horizon,
        dtype=[
            ("step", "i4"), ("t", "f8"),
            ("px", "f8"), ("py", "f8"), ("pz", "f8"),
            ("vx", "f8"), ("vy", "f8"), ("vz", "f8"),
            ("qw", "f8"), ("qx", "f8"), ("qy", "f8"), ("qz", "f8"),
            ("wx", "f8"), ("wy", "f8"), ("wz", "f8"),
            ("altitude", "f8"), ("leg_contact", "i4"),
            ("throttle", "f8"), ("gimbal_x", "f8"), ("gimbal_y", "f8"),
            ("wall_ns", "i8"),
        ],
    )

    warmup = int(args.warmup)
    state = _current_state(adapter, kind) if closed_loop else None
    t_wall0 = time.perf_counter_ns()
    for i in range(horizon):
        if closed_loop:
            a = np.asarray(controller(state), dtype=np.float32)
            actions[i] = a
        else:
            a = actions[i]
        c0 = time.perf_counter_ns()
        state, _sensors, _terrain, contact = adapter.step(a.astype(np.float32))
        c1 = time.perf_counter_ns()
        pos = np.asarray(state.position, dtype=np.float64)
        vel = np.asarray(state.linear_velocity, dtype=np.float64)
        quat = np.asarray(state.quaternion_wxyz, dtype=np.float64)
        omega = np.asarray(state.angular_velocity, dtype=np.float64)
        rows[i] = (
            i, (i + 1) * args.effective_dt,
            pos[0], pos[1], pos[2], vel[0], vel[1], vel[2],
            quat[0], quat[1], quat[2], quat[3],
            omega[0], omega[1], omega[2],
            float(state.altitude_to_ground),
            int(bool(np.any(contact.leg_contact))),
            float(a[0]), float(a[1]), float(a[2]),
            c1 - c0,
        )
    t_wall1 = time.perf_counter_ns()

    steady = rows[warmup:]
    total_s = (t_wall1 - t_wall0) / 1e9
    step_ns_steady = steady["wall_ns"].astype(np.float64)
    meta = {
        "scenario": args.scenario,
        "adapter": kind,
        "closed_loop": closed_loop,
        "repeat": int(args.repeat),
        "seed": int(args.seed),
        "action_seed": int(args.action_seed),
        "horizon": horizon,
        "warmup": warmup,
        "effective_dt": float(args.effective_dt),
        "hover_throttle": hover_throttle(configs),
        "git_commit": git_commit(root),
        "isaac_sim": _isaac_version(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "reset_sample": _reset_sample_dict(rs),
        "init_condition": {"position": init_pos.tolist(), "euler_deg": init_euler.tolist()},
        "init_sync_error": {
            "pos_before": float(np.linalg.norm(pos1 - pos0)),
            "pos_after_force": float(np.linalg.norm(pos1 - init_pos)),
        },
        "speed": {
            "wall_clock_total_s": total_s,
            "step_per_s_overall": horizon / total_s,
            "step_per_s_steady_median": 1e9 / float(np.median(step_ns_steady)),
            "step_per_s_steady_iqr": [
                1e9 / float(np.percentile(step_ns_steady, 75)),
                1e9 / float(np.percentile(step_ns_steady, 25)),
            ],
            "rt_factor_steady_median": (1e9 / float(np.median(step_ns_steady))) * float(args.effective_dt),
        },
    }

    out_dir = resolve_project_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{args.scenario}__r{int(args.repeat):02d}__{kind}"
    # .npz keeps the in-container side dependency-free (Isaac ships numpy; no pandas/pyarrow).
    np.savez_compressed(out_dir / f"{stem}.npz", rows=rows)
    (out_dir / f"{stem}.json").write_text(json.dumps(meta, indent=2))
    print(f"[fidelity] {stem}: {meta['speed']['step_per_s_steady_median']:.1f} step/s "
          f"(rt x{meta['speed']['rt_factor_steady_median']:.2f}), "
          f"init_sync_err={meta['init_sync_error']['pos_after_force']:.5f} m "
          f"(moved {meta['init_sync_error']['pos_before']:.2f} m to align)", flush=True)

    if app is not None and not args.skip_close:
        try:
            app.close()
        except Exception:
            pass
    return 0


def _isaac_version() -> str:
    try:
        import isaacsim  # type: ignore

        return getattr(isaacsim, "__version__", "unknown")
    except Exception:
        return "n/a"


def _reset_sample_dict(rs) -> dict:
    try:
        return {k: (list(v) if isinstance(v, tuple) else v) for k, v in asdict(rs).items()}
    except Exception:
        return {}


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--adapter", choices=("analytic", "physx"), default="analytic")
    p.add_argument("--scenario", default="s1_hover")
    p.add_argument("--repeat", type=int, default=0)
    p.add_argument("--seed", type=int, default=1234, help="sim.seed -> drives the shared ResetSample")
    p.add_argument("--action-seed", type=int, default=0, help="seed for stochastic scenarios (S5)")
    p.add_argument("--horizon", type=int, default=1000)
    p.add_argument("--warmup", type=int, default=50)
    p.add_argument("--effective-dt", type=float, default=1.0 / 60.0)
    p.add_argument("--config-dir", default="configs")
    p.add_argument("--out-dir", default="outputs/adapter_fidelity")
    p.add_argument("--init-json", default=None, help="force initial condition from {position, euler_deg}")
    p.add_argument("--emit-init", default=None,
                   help="instead of running: write the canonical init JSON for --seed and exit")
    p.add_argument("--init-yaw-clamp-deg", type=float, default=30.0,
                   help="clamp the drawn initial heading to +/- this (deg) in emit-init; "
                        "negative disables (keep the repo's uniform [-180,180])")
    p.add_argument("--init-yaw-deg", type=float, default=None,
                   help="override the initial heading to exactly this (deg) in emit-init "
                        "(for the heading sweep); overrides --init-yaw-clamp-deg")
    p.add_argument("--no-six-dof", action="store_true",
                   help="ablation: force thrust.six_dof=false (legacy CoM world-frame point force)")
    p.add_argument("--emit-actions", default=None,
                   help="instead of running: fly the scripted PD descent (analytic) and write "
                        "the S6 replay action sequence to this path, then exit")
    p.add_argument("--actions-json", default=None,
                   help="replay a pre-emitted action sequence (required for scenario s6_scripted_descent)")
    p.add_argument("--closed-loop", action="store_true",
                   help="run the shared DescentController in the loop (scenario names a mission variant "
                        "from adapter_fidelity_controller.CLOSED_LOOP_SCENARIOS) instead of replaying actions")
    p.add_argument("--gui", action="store_true", help="physx: run with a window (debug only)")
    p.add_argument("--skip-close", action="store_true")
    return p.parse_args(argv)


if __name__ == "__main__":
    _args = parse_args()
    if _args.emit_init:
        raise SystemExit(emit_init(_args))
    if _args.emit_actions:
        raise SystemExit(emit_actions(_args))
    raise SystemExit(run(_args))
