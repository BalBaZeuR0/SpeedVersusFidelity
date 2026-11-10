"""Open-loop action-sequence generators for the analytic-vs-PhysX fidelity study.

Each scenario returns a deterministic ``(horizon, 3)`` float32 array of actions
``[throttle in [0,1], gimbal_x in [-1,1], gimbal_y in [-1,1]]``. The same array is
replayed through both adapters from an identical initial condition, so the only
thing that differs between the two runs is the physics integration.

Gimbal magnitudes are deliberately small (order 1 deg): with the repo's 6-DOF
thrust model the vehicle's attitude is an *undamped double integrator* in vacuum,
so a sustained gimbal command diverges without bound. The suite spans zero gimbal
(S1, S2), brief net-zero pulses (S3), a sustained deflection (S3b, the divergence
case), a realistic descent (S4, S6) and small broadband noise (S5).

See ``D:/obsidianVault/IsaacSim_Adapter/01_Plan.md`` section 3.2.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import numpy as np

ActionSequence = np.ndarray  # shape (horizon, 3), float32
ScenarioFn = Callable[[int, dict[str, Any], np.random.Generator], ActionSequence]

# S6 is not a pure open-loop generator: its action sequence is produced by flying a
# scripted PD descent/divert controller through the analytic adapter (which needs the
# real initial condition + terrain), then replayed open-loop through both adapters.
# ``benchmark_adapter_fidelity.py --emit-actions`` writes the file; here we only load it.
S6_NAME = "s6_scripted_descent"


def hover_throttle(configs: dict[str, dict[str, Any]]) -> float:
    """Throttle that exactly cancels lunar gravity for the configured rocket."""
    mass = float(configs["rocket"].get("mass_kg", 1.63))
    g = float(configs["sim"].get("gravity_mps2", 1.62))
    f_max = float(configs["rocket"].get("thrust", {}).get("max_newtons", 30.0))
    return float(np.clip(mass * g / max(f_max, 1e-6), 0.0, 1.0))


def _empty(horizon: int) -> ActionSequence:
    return np.zeros((horizon, 3), dtype=np.float32)


def s1_hover(horizon: int, configs: dict[str, Any], rng: np.random.Generator) -> ActionSequence:
    """S1 - constant hover throttle, zero gimbal. Pure integration-error floor."""
    actions = _empty(horizon)
    actions[:, 0] = hover_throttle(configs)
    return actions


def s2_freefall_then_burn(horizon: int, configs: dict[str, Any], rng: np.random.Generator) -> ActionSequence:
    """S2 - coast with the engine off, then hold a constant deceleration burn."""
    actions = _empty(horizon)
    switch = horizon // 3
    actions[switch:, 0] = min(1.0, 1.6 * hover_throttle(configs))
    return actions


def s3_gimbal_doublet(horizon: int, configs: dict[str, Any], rng: np.random.Generator) -> ActionSequence:
    """S3 - hover throttle with a brief +/- gimbal_x doublet (net-zero attitude).

    Two short opposite pulses (~1.5 deg) that cancel: the commanded attitude
    returns to zero, but the analytic model tracks the gimbal instantly whereas
    PhysX carries the rate history, so the two trajectories separate by a bounded,
    measurable amount.
    """
    actions = _empty(horizon)
    actions[:, 0] = hover_throttle(configs)
    p = max(1, horizon // 12)
    a = horizon // 4
    actions[a : a + p, 1] = 0.12
    actions[a + p : a + 2 * p, 1] = -0.12
    return actions


def s3b_gimbal_hold(horizon: int, configs: dict[str, Any], rng: np.random.Generator) -> ActionSequence:
    """S3b - hover throttle with a *sustained* small gimbal deflection, then release.

    The divergence case. In vacuum the vehicle attitude is an undamped double
    integrator, so the sustained ~1 deg deflection spins PhysX up continuously and
    it keeps rotating after release; the analytic model just holds the body ~1 deg
    off vertical while gimballed and snaps back to zero on release. Position and
    tilt error grow without bound - this is the trajectory the analytic adapter
    cannot represent.
    """
    actions = _empty(horizon)
    actions[:, 0] = hover_throttle(configs)
    hold_end = int(0.12 * horizon)
    actions[:hold_end, 1] = 0.06
    return actions


def s4_landing_profile(horizon: int, configs: dict[str, Any], rng: np.random.Generator) -> ActionSequence:
    """S4 - piecewise throttle schedule for a soft descent (end-to-end task)."""
    actions = _empty(horizon)
    hov = hover_throttle(configs)
    segments = [
        (0.00, 0.25, 0.35 * hov),   # let it fall
        (0.25, 0.55, 1.25 * hov),   # arrest descent
        (0.55, 0.85, 0.95 * hov),   # controlled sink
        (0.85, 1.00, 1.15 * hov),   # final flare
    ]
    for lo, hi, thr in segments:
        a, b = int(lo * horizon), int(hi * horizon)
        actions[a:b, 0] = min(1.0, thr)
    # gentle lateral trim so the trajectory is not perfectly axial (~0.5 deg)
    actions[:, 1] = 0.04 * np.sin(np.linspace(0.0, 3.0 * np.pi, horizon, dtype=np.float32))
    return actions


def s5_fixed_random(horizon: int, configs: dict[str, Any], rng: np.random.Generator) -> ActionSequence:
    """S5 - zero-order-hold random commands from a fixed RNG. Neutral broadband excitation."""
    hold = 20
    actions = _empty(horizon)
    hov = hover_throttle(configs)
    n_holds = (horizon + hold - 1) // hold
    for k in range(n_holds):
        thr = float(np.clip(rng.normal(hov, 0.15), 0.0, 1.0))
        gx = float(np.clip(rng.normal(0.0, 0.03), -1.0, 1.0))
        gy = float(np.clip(rng.normal(0.0, 0.03), -1.0, 1.0))
        actions[k * hold : (k + 1) * hold] = (thr, gx, gy)
    return actions


SCENARIOS: dict[str, ScenarioFn] = {
    "s1_hover": s1_hover,
    "s2_freefall_then_burn": s2_freefall_then_burn,
    "s3_gimbal_doublet": s3_gimbal_doublet,
    "s3b_gimbal_hold": s3b_gimbal_hold,
    "s4_landing_profile": s4_landing_profile,
    "s5_fixed_random": s5_fixed_random,
}

# Full matrix = the pure open-loop generators plus the scripted-descent replay (S6).
ALL_SCENARIOS: list[str] = sorted(SCENARIOS) + [S6_NAME]


def _clip_actions(seq: np.ndarray, horizon: int) -> ActionSequence:
    seq = np.asarray(seq, dtype=np.float32).reshape(-1, 3)
    if seq.shape[0] >= horizon:
        seq = seq[:horizon]
    else:
        seq = np.vstack([seq, np.zeros((horizon - seq.shape[0], 3), dtype=np.float32)])
    seq[:, 0] = np.clip(seq[:, 0], 0.0, 1.0)
    seq[:, 1:] = np.clip(seq[:, 1:], -1.0, 1.0)
    return seq


def build_action_sequence(
    name: str,
    horizon: int,
    configs: dict[str, dict[str, Any]],
    seed: int,
    actions_json: str | Path | None = None,
) -> ActionSequence:
    if name == S6_NAME:
        if not actions_json:
            raise ValueError(
                f"{S6_NAME} needs a pre-emitted PD rollout; run "
                f"`benchmark_adapter_fidelity.py --emit-actions <path>` first and pass --actions-json"
            )
        data = json.loads(Path(actions_json).read_text())
        return _clip_actions(np.asarray(data["actions"], dtype=np.float32), int(horizon))
    if name not in SCENARIOS:
        raise KeyError(f"unknown scenario {name!r}; known: {sorted(SCENARIOS) + [S6_NAME]}")
    rng = np.random.default_rng(seed)
    seq = SCENARIOS[name](int(horizon), configs, rng)
    return _clip_actions(seq, int(horizon))


def rollout_scripted_descent(
    adapter: Any,
    configs: dict[str, dict[str, Any]],
    horizon: int,
    *,
    max_gimbal_cmd: float = 0.03,
) -> ActionSequence:
    """Fly a scripted PD descent+divert on ``adapter`` and return the applied actions.

    Deterministic given the adapter's (already forced) initial state. The controller
    tracks an altitude-scheduled sink rate and steers the horizontal position toward
    the landing target, so the replayed sequence exercises real lateral thrust
    vectoring - the regime where the analytic and PhysX attitude models disagree.

    ``adapter`` is mutated (stepped ``horizon`` times); pass a throwaway instance.
    """
    mass = float(configs["rocket"].get("mass_kg", 1.63))
    g = float(configs["sim"].get("gravity_mps2", 1.62))
    f_max = float(configs["rocket"].get("thrust", {}).get("max_newtons", 30.0))
    max_gimbal_rad = np.deg2rad(float(getattr(adapter, "max_gimbal_deg", 12.0)))
    target_xy = np.asarray(adapter.reset_sample.target_position, dtype=np.float64)[:2]

    actions = np.zeros((int(horizon), 3), dtype=np.float32)
    for i in range(int(horizon)):
        pos = np.asarray(adapter.state.position, dtype=np.float64)
        vel = np.asarray(adapter.state.velocity, dtype=np.float64)
        alt = float(adapter._altitude_to_ground(adapter.state.position))
        if alt <= 0.15:
            break  # touched down - leave the rest of the sequence at zero

        # vertical: track an altitude-scheduled descent rate (fast high up, flare near ground)
        vz_ref = -float(np.clip(0.25 * alt + 0.50, 0.35, 3.0))
        az_cmd = 1.6 * (vz_ref - vel[2])
        throttle = float(np.clip(mass * (g + az_cmd) / max(f_max, 1e-6), 0.0, 1.0))

        # lateral: gentle PD toward the landing target, mapped through the thrust
        # magnitude. Gains + cap are deliberately soft: the vehicle attitude has no
        # damping in vacuum, so this steers slowly and leaves a residual offset.
        if throttle < 1e-3:
            gimbal = np.zeros(2)
        else:
            axy_cmd = 0.012 * (target_xy - pos[:2]) - 0.20 * vel[:2]
            thrust_acc = max(f_max * throttle / mass, 1e-3)
            sin_cmd = np.clip(axy_cmd / thrust_acc, -0.9, 0.9)
            gimbal = np.clip(np.arcsin(sin_cmd) / max_gimbal_rad, -max_gimbal_cmd, max_gimbal_cmd)

        a = np.array([throttle, gimbal[0], gimbal[1]], dtype=np.float32)
        actions[i] = a
        adapter.step(a)

    return _clip_actions(actions, int(horizon))
