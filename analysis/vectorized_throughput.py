"""Vectorised-throughput ceiling of the analytic adapter.

The single-environment speedup (Section 3.1) understates the analytic adapter's
advantage: its dynamics are a handful of elementwise array operations, so N
environments cost almost the same as one. This script measures that ceiling two ways:

  * ``looped``  - the production ``AnalyticLunarLandingAdapter`` stepped N times per
    tick in a Python loop (what you get with no code change; Python-loop bound);
  * ``batched`` - a NumPy reimplementation of the *same* step maths on (N, 3) arrays,
    including a shared-terrain height clamp (the ceiling).

Reported: environment-steps per second vs N, against the single-environment PhysX
rate (~160 step/s in the main matrix). Pure NumPy, no Isaac Sim.

    python analysis/vectorized_throughput.py
"""

from __future__ import annotations

import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

MASS, G, FMAX, DT = 1.63, 1.62, 30.0, 1.0 / 60.0
THETA = np.deg2rad(12.0)
N_STEPS = 1000
WARMUP = 100
N_STEPS_LOOPED = 200          # production-adapter loop is Python-bound; fewer steps suffice
WARMUP_LOOPED = 40
PHYSX_REF = 160.0                     # single-env PhysX step/s, main matrix
OUT_DIR = Path("outputs/adapter_fidelity/paper_figs")
CSV = Path("outputs/adapter_fidelity/analysis/vectorized_throughput.csv")

# shared terrain heightmap (all envs see the same terrain, as in a typical RL batch)
_RNG = np.random.default_rng(0)
_HMAP = _RNG.normal(0.0, 0.6, size=(192, 192)).astype(np.float64)
_HMAP = np.stack([np.convolve(r, np.ones(9) / 9, "same") for r in _HMAP])
_TERR_SIZE = 80.0


def _terrain_h(xy: np.ndarray) -> np.ndarray:
    """Vectorised bilinear height lookup, xy shape (N, 2)."""
    rows, cols = _HMAP.shape
    u = np.clip((xy[:, 0] + _TERR_SIZE / 2) / _TERR_SIZE * (cols - 1), 0, cols - 1)
    v = np.clip((xy[:, 1] + _TERR_SIZE / 2) / _TERR_SIZE * (rows - 1), 0, rows - 1)
    c0 = np.floor(u).astype(int); r0 = np.floor(v).astype(int)
    c1 = np.minimum(c0 + 1, cols - 1); r1 = np.minimum(r0 + 1, rows - 1)
    cf = u - c0; rf = v - r0
    top = _HMAP[r0, c0] * (1 - cf) + _HMAP[r0, c1] * cf
    bot = _HMAP[r1, c0] * (1 - cf) + _HMAP[r1, c1] * cf
    return top * (1 - rf) + bot * rf


def _quat_from_dir(d: np.ndarray) -> np.ndarray:
    """(N,3) unit dirs -> (N,4) wxyz quaternion aligning +z with d."""
    z = np.array([0.0, 0.0, 1.0])
    dot = np.clip(d @ z, -1.0, 1.0)
    axis = np.cross(np.broadcast_to(z, d.shape), d)
    an = np.linalg.norm(axis, axis=1, keepdims=True)
    axis = np.where(an > 1e-9, axis / np.maximum(an, 1e-12), np.array([1.0, 0.0, 0.0]))
    half = np.arccos(dot)[:, None] * 0.5
    return np.concatenate([np.cos(half), axis * np.sin(half)], axis=1)


def batched(n: int) -> float:
    pos = np.tile([0.0, 0.0, 24.0], (n, 1)).astype(np.float64)
    vel = np.zeros((n, 3))
    q_prev = np.tile([1.0, 0.0, 0.0, 0.0], (n, 1)).astype(np.float64)
    # fixed near-hover action, tiny gimbal so the whole path is exercised
    a = np.tile([MASS * G / FMAX, 0.02, -0.015], (n, 1))
    gx, gy, thr = a[:, 1] * THETA, a[:, 2] * THETA, a[:, 0]
    d = np.stack([np.sin(gx), np.sin(gy), np.cos(gx) * np.cos(gy)], axis=1)
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    grav = np.array([0.0, 0.0, -G])
    t0 = None
    for i in range(N_STEPS):
        if i == WARMUP:
            t0 = time.perf_counter()
        acc = grav + d * (FMAX * thr / MASS)[:, None]
        vel += acc * DT
        pos += vel * DT
        q = _quat_from_dir(d)
        # omega finite-diff (relative rotation angle / dt) -- part of the step cost
        rel_w = np.abs(np.sum(q * q_prev, axis=1))
        _ = 2.0 * np.arccos(np.clip(rel_w, -1, 1)) / DT
        q_prev = q
        gh = _terrain_h(pos[:, :2])
        below = pos[:, 2] < gh
        pos[below, 2] = gh[below]
    dt_wall = time.perf_counter() - t0
    return n * (N_STEPS - WARMUP) / dt_wall


def looped(n: int) -> float:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from app.analytic_world_adapter import AnalyticLunarLandingAdapter
    from app.config import load_configs

    cfg = load_configs("configs")
    cfg["sim"]["physics_dt"] = DT
    cfg["sim"]["rendering_dt"] = DT
    adapters = []
    for _ in range(n):
        ad = AnalyticLunarLandingAdapter(cfg, {"control_dt": DT, "max_lidar_distance_m": 30.0})
        ad.reset()
        adapters.append(ad)
    act = np.array([MASS * G / FMAX, 0.02, -0.015], dtype=np.float32)
    t0 = None
    for i in range(N_STEPS_LOOPED):
        if i == WARMUP_LOOPED:
            t0 = time.perf_counter()
        for ad in adapters:
            ad.step(act)
    return n * (N_STEPS_LOOPED - WARMUP_LOOPED) / (time.perf_counter() - t0)


def main() -> None:
    batched_ns = [1, 4, 16, 64, 256, 1024, 4096, 16384, 65536]
    looped_ns = [1, 4, 16, 64]   # only to show the Python-loop path does NOT scale
    rows = []
    print(f"{'N':>7} | {'batched step/s':>16} | {'looped step/s':>14} | {'batched speedup':>15}", flush=True)
    for n in batched_ns:
        b = batched(n)
        lp = looped(n) if n in looped_ns else np.nan
        rows.append((n, b, lp, b / PHYSX_REF))
        print(f"{n:>7} | {b:>16,.0f} | {('' if np.isnan(lp) else f'{lp:,.0f}'):>14} | {b / PHYSX_REF:>14.0f}x", flush=True)

    CSV.parent.mkdir(parents=True, exist_ok=True)
    import csv
    with CSV.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["n_envs", "batched_step_per_s", "looped_step_per_s", "batched_speedup_vs_physx"])
        w.writerows(rows)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    N = np.array([r[0] for r in rows])
    ax.loglog(N, [r[1] for r in rows], "o-", color="#E08214", label="batched analytic core")
    lp = np.array([r[2] for r in rows], dtype=float)
    m = ~np.isnan(lp)
    ax.loglog(N[m], lp[m], "s--", color="#B0722B", label="looped production adapter")
    ax.axhline(PHYSX_REF, color="#2C6FBB", lw=1.2, ls=":", label="single-env PhysX (~160/s)")
    ax.set_xlabel("parallel environments N")
    ax.set_ylabel("environment-steps per second")
    ax.grid(alpha=0.25, which="both", lw=0.4)
    ax.legend(fontsize=7.5)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT_DIR / f"F9_vectorized_throughput.{ext}", bbox_inches="tight", dpi=320)
    plt.close(fig)
    print(f"\n[vec] wrote {CSV} and F9_vectorized_throughput")


if __name__ == "__main__":
    main()
