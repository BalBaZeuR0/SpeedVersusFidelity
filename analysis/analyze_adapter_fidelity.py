"""Pair analytic vs PhysX trajectories and compute the speed/accuracy metrics.

Reads ``outputs/adapter_fidelity/<scenario>__r<NN>__<adapter>.parquet`` (+ ``.json``
sidecars) and produces:

* ``metrics.csv``       - one row per (scenario, repeat): speed + accuracy metrics
* ``summary.csv``       - aggregated (median / IQR) per scenario
* ``fig_error_time.png``   - position/velocity/attitude error vs time
* ``fig_tradeoff.png``     - speedup vs pos_rmse (the headline figure)
* ``fig_speed.png``        - step/s bar chart, analytic vs physx

PhysX is treated as the reference ("ground truth"); the analytic run is scored
against it. See D:/obsidianVault/IsaacSim_Adapter/01_Plan.md section 4.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

REF = "physx"       # reference adapter
TEST = "analytic"   # adapter scored against the reference


# ------------------------------------------------------------------------ metrics


def quat_geodesic_deg(q_a: np.ndarray, q_b: np.ndarray) -> np.ndarray:
    """Angle (deg) of the relative rotation between two wxyz quaternion tracks."""
    q_a = q_a / np.clip(np.linalg.norm(q_a, axis=1, keepdims=True), 1e-12, None)
    q_b = q_b / np.clip(np.linalg.norm(q_b, axis=1, keepdims=True), 1e-12, None)
    dot = np.abs(np.sum(q_a * q_b, axis=1))
    return np.degrees(2.0 * np.arccos(np.clip(dot, 0.0, 1.0)))


def body_up(q_wxyz: np.ndarray) -> np.ndarray:
    """World-frame body +Z axis for a track of wxyz quaternions."""
    w, x, y, z = q_wxyz[:, 0], q_wxyz[:, 1], q_wxyz[:, 2], q_wxyz[:, 3]
    return np.stack([2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)], axis=1)


def tilt_err_deg(q_a: np.ndarray, q_b: np.ndarray) -> np.ndarray:
    """Angle (deg) between the two body-up axes - attitude error ignoring yaw.

    The analytic adapter carries no rotational state (it points the body along the
    thrust vector), so full-quaternion error is dominated by the yaw it discards.
    Tilt is the part that actually matters for landing.
    """
    ua, ub = body_up(q_a), body_up(q_b)
    ua = ua / np.clip(np.linalg.norm(ua, axis=1, keepdims=True), 1e-12, None)
    ub = ub / np.clip(np.linalg.norm(ub, axis=1, keepdims=True), 1e-12, None)
    return np.degrees(np.arccos(np.clip(np.sum(ua * ub, axis=1), -1.0, 1.0)))


def usable_horizon(t: np.ndarray, err: np.ndarray, threshold: float) -> float:
    over = np.where(err > threshold)[0]
    return float(t[over[0]]) if over.size else float(t[-1])


def divergence_rate(t: np.ndarray, err: np.ndarray) -> float:
    """Least-squares slope of ||error|| vs time (m/s)."""
    if t.size < 2:
        return 0.0
    a = np.polyfit(t, err, 1)
    return float(a[0])


def pair_metrics(ref: pd.DataFrame, test: pd.DataFrame, body_length_m: float) -> dict:
    n = min(len(ref), len(test))
    ref, test = ref.iloc[:n], test.iloc[:n]
    t = ref["t"].to_numpy()

    p_ref = ref[["px", "py", "pz"]].to_numpy()
    p_test = test[["px", "py", "pz"]].to_numpy()
    v_ref = ref[["vx", "vy", "vz"]].to_numpy()
    v_test = test[["vx", "vy", "vz"]].to_numpy()
    q_ref = ref[["qw", "qx", "qy", "qz"]].to_numpy()
    q_test = test[["qw", "qx", "qy", "qz"]].to_numpy()
    have_omega = {"wx", "wy", "wz"}.issubset(ref.columns) and {"wx", "wy", "wz"}.issubset(test.columns)
    w_ref = ref[["wx", "wy", "wz"]].to_numpy() if have_omega else np.zeros_like(p_ref)
    w_test = test[["wx", "wy", "wz"]].to_numpy() if have_omega else np.zeros_like(p_ref)

    pos_err = np.linalg.norm(p_test - p_ref, axis=1)
    vel_err = np.linalg.norm(v_test - v_ref, axis=1)
    att_err = quat_geodesic_deg(q_test, q_ref)
    tilt_err = tilt_err_deg(q_test, q_ref)
    omega_err = np.degrees(np.linalg.norm(w_test - w_ref, axis=1))  # deg/s

    # Metrics are computed over the *flight* phase only: once either rocket touches
    # down, the two trajectories are no longer comparable (one clamps, the other
    # slides/bounces/penetrates), so post-touchdown samples would swamp the RMSE.
    td_ref = np.where(ref["leg_contact"].to_numpy() > 0)[0]
    td_test = np.where(test["leg_contact"].to_numpy() > 0)[0]
    td = int(min(td_ref[0] if td_ref.size else n - 1,
                 td_test[0] if td_test.size else n - 1))
    landed = bool(td_ref.size or td_test.size)
    fl = slice(0, td + 1)  # flight window, inclusive of the touchdown sample

    return {
        "n": n,
        "landed": landed,
        "touchdown_step": td,
        "touchdown_time_s": float(t[td]),
        # --- flight-phase accuracy (analytic vs PhysX reference) ---
        "pos_rmse": float(np.sqrt(np.mean(pos_err[fl] ** 2))),
        "pos_err_touchdown": float(pos_err[td]),
        "vel_rmse": float(np.sqrt(np.mean(vel_err[fl] ** 2))),
        "vel_err_touchdown": float(vel_err[td]),
        "tilt_rmse_deg": float(np.sqrt(np.mean(tilt_err[fl] ** 2))),
        "tilt_err_touchdown_deg": float(tilt_err[td]),
        "tilt_err_max_deg": float(np.max(tilt_err[fl])),
        "att_rmse_deg_full": float(np.sqrt(np.mean(att_err[fl] ** 2))),
        # analytic attitude is C0-discontinuous at gimbal steps -> impulsive implied
        # body rate; report the robust median rather than a spike-dominated RMS.
        "omega_err_median_dps": float(np.median(omega_err[fl])),
        "omega_err_touchdown_dps": float(omega_err[td]),
        "divergence_rate_mps": divergence_rate(t[fl], pos_err[fl]),
        "usable_horizon_0p5m_s": usable_horizon(t[fl], pos_err[fl], 0.5),
        "usable_horizon_bodylen_s": usable_horizon(t[fl], pos_err[fl], body_length_m),
        # --- landing-outcome agreement ---
        "touchdown_xy_err_m": float(np.linalg.norm((p_test[td] - p_ref[td])[:2])),
        "touchdown_vspeed_ref": float(abs(v_ref[td, 2])),
        "touchdown_vspeed_test": float(abs(v_test[td, 2])),
        "touchdown_vspeed_err": float(abs(v_test[td, 2]) - abs(v_ref[td, 2])),
        "_series": {"t": t[fl], "pos_err": pos_err[fl], "vel_err": vel_err[fl],
                    "att_err": att_err[fl], "tilt_err": tilt_err[fl], "omega_err": omega_err[fl]},
    }


# --------------------------------------------------------------------------- io


def _read_run(path: Path) -> pd.DataFrame:
    if path.suffix == ".npz":
        with np.load(path, allow_pickle=False) as z:
            return pd.DataFrame.from_records(z["rows"])
    return pd.read_parquet(path)


def load_runs(in_dir: Path) -> dict:
    runs: dict = {}
    paths = sorted(list(in_dir.glob("*.npz")) + list(in_dir.glob("*.parquet")))
    for pth in paths:
        js = pth.with_suffix(".json")
        if not js.exists():
            continue
        meta = json.loads(js.read_text())
        key = (meta["scenario"], int(meta["repeat"]))
        runs.setdefault(key, {})[meta["adapter"]] = {"df": _read_run(pth), "meta": meta}
    return runs


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--in-dir", default="outputs/adapter_fidelity")
    p.add_argument("--out-dir", default="outputs/adapter_fidelity/analysis")
    p.add_argument("--body-length-m", type=float, default=2.0)
    p.add_argument("--no-figures", action="store_true")
    args = p.parse_args(argv)

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    runs = load_runs(in_dir)
    rows, series = [], []
    for (scenario, repeat), by_adapter in sorted(runs.items()):
        if REF not in by_adapter or TEST not in by_adapter:
            print(f"[analyze] skip {scenario} r{repeat}: need both {REF} and {TEST}")
            continue
        ref, test = by_adapter[REF], by_adapter[TEST]
        m = pair_metrics(ref["df"], test["df"], args.body_length_m)
        s = m.pop("_series")
        s.update(scenario=scenario, repeat=repeat)
        series.append(s)

        sp_ref = ref["meta"]["speed"]["step_per_s_steady_median"]
        sp_test = test["meta"]["speed"]["step_per_s_steady_median"]
        rows.append({
            "scenario": scenario, "repeat": repeat,
            "step_per_s_physx": sp_ref, "step_per_s_analytic": sp_test,
            "speedup": sp_test / sp_ref if sp_ref else np.nan,
            "rt_factor_physx": ref["meta"]["speed"]["rt_factor_steady_median"],
            "rt_factor_analytic": test["meta"]["speed"]["rt_factor_steady_median"],
            **m,
        })

    if not rows:
        print("[analyze] no analytic/physx pairs found - nothing to do")
        return 1

    metrics = pd.DataFrame(rows)
    metrics.to_csv(out_dir / "metrics.csv", index=False)

    agg_cols = ["speedup", "pos_rmse", "pos_err_touchdown", "vel_rmse",
                "tilt_rmse_deg", "tilt_err_max_deg", "omega_err_median_dps",
                "divergence_rate_mps", "usable_horizon_0p5m_s",
                "touchdown_time_s", "touchdown_xy_err_m", "touchdown_vspeed_err",
                "step_per_s_physx", "step_per_s_analytic"]
    summary = metrics.groupby("scenario")[agg_cols].agg(["median", _iqr])
    summary.to_csv(out_dir / "summary.csv")
    print(metrics.to_string(index=False))
    print("\n[analyze] wrote", out_dir / "metrics.csv", "and summary.csv")

    if not args.no_figures:
        _figures(metrics, series, out_dir)
    return 0


def _iqr(x):
    return float(np.percentile(x, 75) - np.percentile(x, 25))


def _figures(metrics: pd.DataFrame, series: list, out_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    scenarios = sorted({s["scenario"] for s in series})

    # --- error vs time (one line per scenario, repeat 0) ---
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for sc in scenarios:
        s0 = next((s for s in series if s["scenario"] == sc and s["repeat"] == 0), None)
        if s0 is None:
            continue
        axes[0].plot(s0["t"], s0["pos_err"], label=sc)
        axes[1].plot(s0["t"], s0["vel_err"], label=sc)
        axes[2].plot(s0["t"], s0["tilt_err"], label=sc)
    for ax, ttl, yl in zip(
        axes,
        ["Position error", "Velocity error", "Tilt error (yaw-free)"],
        ["|dp| (m)", "|dv| (m/s)", "angle between body-up axes (deg)"],
    ):
        ax.set_title(ttl); ax.set_xlabel("t (s)"); ax.set_ylabel(yl); ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out_dir / "fig_error_time.png", dpi=140); plt.close(fig)

    # --- tradeoff: speedup vs pos_rmse ---
    fig, ax = plt.subplots(figsize=(6, 5))
    g = metrics.groupby("scenario")
    ax.errorbar(g["speedup"].median(), g["pos_rmse"].median(),
                xerr=g["speedup"].agg(_iqr), yerr=g["pos_rmse"].agg(_iqr),
                fmt="o", capsize=3)
    for sc, x, y in zip(g.groups.keys(), g["speedup"].median(), g["pos_rmse"].median()):
        ax.annotate(sc, (x, y), fontsize=8, xytext=(5, 5), textcoords="offset points")
    ax.set_xlabel("speedup  (analytic step/s / PhysX step/s)")
    ax.set_ylabel("flight-phase position RMSE vs PhysX (m)")
    ax.set_title("Speed / accuracy tradeoff")
    ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out_dir / "fig_tradeoff.png", dpi=140); plt.close(fig)

    # --- speed bar chart ---
    fig, ax = plt.subplots(figsize=(7, 4))
    piv = metrics.groupby("scenario")[["step_per_s_physx", "step_per_s_analytic"]].median()
    piv.plot(kind="bar", ax=ax, logy=True)
    ax.set_ylabel("steps / s (log)"); ax.set_title("Throughput"); ax.grid(alpha=0.3, axis="y")
    fig.tight_layout(); fig.savefig(out_dir / "fig_speed.png", dpi=140); plt.close(fig)
    print("[analyze] wrote figures ->", out_dir)


if __name__ == "__main__":
    raise SystemExit(main())
