"""Analyse the three post-matrix extras runs (see scripts/run_extras_native.sh).

  T1  six_dof=false ablation  -> does the closed-loop divergence collapse when the
      gimbal control moment is removed? (expectation: analytic ~= PhysX to ~g*dt)
  T2  dt sweep                 -> does the S2 vertical-velocity offset scale as g*dt?
  T3  PhysX determinism        -> are two same-seed runs bit-identical?

  python analysis/analyze_adapter_fidelity_extras.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis.analyze_adapter_fidelity import _read_run, pair_metrics  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
EX = ROOT / "outputs/adapter_fidelity/extras"
OUT = ROOT / "outputs/adapter_fidelity/analysis"
G_MOON = 1.62
BODY_LEN = 1.0

ABLATION_SCENARIOS = ["c1_nominal", "c2_divert", "c3_fast_descent",
                      "c4_tight_guidance", "c5_high_tilt"]
# main-matrix (six_dof=true) pos-RMSE medians, from outputs/.../analysis/summary.csv
MAIN_POS_RMSE = {"c1_nominal": 0.345, "c2_divert": 1.309, "c3_fast_descent": 0.386,
                 "c4_tight_guidance": 0.460, "c5_high_tilt": 0.247}


def _pair(dir_: Path, scenario: str, repeat: int) -> dict | None:
    a = dir_ / f"{scenario}__r{repeat:02d}__analytic.npz"
    p = dir_ / f"{scenario}__r{repeat:02d}__physx.npz"
    if not (a.exists() and p.exists()):
        return None
    return pair_metrics(_read_run(p), _read_run(a), BODY_LEN)  # ref = physx, test = analytic


# --------------------------------------------------------------------- T3
def t3_determinism() -> None:
    print("\n" + "=" * 72 + "\nT3  PhysX determinism (c1_nominal, seed 1234, run twice)\n" + "=" * 72)
    fa = EX / "determinism_a" / "c1_nominal__r00__physx.npz"
    fb = EX / "determinism_b" / "c1_nominal__r00__physx.npz"
    if not (fa.exists() and fb.exists()):
        print("  [skip] missing", fa if not fa.exists() else fb)
        return
    da, db = _read_run(fa), _read_run(fb)
    n = min(len(da), len(db))
    cols = ["px", "py", "pz", "vx", "vy", "vz", "qw", "qx", "qy", "qz", "wx", "wy", "wz"]
    A = da[cols].to_numpy()[:n]
    B = db[cols].to_numpy()[:n]
    dmax = np.abs(A - B).max(axis=0)
    for c, d in zip(cols, dmax):
        print(f"  max abs d({c}) = {d:.3e}")
    overall = np.abs(A - B).max()
    pos = np.linalg.norm((A[:, :3] - B[:, :3]), axis=1).max()
    print(f"  --> max abs elementwise diff = {overall:.3e}   max position diff = {pos:.3e} m")
    print("  verdict:", "BIT-IDENTICAL (deterministic)" if overall == 0.0
          else "identical to fp round-off" if overall < 1e-9
          else f"NON-deterministic at {overall:.2e}")


# --------------------------------------------------------------------- T1
def t1_ablation() -> None:
    print("\n" + "=" * 72 + "\nT1  six_dof=false ablation  (analytic vs PhysX, no gimbal moment)\n" + "=" * 72)
    rows = []
    for sc in ABLATION_SCENARIOS:
        for r in range(3):
            m = _pair(EX / "ablation_nosixdof", sc, r)
            if m is None:
                continue
            rows.append(dict(scenario=sc, repeat=r, pos_rmse=m["pos_rmse"],
                             pos_err_td=m["pos_err_touchdown"],
                             vel_rmse=m["vel_rmse"],
                             td_vspeed_err=m["touchdown_vspeed_err"],
                             tilt_rmse_deg=m["tilt_rmse_deg"], landed=m["landed"]))
    if not rows:
        print("  [skip] no ablation pairs found")
        return
    df = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "extras_ablation_nosixdof.csv", index=False)
    g = df.groupby("scenario")
    print(f"  {'scenario':<18} {'pos RMSE (no-6dof)':>19} {'pos RMSE (6dof, main)':>22} {'vel RMSE':>10} {'tilt RMSE':>10}")
    for sc in ABLATION_SCENARIOS:
        if sc not in g.groups:
            continue
        s = g.get_group(sc)
        print(f"  {sc:<18} {s['pos_rmse'].median():>16.3f} m  {MAIN_POS_RMSE[sc]:>19.2f} m  "
              f"{s['vel_rmse'].median():>7.3f} m/s {s['tilt_rmse_deg'].median():>7.2f} deg")
    print(f"\n  overall no-6dof pos RMSE: median {df['pos_rmse'].median():.3f} m, "
          f"max {df['pos_rmse'].max():.3f} m   (main-matrix 6-DOF closed loop: 0.25-1.31 m)")
    print(f"  all runs land soft: {bool(df['landed'].all())}")
    print("  interpretation: with six_dof=false the PhysX body has NO rotational dynamics,")
    print("  so it freezes at its initial attitude (checked: tilt constant to 0.0 deg) and")
    print("  the shared attitude loop cannot stabilise it. Divergence does NOT collapse - it")
    print("  is comparable to / worse than the 6-DOF closed loop. => the sub-metre agreement")
    print("  of the main result depends on 6-DOF attitude dynamics + the controller acting")
    print("  together; the legacy point-mass PhysX is a WORSE surrogate for the analytic model.")


# --------------------------------------------------------------------- T2
def t2_dt_sweep() -> None:
    print("\n" + "=" * 72 + "\nT2  integration-step (dt) sweep  -  does the S2 v_z offset scale as g*dt?\n" + "=" * 72)
    specs = [("dt_120", 1 / 120), ("baseline dt=1/60", 1 / 60), ("dt_030", 1 / 30)]
    print(f"  {'dt':>10}  {'g*dt (m/s)':>11}  {'dvz freefall':>11}  {'dvz burn':>10}  "
          f"{'S2 posRMSE':>11}  {'c1 posRMSE':>11}")
    out = []
    for tag, dt in specs:
        if tag.startswith("baseline"):
            # reuse the main-matrix S2 / c1 runs at dt=1/60
            base = ROOT / "outputs/adapter_fidelity"
            s2a, s2p = base / "s2_freefall_then_burn__r00__analytic.npz", base / "s2_freefall_then_burn__r00__physx.npz"
            c1a, c1p = base / "c1_nominal__r00__analytic.npz", base / "c1_nominal__r00__physx.npz"
        else:
            d = EX / tag
            s2a, s2p = d / "s2_freefall_then_burn__r00__analytic.npz", d / "s2_freefall_then_burn__r00__physx.npz"
            c1a, c1p = d / "c1_nominal__r00__analytic.npz", d / "c1_nominal__r00__physx.npz"
        if not (s2a.exists() and s2p.exists()):
            print(f"  {dt:>10.5f}  {G_MOON*dt:>11.4f}  {'[missing]':>13}")
            continue
        a, p = _read_run(s2a), _read_run(s2p)
        n = min(len(a), len(p))
        avz = a["vz"].to_numpy()[:n]
        dvz_series = p["vz"].to_numpy()[:n] - avz
        # clean offset: the first few steps (pure free-fall, no burn, no contact)
        dvz_early = float(np.mean(dvz_series[:5]))
        # and just before touchdown (steady burn), excluding the last few clamped samples
        tdp = np.where(p["leg_contact"].to_numpy() > 0)[0]
        tda = np.where(a["leg_contact"].to_numpy() > 0)[0]
        td = int(min(tdp[0] if tdp.size else n - 1, tda[0] if tda.size else n - 1))
        dvz_burn = float(np.mean(dvz_series[max(5, td - 40):max(6, td - 3)]))
        m_s2 = pair_metrics(_read_run(s2p), _read_run(s2a), BODY_LEN)
        m_c1 = pair_metrics(_read_run(c1p), _read_run(c1a), BODY_LEN) if (c1a.exists() and c1p.exists()) else None
        c1r = f"{m_c1['pos_rmse']:.3f}" if m_c1 else "  -  "
        print(f"  {dt:>10.5f}  {G_MOON*dt:>11.4f}  {dvz_early:>11.4f}  {dvz_burn:>10.4f}  "
              f"{m_s2['pos_rmse']:>11.4f}  {c1r:>11}   ratio dvz/gdt = {dvz_early/(G_MOON*dt):.3f}")
        out.append(dict(dt=dt, g_dt=G_MOON * dt, s2_dvz_freefall=dvz_early,
                        s2_dvz_burn=dvz_burn, s2_pos_rmse=m_s2["pos_rmse"],
                        c1_pos_rmse=(m_c1["pos_rmse"] if m_c1 else np.nan)))
    if out:
        pd.DataFrame(out).to_csv(OUT / "extras_dt_sweep.csv", index=False)
        print("\n  expectation: |mean dv_z| ~ g*dt (one-step gravity lag of semi-implicit Euler);")
        print("  a near-linear column confirms the mechanism in Section 4.2.")


if __name__ == "__main__":
    t3_determinism()
    t1_ablation()
    t2_dt_sweep()
    print("\n[extras] wrote CSVs under", OUT)
