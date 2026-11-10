"""Publication figures for the analytic-vs-PhysX adapter-fidelity paper.

Consistent house style (>=300 dpi, shared palette, sentence-case where captions live
in the paper). Reads outputs/adapter_fidelity/{*.npz, analysis/metrics.csv} and the
yaw-sweep metrics CSV. Writes PNG + PDF into outputs/adapter_fidelity/paper_figs/.

    python analysis/paper_figures.py [--yawmix path/to/metrics_yawmix.csv]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# --------------------------------------------------------------------- house style

plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 320,
    "font.size": 9,
    "axes.titlesize": 9,
    "axes.labelsize": 9,
    "legend.fontsize": 8,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "lines.linewidth": 1.4,
})

C_ANALYTIC = "#E08214"   # amber
C_PHYSX = "#2C6FBB"      # slate blue
C_TARGET = "#4D4D4D"
SCEN_ORDER = ["s2_freefall_then_burn", "s1_hover",
              "c1_nominal", "c2_divert", "c3_fast_descent",
              "c4_tight_guidance", "c5_high_tilt"]
SCEN_LABEL = {
    "s1_hover": "S1 hover", "s2_freefall_then_burn": "S2 free-fall+burn",
    "c1_nominal": "c1 nominal", "c2_divert": "c2 divert",
    "c3_fast_descent": "c3 fast descent", "c4_tight_guidance": "c4 tight guidance",
    "c5_high_tilt": "c5 recovery",
}
SCEN_CMAP = dict(zip(SCEN_ORDER, plt.cm.viridis(np.linspace(0.05, 0.9, len(SCEN_ORDER)))))
# initial |yaw| (deg) drawn per seed by the native reset randomiser (unclamped run)
SEED_ABS_YAW = {0: 21.0, 1: 31.0, 2: 150.0, 3: 9.0, 4: 121.0}

IN_DIR = Path("outputs/adapter_fidelity")
OUT_DIR = IN_DIR / "paper_figs"


def _save(fig, name: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(OUT_DIR / f"{name}.{ext}", bbox_inches="tight")
    plt.close(fig)
    print(f"[paper_figs] {name}")


def _load(scenario: str, repeat: int, adapter: str) -> pd.DataFrame | None:
    p = IN_DIR / f"{scenario}__r{repeat:02d}__{adapter}.npz"
    if not p.exists():
        return None
    with np.load(p) as z:
        return pd.DataFrame.from_records(z["rows"])


def _tilt_deg(df: pd.DataFrame) -> np.ndarray:
    w, x, y, z = df.qw.to_numpy(), df.qx.to_numpy(), df.qy.to_numpy(), df.qz.to_numpy()
    up_z = 1 - 2 * (x * x + y * y)
    return np.degrees(np.arccos(np.clip(up_z, -1, 1)))


def _flight_slice(a: pd.DataFrame, p: pd.DataFrame) -> slice:
    n = min(len(a), len(p))
    lc = np.where((a.leg_contact.to_numpy()[:n] > 0) | (p.leg_contact.to_numpy()[:n] > 0))[0]
    return slice(0, int(lc[0]) + 1 if lc.size else n)


# ------------------------------------------------------------------ F1  flow chart


def fig_flow() -> None:
    fig, ax = plt.subplots(figsize=(7.6, 3.6))
    ax.set_xlim(0, 100); ax.set_ylim(0, 46); ax.axis("off"); ax.grid(False)

    def box(cx, cy, w, h, lines, fc="#F3F3F3"):
        ax.add_patch(plt.Rectangle((cx - w / 2, cy - h / 2), w, h, fc=fc, ec="#333",
                                   lw=1.0, joinstyle="round"))
        ax.text(cx, cy, "\n".join(lines), ha="center", va="center", fontsize=7.3,
                linespacing=1.4)

    def arrow(x1, y1, x2, y2, rad=0.0):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>", color="#333", lw=1.0,
                                    shrinkA=0, shrinkB=0,
                                    connectionstyle=f"arc3,rad={rad}"))

    box(10, 24, 17, 13, ["Reset", "randomiser", "(one shared", "initial condition)"])
    box(35, 34, 20, 12, ["Analytic adapter", "point mass,", "kinematic attitude"], fc="#FBE7CF")
    box(35, 12, 20, 12, ["PhysX adapter", "6-DOF rigid body", "+ gimbal moment"], fc="#D8E5F3")
    box(62, 23, 20, 30, ["Shared", "guidance", "", "+ adapter-", "matched inner", "attitude loop"])
    box(86, 23, 15, 13, ["Pair &", "score", "(PhysX", "= reference)"])

    # reset -> adapters
    arrow(18.5, 27, 25, 32)
    arrow(18.5, 21, 25, 14)
    # analytic <-> guidance  (state out at top, action in below)
    arrow(45, 36, 52, 36); ax.text(48.5, 37.4, "state", fontsize=6.4, ha="center", color="#555")
    arrow(52, 31, 45, 31); ax.text(48.5, 29.4, "action", fontsize=6.4, ha="center", color="#555")
    # physx <-> guidance
    arrow(45, 15, 52, 15); ax.text(48.5, 16.4, "state", fontsize=6.4, ha="center", color="#555")
    arrow(52, 10, 45, 10); ax.text(48.5, 8.4, "action", fontsize=6.4, ha="center", color="#555")
    # guidance -> pair
    arrow(72, 23, 78.5, 23)
    ax.text(50, 44.5, "S1, S2 baseline: a fixed action sequence is replayed in place of the guidance loop",
            fontsize=6.6, style="italic", color="#666", ha="center")
    _save(fig, "F1_flow")


# ---------------------------------------------------------- F2  3D trajectories


def fig_trajectories() -> None:
    from app.adapter_fidelity_controller import CLOSED_LOOP_SCENARIOS
    scens = ["c1_nominal", "c2_divert", "c5_high_tilt"]
    fig = plt.figure(figsize=(7.4, 2.8))
    for k, sc in enumerate(scens):
        a, p = _load(sc, 0, "analytic"), _load(sc, 0, "physx")
        if a is None or p is None:
            continue
        fl = _flight_slice(a, p)
        ax = fig.add_subplot(1, 3, k + 1, projection="3d")
        ax.plot(a.px[fl], a.py[fl], a.pz[fl], color=C_ANALYTIC, label="analytic")
        ax.plot(p.px[fl], p.py[fl], p.pz[fl], color=C_PHYSX, label="PhysX")
        t = json.loads((IN_DIR / f"{sc}__r00__physx.json").read_text())["reset_sample"]["target_position"]
        off = CLOSED_LOOP_SCENARIOS[sc].target_offset_xy
        tgt = (t[0] + off[0], t[1] + off[1], 0.0)
        # terrain height at the target (from the analytic trajectory's ground track end)
        gz = float(min(a.pz[fl].min(), p.pz[fl].min(), 0.0))
        ax.scatter([tgt[0]], [tgt[1]], [gz], color=C_TARGET, marker="x", s=30)
        ax.set_title(SCEN_LABEL[sc], pad=0)
        ax.set_xlabel("x (m)", labelpad=-6); ax.set_ylabel("y (m)", labelpad=-6)
        ax.set_zlabel("z (m)", labelpad=-6)
        ax.tick_params(labelsize=6, pad=-2)
        ax.view_init(elev=18, azim=-60)
        if k == 0:
            ax.legend(loc="upper left", fontsize=7, bbox_to_anchor=(-0.1, 1.05))
    fig.tight_layout(w_pad=0.5)
    _save(fig, "F2_trajectories")


# ---------------------------------------------------- F3  error vs time (3 panels)


def fig_error_time() -> None:
    fig, axes = plt.subplots(1, 3, figsize=(7.6, 2.6), sharex=True)
    for sc in SCEN_ORDER:
        a, p = _load(sc, 0, "analytic"), _load(sc, 0, "physx")
        if a is None or p is None:
            continue
        fl = _flight_slice(a, p)
        n = fl.stop
        t = a.t.to_numpy()[:n]
        pe = np.linalg.norm(a[["px", "py", "pz"]].to_numpy()[:n] - p[["px", "py", "pz"]].to_numpy()[:n], axis=1)
        ve = np.linalg.norm(a[["vx", "vy", "vz"]].to_numpy()[:n] - p[["vx", "vy", "vz"]].to_numpy()[:n], axis=1)
        pt = _tilt_deg(p.iloc[:n])                         # PhysX body tilt (analytic ~ 0)
        k = 7
        pt_s = np.convolve(pt, np.ones(k) / k, mode="same")
        col = SCEN_CMAP[sc]
        axes[0].plot(t, pe, color=col, label=SCEN_LABEL[sc])
        axes[1].plot(t, ve, color=col)
        axes[2].plot(t[k:-k], pt_s[k:-k], color=col)
    for ax, yl in zip(axes, ["position error (m)", "velocity error (m/s)",
                             "PhysX body tilt (deg)"]):
        ax.set_xlabel("time (s)"); ax.set_ylabel(yl)
    axes[2].set_ylim(0, 16)
    axes[2].text(0.5, 15, "analytic stays < 1° from vertical", fontsize=6.2, color="#666")
    axes[0].legend(fontsize=6.3, ncol=1, loc="upper left")
    fig.tight_layout(w_pad=0.6)
    _save(fig, "F3_error_time")


# --------------------------------------------------------- F4  speed/accuracy


def fig_tradeoff(m: pd.DataFrame) -> None:
    g = m.groupby("scenario")
    sp_all = m.speedup.median()
    fig, ax = plt.subplots(figsize=(5.0, 3.4))
    for sc in SCEN_ORDER:
        d = g.get_group(sc)
        x = d.speedup.median()
        y = d.pos_rmse.median()
        ylo, yhi = d.pos_rmse.min(), d.pos_rmse.max()
        mk = "s" if sc.startswith("s") else "o"
        ax.plot([x, x], [ylo, yhi], color=SCEN_CMAP[sc], lw=1.0, alpha=0.6, zorder=1)
        ax.scatter([x], [y], marker=mk, s=42, color=SCEN_CMAP[sc], ec="#333", lw=0.5, zorder=2)
        dy = 1.18 if sc != "c4_tight_guidance" else 0.72
        ax.annotate(SCEN_LABEL[sc], (x, y * dy), fontsize=7, ha="center")
    ax.set_yscale("log")
    ax.set_ylim(0.05, 14)
    ax.set_xlim(4.9, 6.4)
    ax.axhline(1.0, color="#BBB", lw=0.6, ls="--")
    ax.set_xlabel("speedup  (analytic step/s ÷ PhysX step/s)")
    ax.set_ylabel("flight-phase position RMSE vs PhysX (m, log)")
    ax.text(4.95, 10, "open-loop\nhover (S1)", fontsize=6.6, color="#555")
    ax.text(4.95, 0.11, "closed-loop descents: sub-metre", fontsize=6.6, color="#555")
    fig.tight_layout()
    _save(fig, "F4_tradeoff")


# ------------------------------------------------------------------ F5  speed


def fig_speed(m: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    piv = m.groupby("scenario")[["step_per_s_physx", "step_per_s_analytic"]].median().reindex(SCEN_ORDER)
    x = np.arange(len(piv))
    ax.bar(x - 0.2, piv.step_per_s_physx, 0.38, color=C_PHYSX, label="PhysX")
    ax.bar(x + 0.2, piv.step_per_s_analytic, 0.38, color=C_ANALYTIC, label="analytic")
    ax.set_yscale("log")
    ax.set_xticks(x); ax.set_xticklabels([SCEN_LABEL[s] for s in piv.index], rotation=30, ha="right")
    ax.set_ylabel("throughput (steps/s, log)")
    for xi, (a, b) in enumerate(zip(piv.step_per_s_physx, piv.step_per_s_analytic)):
        ax.text(xi, b * 1.15, f"{b / a:.1f}×", ha="center", fontsize=7)
    ax.legend()
    fig.tight_layout()
    _save(fig, "F5_speed")


# ------------------------------------------------------ F6  touchdown dispersion


def fig_touchdown(m: pd.DataFrame) -> None:
    from app.adapter_fidelity_controller import CLOSED_LOOP_SCENARIOS
    fig, ax = plt.subplots(figsize=(4.4, 4.0))
    first = True
    for sc in ["c1_nominal", "c2_divert", "c3_fast_descent", "c4_tight_guidance", "c5_high_tilt"]:
        off = CLOSED_LOOP_SCENARIOS[sc].target_offset_xy
        for r in range(5):
            a, p = _load(sc, r, "analytic"), _load(sc, r, "physx")
            if a is None or p is None:
                continue
            fl = _flight_slice(a, p); td = fl.stop - 1
            t = json.loads((IN_DIR / f"{sc}__r{r:02d}__physx.json").read_text())["reset_sample"]["target_position"]
            tx, ty = t[0] + off[0], t[1] + off[1]
            axy = (a.px[td] - tx, a.py[td] - ty)
            pxy = (p.px[td] - tx, p.py[td] - ty)
            ax.plot([axy[0], pxy[0]], [axy[1], pxy[1]], color="#C4C4C4", lw=0.7, zorder=1)
            ax.scatter(*axy, color=C_ANALYTIC, s=18, zorder=2, label="analytic" if first else None)
            ax.scatter(*pxy, color=C_PHYSX, s=18, marker="^", zorder=2, label="PhysX" if first else None)
            first = False
    ax.scatter([0], [0], color=C_TARGET, marker="x", s=55, zorder=3, label="target")
    ax.set_xlabel("touchdown x − target (m)"); ax.set_ylabel("touchdown y − target (m)")
    ax.set_aspect("equal"); ax.legend(loc="upper left")
    ax.set_xlim(-2, 2); ax.set_ylim(-2, 2)
    fig.tight_layout()
    _save(fig, "F6_touchdown")


# ------------------------------------------------------- F7  yaw sensitivity


def fig_yaw(yawmix_csv: Path) -> None:
    if not yawmix_csv.exists():
        print(f"[paper_figs] skip F7 (no {yawmix_csv})")
        return
    y = pd.read_csv(yawmix_csv)
    y = y[y.scenario.str.startswith("c")].copy()
    y["abs_yaw"] = y.repeat.map(SEED_ABS_YAW)
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 3.0))
    for ax, col, yl in zip(axes, ["tilt_rmse_deg", "pos_rmse"],
                           ["attitude (tilt) RMSE (deg)", "position RMSE (m)"]):
        for sc in ["c1_nominal", "c2_divert", "c3_fast_descent", "c4_tight_guidance", "c5_high_tilt"]:
            d = y[y.scenario == sc].sort_values("abs_yaw")
            ax.plot(d.abs_yaw, d[col], "o", ms=5, color=SCEN_CMAP.get(sc, "#888"),
                    label=SCEN_LABEL.get(sc, sc))
            # connect within the two sampled clusters, not across the 31-121 gap
            lo = d[d.abs_yaw <= 40]; hi = d[d.abs_yaw >= 100]
            ax.plot(lo.abs_yaw, lo[col], "-", lw=0.8, color=SCEN_CMAP.get(sc, "#888"), alpha=0.5)
            ax.plot(hi.abs_yaw, hi[col], "-", lw=0.8, color=SCEN_CMAP.get(sc, "#888"), alpha=0.5)
        ax.axvspan(0, 30, color="#DFF0D8", alpha=0.6, zorder=0)
        ax.set_xlabel("initial |heading| (deg)"); ax.set_ylabel(yl)
        ax.set_xlim(-5, 165)
    axes[0].legend(fontsize=6.3, loc="upper left")
    axes[0].text(15, axes[0].get_ylim()[1] * 0.05, "±30° main-matrix band",
                 fontsize=6.2, ha="center", rotation=90)
    fig.tight_layout(w_pad=0.6)
    _save(fig, "F7_yaw_sensitivity")


# ---------------------------------------------------------- F8  g.dt offset


def fig_gdt_offset() -> None:
    a, p = _load("s2_freefall_then_burn", 0, "analytic"), _load("s2_freefall_then_burn", 0, "physx")
    a1, p1 = _load("s1_hover", 0, "analytic"), _load("s1_hover", 0, "physx")
    if a is None or p is None:
        return
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.7))
    n = min(len(a), len(p))
    axes[0].plot(a.t[:n], a.vz[:n], color=C_ANALYTIC, label="analytic $v_z$")
    axes[0].plot(p.t[:n], p.vz[:n], color=C_PHYSX, label="PhysX $v_z$")
    axes[0].axhline(0, color="#999", lw=0.6)
    axes[0].set_title("S2: vertical velocity")
    axes[0].set_xlabel("time (s)"); axes[0].set_ylabel("$v_z$ (m/s)")
    axes[0].legend()
    if a1 is not None and p1 is not None:
        n1 = min(len(a1), len(p1))
        drift = np.hypot(a1.px[:n1] - p1.px[:n1], a1.py[:n1] - p1.py[:n1])
        axes[1].plot(a1.t[:n1], drift, color="#8856A7")
        axes[1].set_title("S1: analytic–PhysX lateral drift")
        axes[1].set_xlabel("time (s)"); axes[1].set_ylabel("horizontal separation (m)")
    fig.tight_layout(w_pad=0.6)
    _save(fig, "F8_gdt_offset")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--yawmix", default=str(IN_DIR / "analysis" / "metrics_yawmix.csv"))
    args = ap.parse_args()
    m = pd.read_csv(IN_DIR / "analysis" / "metrics.csv")
    fig_flow()
    fig_trajectories()
    fig_error_time()
    fig_tradeoff(m)
    fig_speed(m)
    fig_touchdown(m)
    fig_yaw(Path(args.yawmix))
    fig_gdt_offset()
    print(f"[paper_figs] all figures -> {OUT_DIR}")


if __name__ == "__main__":
    main()
