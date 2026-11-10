"""Regenerate the manuscript figures (experiment flow + speed/fidelity trade-off).

Rules applied (journals.tubitak.gov.tr/elektrik/styleguide.html):
  - Times New Roman in every figure,
  - graph/diagram line weights between 0.5 and 1 pt,
  - drawn at the printed width (8-16 cm), so font and line sizes are true sizes,
  - written as PDF (vector) and PNG (README) into plots/.

Figure 1 (flow diagram) follows fig_flow() of analysis/paper_figures.py.
Figure 2 (speed-accuracy tradeoff) follows fig_tradeoff(); the per-scenario values are the
medians and per-repeat min-max of outputs/adapter_fidelity/analysis/metrics.csv as printed
by the analysis run of 2026-09-02 (the CSV itself is no longer on disk). They match
Tables 3 and 4 of the manuscript.

    python scripts/make_paper_figures.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

CM = 1 / 2.54
OUT = Path(__file__).resolve().parents[1] / "plots"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "mathtext.fontset": "stix",
    "font.size": 8,
    "axes.labelsize": 8,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.minor.width": 0.5,
    "ytick.minor.width": 0.5,
    "lines.linewidth": 0.8,
    "grid.linewidth": 0.5,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "pdf.fonttype": 42,
    "savefig.dpi": 1200,
})


def _save(fig, name: str) -> None:
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(OUT / f"{name}.png", bbox_inches="tight", pad_inches=0.02, dpi=300)
    plt.close(fig)
    print(f"wrote plots/{name}.pdf / .png")


def fig_flow() -> None:
    fig = plt.figure(figsize=(12 * CM, 5.4 * CM))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 46)
    ax.axis("off")

    def box(cx, cy, w, h, lines, fc="#F3F3F3"):
        ax.add_patch(plt.Rectangle((cx - w / 2, cy - h / 2), w, h, fc=fc, ec="#333",
                                   lw=0.8, joinstyle="round"))
        ax.text(cx, cy, "\n".join(lines), ha="center", va="center", fontsize=7,
                linespacing=1.25)

    def arrow(x1, y1, x2, y2):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>", color="#333", lw=0.8,
                                    shrinkA=0, shrinkB=0, mutation_scale=7))

    box(10, 23, 18, 14, ["Reset", "randomiser", "(one shared", "initial condition)"])
    box(35, 34, 21, 12, ["Analytic adapter", "point mass,", "kinematic attitude"], fc="#FBE7CF")
    box(35, 12, 21, 12, ["PhysX adapter", "6-DOF rigid body", "+ gimbal moment"], fc="#D8E5F3")
    box(62, 23, 20, 30, ["Shared", "guidance", "+ adapter-", "matched inner", "attitude loop"])
    box(87, 23, 16, 14, ["Pair and", "score", "(PhysX =", "reference)"])

    arrow(19, 27, 24.5, 32)
    arrow(19, 19, 24.5, 14)
    arrow(45.5, 36, 52, 36)
    ax.text(48.7, 37.2, "state", fontsize=6.5, ha="center", color="#444")
    arrow(52, 31, 45.5, 31)
    ax.text(48.7, 28.2, "action", fontsize=6.5, ha="center", color="#444")
    arrow(45.5, 15, 52, 15)
    ax.text(48.7, 16.2, "state", fontsize=6.5, ha="center", color="#444")
    arrow(52, 10, 45.5, 10)
    ax.text(48.7, 7.2, "action", fontsize=6.5, ha="center", color="#444")
    arrow(72, 23, 79, 23)
    ax.text(50, 43.3, "S1, S2 baseline: a fixed action sequence is replayed in place of the guidance loop",
            fontsize=6.5, style="italic", color="#555", ha="center")
    _save(fig, "F1_flow")


# scenario: (median speedup, median position RMSE [m], min, max) over five repeats
TRADEOFF = {
    "S2 free-fall+burn": (5.83, 0.087, 0.087, 0.087, "s"),
    "S1 hover": (5.93, 8.68, 7.29, 9.40, "s"),
    "c1 nominal": (5.50, 0.34, 0.16, 0.63, "o"),
    "c2 divert": (5.93, 1.31, 1.04, 2.03, "o"),
    "c3 fast descent": (5.69, 0.39, 0.14, 0.77, "o"),
    "c4 tight guidance": (5.65, 0.46, 0.14, 1.14, "o"),
    "c5 recovery": (5.54, 0.25, 0.21, 0.68, "o"),
}
LABEL_POS = {  # label anchor (x, y), ha, leader line?  -- keeps every label clear of the bars
    "S2 free-fall+burn": ((5.855, 0.087), "left", False),
    "S1 hover": ((5.955, 8.68), "left", False),
    "c1 nominal": ((5.475, 0.40), "right", False),
    "c2 divert": ((5.955, 1.31), "left", False),
    "c3 fast descent": ((5.715, 0.36), "left", False),
    "c4 tight guidance": ((5.65, 1.55), "center", False),
    "c5 recovery": ((5.475, 0.20), "right", True),
}


def fig_tradeoff() -> None:
    colors = plt.cm.viridis(np.linspace(0.05, 0.9, len(TRADEOFF)))
    fig, ax = plt.subplots(figsize=(9 * CM, 6.2 * CM))
    for (name, (x, y, lo, hi, mk)), c in zip(TRADEOFF.items(), colors):
        ax.plot([x, x], [lo, hi], color=c, lw=0.8, zorder=1)
        ax.scatter([x], [y], marker=mk, s=22, color=c, edgecolors="#333", linewidths=0.5, zorder=2)
        (lx, ly), ha, leader = LABEL_POS[name]
        arrowprops = dict(arrowstyle="-", color="#666", lw=0.5, shrinkA=1, shrinkB=3) if leader else None
        ax.annotate(name, xy=(x, y), xytext=(lx, ly), fontsize=6.5, ha=ha, va="center",
                    arrowprops=arrowprops)
    ax.set_yscale("log")
    ax.set_ylim(0.05, 14)
    ax.set_xlim(5.2, 6.3)
    ax.axhline(1.0, color="#999", lw=0.6, ls="--")
    ax.set_xlabel("Speedup (analytic step/s ÷ PhysX step/s)")
    ax.set_ylabel("Flight-phase position RMSE (m)")
    ax.text(5.22, 0.062, "closed-loop descents: sub-metre", fontsize=6.5, color="#444")
    fig.tight_layout(pad=0.3)
    _save(fig, "F4_tradeoff")


if __name__ == "__main__":
    fig_flow()
    fig_tradeoff()
