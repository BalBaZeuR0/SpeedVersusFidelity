"""JSON tanımından README için akış diyagramı / blok şeması (PNG + SVG) çizer.

Kullanım:
    python flowchart.py assets/architecture.json            # -> assets/architecture.png + .svg

Tanım biçimi (koordinatlar ızgara birimi; x sağa, y aşağı doğru artar):
{
  "title": "Sistem mimarisi",            # isteğe bağlı
  "lang": "en",                          # isteğe bağlı: lejant dili (tr varsayılan, en)
  "nodes": {
    "veri":  {"label": "Ham video\\n(21 kamera)", "pos": [0, 0], "kind": "data"},
    "emd":   {"label": "EMD", "pos": [2, 0], "kind": "model"},
    "karar": {"label": "Eşik?", "pos": [4, 0], "kind": "decision"}
  },
  "edges": [["veri", "emd"], ["emd", "karar", "yanıt haritası"],
            {"from": "karar", "to": "veri", "label": "hayır", "rad": 0.3, "style": "dashed"}],
  "groups": [{"label": "noron (GPU)", "nodes": ["emd", "karar"], "label_at": "top-right"}]   # label_at: top | top-right | bottom
}
kind: data | process | model | output | decision | external   (renk ve şekli belirler)
Kutu boyu varsayılan 2.2 x 0.9; düğüm başına "w"/"h" ile değiştirilebilir.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon

# Açık zeminli, renk körlüğüne dayanıklı palet (dolgu, kenar)
KINDS = {
    "data": ("#E7F0FA", "#2B6CB0", "Veri / girdi"),
    "process": ("#EAF5EC", "#2F7D46", "İşlem"),
    "model": ("#FFF3DC", "#B26B00", "Model / algoritma"),
    "output": ("#F1ECFA", "#6B46C1", "Çıktı / sonuç"),
    "decision": ("#FDECEC", "#C53030", "Karar"),
    "external": ("#F2F2F2", "#5A5A5A", "Dış araç / kütüphane"),
}
EN_NAMES = {"data": "Data / input", "process": "Processing", "model": "Model / algorithm",
            "output": "Output / result", "decision": "Decision", "external": "External tool / library"}
INK = "#1F2328"
SCALE = 1.15  # ızgara birimi -> inç


def box_edge(cx, cy, w, h, tx, ty, diamond=False):
    """(cx,cy) merkezli kutunun, (tx,ty) yönündeki kenar noktası."""
    dx, dy = tx - cx, ty - cy
    if dx == 0 and dy == 0:
        return cx, cy
    if diamond:
        t = 1 / (abs(dx) / (w / 2) + abs(dy) / (h / 2))
    else:
        t = min((w / 2) / abs(dx) if dx else float("inf"), (h / 2) / abs(dy) if dy else float("inf"))
    return cx + dx * t, cy + dy * t


def render(spec: dict, out_base: Path) -> list[Path]:
    nodes = spec["nodes"]
    for n in nodes.values():
        n.setdefault("w", 2.2)
        n.setdefault("h", 0.9 if n.get("kind") != "decision" else 1.1)
    def group_rect(g):
        mem = [nodes[i] for i in g["nodes"]]
        top, bottom = (0.3, 0.55) if g.get("label_at") == "bottom" else (0.55, 0.3)
        return (min(n["pos"][0] - n["w"] / 2 for n in mem) - 0.35, min(n["pos"][1] - n["h"] / 2 for n in mem) - top,
                max(n["pos"][0] + n["w"] / 2 for n in mem) + 0.35, max(n["pos"][1] + n["h"] / 2 for n in mem) + bottom)

    # Sınırlar: kutuların ve grup çerçevelerinin birleşimi (grup başlığı için üstte 0.55 pay)
    rects = [(n["pos"][0] - n["w"] / 2, n["pos"][1] - n["h"] / 2, n["pos"][0] + n["w"] / 2, n["pos"][1] + n["h"] / 2)
             for n in nodes.values()] + [group_rect(g) for g in spec.get("groups", [])]
    bx0, by0 = min(r[0] for r in rects), min(r[1] for r in rects)
    bx1, by1 = max(r[2] for r in rects), max(r[3] for r in rects)
    title = spec.get("title")
    used_kinds = [k for k in KINDS if any(n.get("kind", "process") == k for n in nodes.values())]
    m, legend_h, title_h = 0.3, 0.75, (0.6 if title else 0)
    x0, x1, y0, y1 = bx0 - m, bx1 + m, by0 - m - title_h, by1 + m
    fig, ax = plt.subplots(figsize=((x1 - x0) * SCALE, (y1 - y0 + legend_h) * SCALE), dpi=150)
    fig.patch.set_facecolor("white")
    ax.set_position([0, 0, 1, 1])
    ax.set_xlim(x0, x1)
    ax.set_ylim(y1 + legend_h, y0)  # y aşağı
    ax.set_aspect("equal")
    ax.axis("off")
    if title:
        ax.text(x0 + m, y0 + title_h / 2, title, fontsize=15, fontweight="bold", color=INK, va="center")

    for g in spec.get("groups", []):
        gx0, gy0, gx1, gy1 = group_rect(g)
        ax.add_patch(FancyBboxPatch((gx0, gy0), gx1 - gx0, gy1 - gy0, boxstyle="round,pad=0,rounding_size=0.25",
                                    fc=g.get("fill", "#FAFAFA"), ec="#9AA0A6", lw=1.1, ls=(0, (4, 3)), zorder=0))
        at = g.get("label_at", "top")  # top | top-right | bottom
        ly_ = gy1 - 0.2 if at == "bottom" else gy0 + 0.22
        lx_, ha_ = (gx1 - 0.2, "right") if at == "top-right" else (gx0 + 0.2, "left")
        ax.text(lx_, ly_, g["label"], fontsize=9.5, color="#57606A", va="center", ha=ha_,
                fontweight="bold", zorder=1)

    for n in nodes.values():
        fc, ec, _ = KINDS[n.get("kind", "process")]
        cx, cy, w, h = n["pos"][0], n["pos"][1], n["w"], n["h"]
        if n.get("kind") == "decision":
            ax.add_patch(Polygon([(cx, cy - h / 2), (cx + w / 2, cy), (cx, cy + h / 2), (cx - w / 2, cy)],
                                 closed=True, fc=fc, ec=ec, lw=1.6, zorder=2))
        else:
            ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h, boxstyle="round,pad=0,rounding_size=0.14",
                                        fc=fc, ec=ec, lw=1.6, zorder=2))
        ax.text(cx, cy, n["label"], ha="center", va="center", fontsize=n.get("fontsize", 9.5),
                color=INK, zorder=3, linespacing=1.25)

    for e in spec.get("edges", []):
        e = {"from": e[0], "to": e[1], "label": e[2] if len(e) > 2 else None} if isinstance(e, list) else e
        a, b = nodes[e["from"]], nodes[e["to"]]
        ax_, ay_ = box_edge(*a["pos"], a["w"], a["h"], *b["pos"], a.get("kind") == "decision")
        bx_, by_ = box_edge(*b["pos"], b["w"], b["h"], *a["pos"], b.get("kind") == "decision")
        rad = e.get("rad", 0)
        ax.add_patch(FancyArrowPatch((ax_, ay_), (bx_, by_), arrowstyle="-|>", mutation_scale=13, lw=1.4,
                                     color="#444C56", connectionstyle=f"arc3,rad={rad}",
                                     linestyle=e.get("style", "solid"), shrinkA=2, shrinkB=2, zorder=1.5))
        if e.get("label"):
            mx, my = (ax_ + bx_) / 2, (ay_ + by_) / 2
            # yaylı oklarda etiketi yayın tepesine kaydır
            mx += -rad * (by_ - ay_) / 2
            my += rad * (bx_ - ax_) / 2
            horizontal = abs(bx_ - ax_) >= abs(by_ - ay_)
            # varsayılan: yatay okta etiket okun üstünde, dikey okta sağında (ok ucu kapanmasın)
            off = e.get("label_offset", [0, -0.2] if horizontal else [0.12, 0])
            ha = "center" if horizontal or "label_offset" in e else "left"
            ax.text(mx + off[0], my + off[1], e["label"], ha=ha, va="center", fontsize=8.3,
                    color="#57606A", style="italic", zorder=4,
                    bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none"))

    lx, ly = x0 + m, y1 + legend_h / 2
    renderer = fig.canvas.get_renderer()
    inv = ax.transData.inverted()
    for k in used_kinds:
        fc, ec, name = KINDS[k]
        if spec.get("lang") == "en":
            name = EN_NAMES[k]
        ax.add_patch(FancyBboxPatch((lx, ly - 0.15), 0.42, 0.3, boxstyle="round,pad=0,rounding_size=0.06",
                                    fc=fc, ec=ec, lw=1.2))
        t = ax.text(lx + 0.55, ly, name, fontsize=8.5, color="#57606A", va="center")
        bb = t.get_window_extent(renderer)
        lx = inv.transform((bb.x1, bb.y0))[0] + 0.45

    out_base.parent.mkdir(parents=True, exist_ok=True)
    outs = [out_base.with_suffix(".png"), out_base.with_suffix(".svg")]
    for o in outs:
        fig.savefig(o, bbox_inches="tight", pad_inches=0.15, facecolor="white")
    plt.close(fig)
    return outs


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    src = Path(sys.argv[1])
    spec = json.loads(src.read_text(encoding="utf-8"))
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else src.with_suffix("")
    for o in render(spec, out):
        print(o)
    return 0


if __name__ == "__main__":
    sys.exit(main())
