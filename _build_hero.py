"""Generate the README hero image from outputs/results_mlp.json.

Pareto frontier: test accuracy vs unstructured sparsity, with lambda-labelled
points and an annotation for each operating point. Saved to
figures/hero_pareto.png.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).parent
with (ROOT / "outputs" / "results_mlp.json").open("r", encoding="utf-8") as f:
    results = json.load(f)

runs = results["runs"]
lams = sorted([r["lambda"] for r in runs.values()])
pts = []
for key, r in runs.items():
    pts.append({
        "lam":         r["lambda"],
        "acc":         r["best_acc"] * 100.0,
        "sparsity":    r["final_sparsity"] * 100.0,
        "compression": r["hard_prune"]["compression_x"],
    })
pts = sorted(pts, key=lambda p: p["sparsity"])

plt.rcParams.update({
    "font.family":    "serif",
    "font.serif":     ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size":      12,
    "axes.titlesize": 16,
    "axes.labelsize": 13,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":      True,
    "grid.alpha":     0.22,
    "figure.dpi":     120,
    "savefig.dpi":    200,
    "savefig.bbox":   "tight",
})

fig, ax = plt.subplots(figsize=(12.5, 5.6))

spars = np.array([p["sparsity"] for p in pts])
accs  = np.array([p["acc"]      for p in pts])
compr = np.array([p["compression"] for p in pts])

ax.plot(spars, accs, "-", color="#5a5a5a", linewidth=1.6, alpha=0.6, zorder=1)
sc = ax.scatter(spars, accs, s=260, c=compr, cmap="viridis",
                edgecolor="white", linewidth=2, zorder=3)

colors   = ["#2e7d32", "#f57c00", "#c62828"]
labels   = ["max accuracy", "balanced", "max compression"]
# per-point text offsets in data units: (dx, dy), alignment
# each annotation is placed **below** its point so it never hits the title,
# and the last one sits to the **left** so it does not clip off the figure.
offsets  = [
    ( +3.5, -3.4, "left",  "top"),
    ( -2.5, -3.4, "right", "top"),
    ( -2.5, -3.2, "right", "top"),
]
for p, col, lab, (dx, dy, ha, va) in zip(pts, colors, labels, offsets):
    ax.annotate(
        f"$\\lambda$ = {p['lam']:.0e}   ({lab})\n"
        f"{p['acc']:.2f}% acc  |  "
        f"{p['sparsity']:.1f}% sparse  |  "
        f"{p['compression']:.2f}x smaller",
        xy=(p["sparsity"], p["acc"]),
        xytext=(p["sparsity"] + dx, p["acc"] + dy),
        textcoords="data",
        ha=ha, va=va,
        fontsize=10.5,
        fontweight="semibold",
        color=col,
        bbox=dict(boxstyle="round,pad=0.4", fc="white", ec=col, alpha=0.95),
        arrowprops=dict(arrowstyle="-", color=col, alpha=0.55),
    )

ax.set_xlabel("unstructured sparsity (%)", fontsize=13)
ax.set_ylabel("CIFAR-10 test accuracy (%)", fontsize=13)
ax.set_title(
    "Self-Pruning PrunableMixer: accuracy vs sparsity Pareto frontier",
    fontsize=14, pad=16,
)
ax.set_xlim(-4, 104)
ax.set_ylim(70, 88)

cbar = fig.colorbar(sc, ax=ax, pad=0.015, fraction=0.04)
cbar.set_label("model-size reduction  (x)", fontsize=11)

ax.axhline(83.86, linestyle=":", color="#2e7d32", alpha=0.35, linewidth=1.0)
ax.text(102, 83.86, "best accuracy", color="#2e7d32",
        fontsize=9, ha="right", va="bottom", alpha=0.75)

mdl = results["model"]
subtitle = (
    f"{mdl['architecture']} | {mdl['prunable_layers']} prunable layers | "
    f"{mdl['weight_params']:,} gated weights | "
    f"one-shot sweep across lambda in {{1e-7, 1e-6, 1e-5}}"
)
fig.text(0.5, -0.035, subtitle, ha="center", fontsize=10, color="#444")

fig.savefig(ROOT / "figures" / "hero_pareto.png")
print("Wrote figures/hero_pareto.png")
