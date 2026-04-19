"""
analyze_structured_sparsity.py
==============================

The training-time report in ``results_mlp.json`` gives **unstructured**
sparsity (fraction of individual gated weights below threshold). That number
tells you how much *storage* you can save, but not how much *compute* you can
save on commodity hardware.

This script reloads each saved checkpoint, computes

- **row sparsity**    = fraction of output-rows where every gate is pruned
  (i.e. entire output neurons that can be physically deleted).
- **column sparsity** = fraction of input-columns where every gate is pruned
  (input channels that can be physically deleted).

These counts are what would translate into real matmul FLOP reductions on
dense GEMM hardware without a sparse kernel.

Outputs
-------
- ``outputs/structured_sparsity.json``  - per-layer + per-lambda summary.
- ``figures/fig9_structured_sparsity.png`` - side-by-side bar chart
  (unstructured vs structured) per lambda.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from mixer_lib import annotate_paths, load_mixer

ROOT     = Path(__file__).parent
CKPT_DIR = ROOT / "checkpoints"
FIG_DIR  = ROOT / "figures"
OUT_DIR  = ROOT / "outputs"
OUT_DIR.mkdir(exist_ok=True)
FIG_DIR.mkdir(exist_ok=True)

LAMBDAS   = (1e-7, 1e-6, 1e-5)
THRESHOLD = 1e-2

plt.rcParams.update({
    "font.family":    "serif",
    "font.serif":     ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size":      11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":      True,
    "grid.alpha":     0.25,
    "legend.frameon": False,
    "figure.dpi":     120,
    "savefig.dpi":    180,
    "savefig.bbox":   "tight",
})


def layer_structured_stats(gate_scores: torch.Tensor, threshold: float) -> dict:
    """Given a gate_scores tensor of shape (out, in), compute structured counts."""
    gates   = torch.sigmoid(gate_scores)
    dead    = gates < threshold                    # (out, in) bool
    # row: all gates in that output row below threshold  -> neuron deletable
    dead_rows = dead.all(dim=1)                    # (out,)
    # column: all gates in that input column below threshold -> channel deletable
    dead_cols = dead.all(dim=0)                    # (in,)
    return {
        "out_features":  int(gate_scores.shape[0]),
        "in_features":   int(gate_scores.shape[1]),
        "unstructured":  float(dead.float().mean().item()),
        "row_sparsity":  float(dead_rows.float().mean().item()),
        "col_sparsity":  float(dead_cols.float().mean().item()),
        "dead_rows":     int(dead_rows.sum().item()),
        "dead_cols":     int(dead_cols.sum().item()),
    }


def analyze_checkpoint(lam: float) -> dict:
    fname = CKPT_DIR / f"mixer_lambda_{lam:.0e}_seed42.pt"
    if not fname.exists():
        return {"lambda": lam, "missing": True, "path": str(fname)}
    net = load_mixer(str(fname), device="cpu")
    annotate_paths(net)

    per_layer = []
    agg = {
        "unstructured": {"pruned": 0, "total": 0},
        "rows":         {"dead":   0, "total": 0},
        "cols":         {"dead":   0, "total": 0},
    }
    # compute retained FLOPs as fraction: for each layer (out,in),
    # dense-FLOPs proportional to out*in; structured-kept = live_rows * live_cols
    total_flops_dense = 0
    total_flops_struct = 0
    total_flops_unstruct = 0

    for idx, m in enumerate(net.prunable_layers()):
        st = layer_structured_stats(m.gate_scores.data, THRESHOLD)
        st["layer_idx"] = idx
        st["path"]      = getattr(m, "_path", "unknown")
        per_layer.append(st)

        out_f, in_f = st["out_features"], st["in_features"]
        dense_flops = out_f * in_f
        live_rows   = out_f - st["dead_rows"]
        live_cols   = in_f  - st["dead_cols"]
        # structured-retained FLOPs (after deleting dead rows & cols)
        struct_flops = live_rows * live_cols
        # unstructured-retained FLOPs (if we could do fully-sparse matmul)
        unstruct_flops = dense_flops * (1.0 - st["unstructured"])

        total_flops_dense    += dense_flops
        total_flops_struct   += struct_flops
        total_flops_unstruct += int(unstruct_flops)

        # aggregate unstructured
        agg["unstructured"]["pruned"] += int(round(st["unstructured"] * dense_flops))
        agg["unstructured"]["total"]  += dense_flops
        agg["rows"]["dead"]  += st["dead_rows"]
        agg["rows"]["total"] += out_f
        agg["cols"]["dead"]  += st["dead_cols"]
        agg["cols"]["total"] += in_f

    summary = {
        "lambda":                  lam,
        "threshold":               THRESHOLD,
        "global_unstructured":     agg["unstructured"]["pruned"] / max(agg["unstructured"]["total"], 1),
        "global_row_sparsity":     agg["rows"]["dead"] / max(agg["rows"]["total"], 1),
        "global_col_sparsity":     agg["cols"]["dead"] / max(agg["cols"]["total"], 1),
        "flops_dense":             total_flops_dense,
        "flops_retained_unstruct": total_flops_unstruct,
        "flops_retained_struct":   total_flops_struct,
        "flop_savings_unstruct":   1.0 - total_flops_unstruct / total_flops_dense,
        "flop_savings_struct":     1.0 - total_flops_struct   / total_flops_dense,
        "per_layer":               per_layer,
    }
    return summary


def per_path_structured(summary: dict) -> dict:
    """Roll up row/col sparsity by semantic path."""
    buckets: dict[str, dict] = {}
    for lay in summary["per_layer"]:
        p = lay["path"]
        b = buckets.setdefault(p, {
            "dead_rows": 0, "total_rows": 0,
            "dead_cols": 0, "total_cols": 0,
            "unstruct_pruned": 0, "unstruct_total": 0,
            "layers": 0,
        })
        b["dead_rows"]      += lay["dead_rows"]
        b["total_rows"]     += lay["out_features"]
        b["dead_cols"]      += lay["dead_cols"]
        b["total_cols"]     += lay["in_features"]
        b["unstruct_pruned"] += int(round(lay["unstructured"] * lay["out_features"] * lay["in_features"]))
        b["unstruct_total"]  += lay["out_features"] * lay["in_features"]
        b["layers"]         += 1
    for p, b in buckets.items():
        b["row_sparsity"]          = b["dead_rows"] / max(b["total_rows"], 1)
        b["col_sparsity"]          = b["dead_cols"] / max(b["total_cols"], 1)
        b["unstructured_sparsity"] = b["unstruct_pruned"] / max(b["unstruct_total"], 1)
    return buckets


def plot_structured_vs_unstructured(all_summaries: list[dict],
                                    out_path: Path) -> None:
    valid = [s for s in all_summaries if not s.get("missing")]
    n = len(valid)
    if n == 0:
        return

    labels = [f"lambda = {s['lambda']:.0e}" for s in valid]
    unstruct = [s["global_unstructured"] * 100 for s in valid]
    rows     = [s["global_row_sparsity"] * 100 for s in valid]
    cols     = [s["global_col_sparsity"] * 100 for s in valid]
    flops_s  = [s["flop_savings_struct"] * 100 for s in valid]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.2))

    x = np.arange(n)
    w = 0.22
    ax1.bar(x - 1.5*w, unstruct, w, label="unstructured (weight-level)", color="#6c8ebf")
    ax1.bar(x - 0.5*w, rows,     w, label="row (output neuron)",         color="#82b366")
    ax1.bar(x + 0.5*w, cols,     w, label="column (input channel)",      color="#d79b00")
    ax1.bar(x + 1.5*w, flops_s,  w, label="dense-GEMM FLOP savings",     color="#b85450")

    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.set_ylabel("fraction pruned (%)")
    ax1.set_ylim(0, 100)
    ax1.set_title("Structured vs unstructured sparsity")
    ax1.legend(loc="upper left", fontsize=9)
    for xi, vals in zip(x, zip(unstruct, rows, cols, flops_s)):
        for dx, v in zip((-1.5*w, -0.5*w, 0.5*w, 1.5*w), vals):
            ax1.text(xi + dx, v + 1.5, f"{v:.1f}", ha="center",
                     va="bottom", fontsize=8)

    # right panel: structured FLOP-savings per path (for the best-compression lambda)
    best = max(valid, key=lambda s: s["flop_savings_struct"])
    paths = per_path_structured(best)
    path_order = ["patch_embed", "token_mix", "channel_mix", "classifier"]
    path_order = [p for p in path_order if p in paths]
    rows_pct = [paths[p]["row_sparsity"] * 100 for p in path_order]
    cols_pct = [paths[p]["col_sparsity"] * 100 for p in path_order]
    unstr_pct = [paths[p]["unstructured_sparsity"] * 100 for p in path_order]
    x2 = np.arange(len(path_order))
    ax2.bar(x2 - w, unstr_pct, w, label="unstructured", color="#6c8ebf")
    ax2.bar(x2,     rows_pct,  w, label="rows",         color="#82b366")
    ax2.bar(x2 + w, cols_pct,  w, label="cols",         color="#d79b00")
    ax2.set_xticks(x2)
    ax2.set_xticklabels(path_order, rotation=15)
    ax2.set_ylabel("fraction pruned (%)")
    ax2.set_ylim(0, 100)
    ax2.set_title(f"Per-path breakdown (lambda = {best['lambda']:.0e})")
    ax2.legend(loc="upper left", fontsize=9)

    fig.suptitle("Structured sparsity analysis - physically-deletable "
                 "rows / columns from gate tensors", y=1.02, fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main() -> int:
    if not CKPT_DIR.exists():
        print(f"ERROR: no {CKPT_DIR} - download from GitHub Releases first.")
        return 2

    print("Analyzing structured sparsity for each checkpoint ...")
    all_summaries = []
    for lam in LAMBDAS:
        s = analyze_checkpoint(lam)
        all_summaries.append(s)
        if s.get("missing"):
            print(f"  lambda={lam:.0e} - checkpoint not found, skipped")
            continue
        print(
            f"  lambda={lam:.0e}  "
            f"unstructured={s['global_unstructured']*100:6.2f}%   "
            f"rows={s['global_row_sparsity']*100:6.2f}%   "
            f"cols={s['global_col_sparsity']*100:6.2f}%   "
            f"dense-GEMM FLOP savings={s['flop_savings_struct']*100:6.2f}%"
        )

    # roll-up per-path for every lambda (for the JSON)
    for s in all_summaries:
        if s.get("missing"):
            continue
        s["per_path_structured"] = per_path_structured(s)

    json_out = OUT_DIR / "structured_sparsity.json"
    # drop per_layer from top-level to keep the JSON compact (path-level is enough
    # for the write-up). We keep per_layer nested under a key for reproducibility.
    compact = []
    for s in all_summaries:
        if s.get("missing"):
            compact.append(s)
            continue
        c = {k: v for k, v in s.items() if k != "per_layer"}
        c["per_layer_count"] = len(s["per_layer"])
        compact.append(c)
    with json_out.open("w", encoding="utf-8") as f:
        json.dump(compact, f, indent=2)
    print(f"\nWrote {json_out}")

    out_png = FIG_DIR / "fig9_structured_sparsity.png"
    plot_structured_vs_unstructured(all_summaries, out_png)
    print(f"Wrote {out_png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
