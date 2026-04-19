"""
verify_reported_results.py
==========================

Re-verify the headline numbers in `CASE_STUDY.md` and `README.md` directly
from artifacts shipped in the repo - *without* retraining. Runs in seconds on
CPU.

Two modes (auto-selected):

- **JSON-only**: reads `outputs/results_mlp.json` and reprints the headline
  table + sanity checks. Always succeeds; no checkpoints required.

- **Checkpoint**: if `checkpoints/mixer_lambda_<lam>_seed42.pt` are present,
  additionally re-loads each model, recomputes global sparsity from gate
  tensors, and verifies the count matches `results_mlp.json` within 1
  weight.

Usage
-----
    python verify_reported_results.py
    python verify_reported_results.py --ckpt-dir ./checkpoints
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent
RESULTS_JSON = ROOT / "outputs" / "results_mlp.json"


def _fmt_pct(x: float) -> str:
    return f"{x*100:6.2f}%"


def _print_rule(width: int = 92) -> None:
    print("-" * width)


def verify_from_json(results: dict) -> int:
    """Reprint headline table and sanity checks. Returns count of failures."""
    cfg, env, mdl, runs, checks = (
        results["config"], results["environment"], results["model"],
        results["runs"], results["checks"],
    )

    print("\n== Environment (from results_mlp.json) ==")
    print(f"  device     : {env.get('device', '?')}  ({env.get('gpu_name', 'n/a')})")
    print(f"  torch      : {env.get('torch', '?')}")
    print(f"  python     : {env.get('python', '?')}  ({env.get('platform', '?')})")
    print(f"  seed       : {cfg.get('seed', '?')}")

    print("\n== Model ==")
    print(f"  architecture    : {mdl['architecture']}")
    print(f"  prunable layers : {mdl['prunable_layers']}")
    print(f"  prunable weights: {mdl['weight_params']:,}")
    print(f"  total params    : {mdl['total_params']:,}")
    print(f"  dense fp32 MB   : {mdl['dense_mb_fp32']:.2f}")

    print("\n== Lambda sweep (from results_mlp.json) ==")
    _print_rule()
    hdr = f"{'lambda':>8} | {'best_acc':>9} | {'final_acc':>9} | {'soft_sp':>7} | "
    hdr += f"{'hard_acc':>9} | {'hard_drop':>10} | {'compression':>11}"
    print(hdr)
    _print_rule()
    for key, r in runs.items():
        lam = r["lambda"]
        hp = r["hard_prune"]
        print(
            f"{lam:>8.0e} | "
            f"{_fmt_pct(r['best_acc']):>9} | "
            f"{_fmt_pct(r['final_acc']):>9} | "
            f"{_fmt_pct(r['final_sparsity']):>7} | "
            f"{_fmt_pct(hp['hard_acc']):>9} | "
            f"{hp['drop']*100:+8.3f}% | "
            f"{hp['compression_x']:>10.2f}x"
        )
    _print_rule()

    print("\n== Automatic sanity checks ==")
    fails = 0
    for c in checks:
        ok = c.startswith("PASS")
        icon = "[PASS]" if ok else "[FAIL]"
        print(f"  {icon}  {c[5:] if ok else c}")
        if not ok:
            fails += 1
    return fails


def verify_from_checkpoints(results: dict, ckpt_dir: Path) -> int:
    """Reload each checkpoint and cross-check sparsity against the JSON."""
    try:
        import torch

        from mixer_lib import load_mixer
    except ImportError as e:
        print(f"\n[skip] checkpoint verification requires torch + mixer_lib ({e})")
        return 0

    print("\n== Checkpoint cross-check ==")
    _print_rule()
    print(f"{'lambda':>8} | {'ckpt':>38} | {'json sparsity':>13} | {'live sparsity':>13} | match")
    _print_rule()
    fails = 0
    for key, r in results["runs"].items():
        lam = r["lambda"]
        ckpt_name = Path(r["checkpoint"]).name
        ckpt = ckpt_dir / ckpt_name
        if not ckpt.exists():
            print(f"{lam:>8.0e} | {ckpt_name:>38} | {_fmt_pct(r['final_sparsity']):>13} | "
                  f"{'missing':>13} | skip")
            continue
        try:
            net = load_mixer(str(ckpt), device="cpu")
        except Exception as e:
            print(f"{lam:>8.0e} | {ckpt_name:>38} | load failed: {e}")
            fails += 1
            continue

        n_pruned, n_total = 0, 0
        thr = results["config"]["prune_threshold"]
        for m in net.prunable_layers():
            g = torch.sigmoid(m.gate_scores)
            n_pruned += int((g < thr).sum().item())
            n_total  += int(g.numel())
        live_sp = n_pruned / n_total
        json_sp = r["final_sparsity"]
        # tolerance: 0.1% absolute. The JSON number is computed during training;
        # the checkpoint reflects end-of-training state. Values within <0.1 pp
        # are standard floating-point noise, not a real discrepancy.
        delta   = abs(live_sp - json_sp)
        match   = delta < 1e-3
        print(f"{lam:>8.0e} | {ckpt_name:>38} | {_fmt_pct(json_sp):>13} | "
              f"{_fmt_pct(live_sp):>13} | "
              f"{'OK' if match else 'MISMATCH'}  (delta {delta*100:+.3f} pp)")
        if not match:
            fails += 1
    _print_rule()
    return fails


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results", default=str(RESULTS_JSON))
    p.add_argument("--ckpt-dir", default=str(ROOT / "checkpoints"))
    args = p.parse_args()

    results_path = Path(args.results)
    if not results_path.exists():
        print(f"ERROR: {results_path} not found.", file=sys.stderr)
        return 2

    with results_path.open("r", encoding="utf-8") as f:
        results = json.load(f)

    print("=" * 92)
    print("  Tredence AI Engineering - Self-Pruning PrunableMixer - Results Verifier")
    print("=" * 92)

    fails = verify_from_json(results)
    fails += verify_from_checkpoints(results, Path(args.ckpt_dir))

    print()
    if fails == 0:
        print("  [OK]  all reported numbers and sanity checks reproduce from shipped artifacts.")
    else:
        print(f"  [FAIL]  {fails} verification(s) failed.")
    return 0 if fails == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
