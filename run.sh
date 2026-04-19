#!/usr/bin/env bash
#
# run.sh - one-command entry point for the Self-Pruning PrunableMixer repo.
#
# Usage:
#   ./run.sh             # equivalent to ./run.sh verify
#   ./run.sh verify      # reproduce reported numbers from json + checkpoints
#   ./run.sh demo        # launch the Gradio demo
#   ./run.sh train       # full 3-lambda sweep (needs a GPU)
#   ./run.sh report      # regenerate docx + xlsx + figures
#   ./run.sh structured  # structured-sparsity analysis + fig9
#   ./run.sh hero        # rebuild the README Pareto hero image
#   ./run.sh all         # verify + structured + hero + report
#
set -euo pipefail

cmd="${1:-verify}"
PY="${PYTHON:-python}"

cd "$(dirname "$0")"

case "$cmd" in
  verify)     "$PY" verify_reported_results.py ;;
  structured) "$PY" analyze_structured_sparsity.py ;;
  hero)       "$PY" _build_hero.py ;;
  report)     "$PY" regenerate_artifacts.py ;;
  demo)       "$PY" app.py ;;
  train)      "$PY" self_pruning_mlp_cifar10.py ;;
  all)
      "$PY" verify_reported_results.py
      "$PY" analyze_structured_sparsity.py
      "$PY" _build_hero.py
      "$PY" regenerate_artifacts.py
      ;;
  *)  echo "unknown target: $cmd"; exit 2 ;;
esac
