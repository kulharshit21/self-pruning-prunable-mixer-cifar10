#
# Self-Pruning PrunableMixer - reproduction Makefile
#
# All commands use the repo root as the working directory. Portable across
# Linux, macOS, and (via mingw32-make or make in Git-Bash) Windows.
#

PYTHON ?= python

.PHONY: help install train verify structured hero report demo release clean fmt lint

help:
	@echo ""
	@echo "Self-Pruning PrunableMixer - available targets"
	@echo "  make install      install runtime dependencies"
	@echo "  make install-dev  install runtime + dev (ruff, black, pre-commit) deps"
	@echo "  make verify       verify reported results from JSON + checkpoints (fast, no GPU)"
	@echo "  make structured   compute structured-sparsity analysis + figure"
	@echo "  make hero         regenerate the README Pareto hero image"
	@echo "  make report       regenerate CASE_STUDY.docx + Excel dashboard + figures"
	@echo "  make demo         launch the Gradio demo on http://127.0.0.1:7860"
	@echo "  make train        run the full 3-lambda 100-epoch sweep (needs GPU, ~45 min on H100)"
	@echo "  make release      list the files to attach to a GitHub Release"
	@echo "  make fmt          run black + ruff --fix on the Python sources"
	@echo "  make lint         run ruff (check-only) on the Python sources"
	@echo ""

install:
	$(PYTHON) -m pip install -r requirements.txt

install-dev: install
	$(PYTHON) -m pip install ruff black pre-commit nbstripout
	pre-commit install

verify:
	$(PYTHON) verify_reported_results.py

structured:
	$(PYTHON) analyze_structured_sparsity.py

hero:
	$(PYTHON) _build_hero.py

report:
	$(PYTHON) regenerate_artifacts.py

demo:
	$(PYTHON) app.py

train:
	$(PYTHON) self_pruning_mlp_cifar10.py

release:
	@echo "Files to attach to a GitHub Release (created via 'gh release create v1.0-checkpoints'):"
	@ls -lh checkpoints/*.pt 2>/dev/null || echo "  (no checkpoints locally - train first)"
	@echo ""
	@echo "Suggested command:"
	@echo "  gh release create v1.0-checkpoints --title 'v1.0 - PrunableMixer checkpoints' \\"
	@echo "      --notes-file RELEASE_NOTES.md checkpoints/*.pt"

fmt:
	ruff check --fix --unsafe-fixes .
	black .

lint:
	ruff check .

clean:
	rm -rf __pycache__ .ruff_cache .pytest_cache
	@echo "Checkpoints, figures, outputs preserved. Run 'make train' to regenerate from scratch."
