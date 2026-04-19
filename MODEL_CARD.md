<!-- Model card formatted in the style of Hugging Face / Google Model Cards for Model Reporting. -->

# Model Card: Self-Pruning PrunableMixer (CIFAR-10)

| Field                       | Value                                                          |
|-----------------------------|----------------------------------------------------------------|
| **Model name**              | `self-pruning-prunable-mixer-cifar10`                          |
| **Model type**              | Feed-forward vision classifier (MLP-Mixer variant)             |
| **Base architecture**       | `PrunableMixer` - custom built for this case study             |
| **Task**                    | 10-way image classification on CIFAR-10 (32x32 RGB)            |
| **Version / commit**        | see `git log -1 --format=%H`                                   |
| **Author**                  | Harshit Kulkarni                                               |
| **License**                 | MIT                                                            |
| **Repository**              | https://github.com/kulharshit21/self-pruning-prunable-mixer-cifar10 |
| **Contact**                 | via GitHub issues on the repository above                       |

---

## 1. Intended use

### 1.1 Primary use

Research artefact for the Tredence AI Engineering internship case study on
**self-pruning neural networks**. The model is a *mechanism demonstrator*,
not a production classifier. Its purpose is to study:

- whether a learnable sigmoid gate per weight can discover a sparse
  sub-network while training end-to-end;
- whether the resulting sparsity survives hard pruning (zeroing gated weights);
- how accuracy trades against compression when the L1 sparsity coefficient
  $\lambda$ is swept.

### 1.2 Intended users

- **ML researchers** studying differentiable pruning, L0 relaxations,
  MLP-Mixer variants, lottery-ticket-style analyses, or structured
  sparsity emergence from unstructured penalties.
- **Engineers** evaluating whether a similar gating scheme could be layered
  on top of an existing linear layer in their own stack.
- **Hiring committees** evaluating the code quality, reproducibility, and
  scientific rigour of the submission.

### 1.3 Out-of-scope use

- **Safety-critical deployments.** The model was not validated for any
  downstream task beyond CIFAR-10 classification.
- **Medical imaging, biometric identification, surveillance, content
  moderation.** Not evaluated on any of these domains; CIFAR-10 is an
  animals-and-vehicles benchmark.
- **Claiming state-of-the-art** on CIFAR-10. The goal is mechanism quality
  under explicit constraints, not raw accuracy.

---

## 2. Training data

| Property                | Value                                                       |
|-------------------------|-------------------------------------------------------------|
| Dataset                 | CIFAR-10 (Krizhevsky, 2009)                                  |
| Images                  | 50,000 train / 10,000 test, 32x32 RGB                       |
| Classes                 | airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck |
| License                 | redistribution permitted for academic use (see dataset site) |
| Preprocessing           | per-channel normalisation with standard CIFAR-10 mean / std   |
| Augmentation (train)    | RandomCrop(32, pad=4), RandomHorizontalFlip, ColorJitter(0.1), RandomErasing (Cutout, p=0.25), MixUp (alpha=0.2) |
| Augmentation (test)     | none                                                        |
| No external data        | no pre-training, no extra labelled or unlabelled data        |

---

## 3. Evaluation

### 3.1 Metrics

| Metric                          | Definition                                                                          |
|---------------------------------|-------------------------------------------------------------------------------------|
| Top-1 test accuracy             | Standard classification accuracy on CIFAR-10 test split.                             |
| Unstructured sparsity           | Fraction of gate values $\sigma(g_{ij})$ below threshold $\tau = 10^{-2}$.           |
| Hard-pruning accuracy           | Accuracy after physically zeroing every $W_{ij}$ whose gate is below $\tau$.         |
| Hard-prune delta                | Hard-pruning accuracy minus soft accuracy.                                           |
| Row / column structured sparsity | Fraction of `Linear` output rows / input columns with *all* gates pruned.            |
| Dense-GEMM FLOP savings         | FLOP reduction realised by physically deleting dead rows and columns.                 |
| Compression                     | Dense fp32 byte-count / effective byte-count after hard pruning.                     |

### 3.2 Headline results (100-epoch single-seed run, reported in the paper)

| lambda  | Best acc | Sparsity | Hard acc | Hard drop | Compression | Row sparsity | Col sparsity | FLOP savings |
|:-------:|:--------:|:--------:|:--------:|:---------:|:-----------:|:------------:|:------------:|:------------:|
| 1e-07   | 83.86%   | 20.04%   | 83.89%   | -0.03%    | 1.25x       | 0.00%        | 0.00%        | 0.00%        |
| 1e-06   | 82.21%   | 88.91%   | 82.24%   | -0.03%    | 9.01x       | 42.02%       | 43.49%       | 57.77%       |
| 1e-05   | 76.18%   | 99.24%   | 76.11%   | +0.07%    | 128.57x     | 80.48%       | 78.68%       | 92.22%       |

All three runs satisfy the automatic sanity checks (non-trivial sparsity
range, monotonicity in $\lambda$, non-trivial accuracy, hard-prune drop
below 0.1 pp).

### 3.3 Variance / confidence intervals

**Not measured.** Reported numbers are from a single-seed run (seed=42) per
$\lambda$. A multi-seed variance study is flagged as future work; it would
require 6-9 additional 100-epoch runs to produce 95% confidence intervals
for each metric, which was outside the compute envelope of the case study.
Downstream users should treat the reported numbers as **point estimates**
and re-measure on their own seed sweep before drawing strong conclusions.

### 3.4 Where does the model fail?

- **Extreme sparsity regime** ($\lambda \geq 10^{-5}$): accuracy drops
  sharply below 80%. The progressive-threshold analysis (§3.6 of the paper)
  shows accuracy collapses once threshold exceeds ~0.1 because essential
  channel-mixing pathways lose quorum.
- **Classes with visual overlap** (e.g. cat vs dog, automobile vs truck).
  CIFAR-10 errors are concentrated in these pairs at all sparsity levels -
  we did not analyse this specifically, but it is well-known for the
  benchmark.
- **OOD inputs.** The model is a pure classifier; fed a non-CIFAR image it
  will still produce a 10-class softmax, and the probabilities should not
  be interpreted as uncertainty.

---

## 4. Architecture & parameters

| Component          | Value                                   |
|--------------------|-----------------------------------------|
| Input              | 3 x 32 x 32 RGB image                   |
| Patchification     | 4 x 4 patches -> 64 tokens of 48 pixels  |
| Token embedding    | `PrunableLinear(48 -> 768)`              |
| Mixer depth        | 12 blocks                               |
| Token-mix MLP      | `PrunableLinear(64 -> 256 -> 64)` + GELU |
| Channel-mix MLP    | `PrunableLinear(768 -> 3072 -> 768)` + GELU |
| Head               | LayerNorm + mean-pool + `PrunableLinear(768 -> 10)` |
| Prunable layers    | 50                                      |
| Prunable weights   | 57,060,864                              |
| Gate parameters    | 57,060,864                              |
| Total parameters   | 114,210,826                             |
| Dense fp32 size    | 435.7 MB                                |

---

## 5. Training

| Hyperparameter       | Value                                                     |
|----------------------|-----------------------------------------------------------|
| Optimiser            | AdamW, two parameter groups                               |
| LR (weights)         | 1e-3, cosine-annealed over 100 epochs                      |
| LR (gates)           | 1e-2, cosine-annealed                                     |
| Weight decay         | 5e-4 on weights, 0 on gates                               |
| Label smoothing      | 0.1                                                       |
| Gradient clip        | max-norm 1.0                                              |
| Gate init            | -2.0 (sigmoid ~ 0.119)                                    |
| Pruning threshold    | 1e-2                                                      |
| Lambda schedule      | 5 epoch warmup (CE only), 5 epoch linear ramp, hold       |
| Lambda sweep         | {1e-7, 1e-6, 1e-5}                                        |
| Precision            | bf16 autocast on CUDA Ampere+, fp32 elsewhere             |
| Seed                 | 42                                                        |
| Epochs               | 100 per $\lambda$                                         |
| Batch size           | auto-tuned (1024 on H100-80, 256 on 16 GB GPU, 64 on CPU) |
| Framework            | PyTorch 2.8                                               |

Training compute for the reported run: single NVIDIA H100-80GB,
~14 minutes per lambda, ~42 minutes total.

---

## 6. Environmental impact

A conservative estimate based on 42 GPU-minutes at ~700W actual power draw
on an H100-SXM (peak board power), times a data-centre PUE of 1.2 and the
EU average grid carbon intensity (~0.26 kgCO2/kWh):

- Energy: 0.42 h x 0.7 kW x 1.2 PUE ~ 0.35 kWh
- CO2 equivalent: ~0.09 kg

These numbers are sub-gram-scale per inference (the pruned deployment
artefacts are ~24 MB at lambda=1e-6, ~3.4 MB at lambda=1e-5) so the
dominant footprint is the training sweep itself, which is negligible.

---

## 7. Ethical considerations

- **Dataset representation.** CIFAR-10 is a vehicles-and-animals benchmark
  with no demographic content; standard ML-fairness concerns do not apply.
- **Dual-use risk.** Classifying 32x32 natural images is low-risk.
- **Licensing.** The model weights are released under MIT. CIFAR-10 itself
  is distributed by its authors for academic use.

---

## 8. How to use

Load a specific operating point:

```python
from mixer_lib import load_mixer, CIFAR10_CLASSES

net = load_mixer("checkpoints/mixer_lambda_1e-06_seed42.pt", device="cpu")
# pass a (B, 3, 32, 32) tensor normalised with CIFAR10_MEAN / CIFAR10_STD
# -> logits of shape (B, 10).
```

Reproduce every reported number without retraining:

```bash
python verify_reported_results.py
```

Interactive demo:

```bash
pip install gradio
python app.py
```

---

## 9. Citation

```bibtex
@techreport{kulkarni2026selfpruning,
  author    = {Kulkarni, Harshit},
  title     = {Self-Pruning {PrunableMixer} on {CIFAR-10}:
               A Case Study in Emergent Structured Sparsity
               from Unstructured Gate Regularisation},
  institution = {Tredence AI Engineering Internship - Case Study},
  year      = {2026},
  url       = {https://github.com/kulharshit21/self-pruning-prunable-mixer-cifar10}
}
```
