# v1.0 - PrunableMixer checkpoints

Three trained `state_dict` checkpoints for the `PrunableMixer` architecture,
one per operating point on the accuracy-sparsity Pareto frontier.

| file                               | lambda | CIFAR-10 acc | sparsity | size    |
|------------------------------------|:------:|:------------:|:--------:|:-------:|
| `mixer_lambda_1e-07_seed42.pt`     | 1e-07  | 83.86%       | 20.04%   | 435.8 MB |
| `mixer_lambda_1e-06_seed42.pt`     | 1e-06  | 82.21%       | 88.91%   | 435.8 MB |
| `mixer_lambda_1e-05_seed42.pt`     | 1e-05  | 76.18%       | 99.24%   | 435.8 MB |

The files are pure PyTorch `OrderedDict` state dicts - no custom pickle
classes, no lambda-captured globals, safe to load with `weights_only=False`.

## How to use

```bash
# 1. clone the repo
git clone https://github.com/kulharshit21/self-pruning-prunable-mixer-cifar10.git
cd self-pruning-prunable-mixer-cifar10

# 2. pull the checkpoints into ./checkpoints/
gh release download v1.0-checkpoints --dir checkpoints

# 3. verify everything
python verify_reported_results.py

# 4. play with the Gradio demo
pip install gradio
python app.py
```

Or, from Python directly:

```python
from mixer_lib import load_mixer
net = load_mixer("checkpoints/mixer_lambda_1e-06_seed42.pt", device="cpu")
```

## Provenance

- Trained with `self_pruning_mlp_cifar10.py` on a single NVIDIA H100-80GB.
- Seed 42, 100 epochs per lambda, bf16 autocast, AdamW + cosine-annealing LR.
- Full training log + per-epoch history available in `outputs/results_mlp.json`.
- The exact git commit that produced these weights is the one tagged
  `v1.0-checkpoints`.
