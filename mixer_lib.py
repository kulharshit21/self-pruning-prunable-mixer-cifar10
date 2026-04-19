"""
mixer_lib.py
============

Minimal, side-effect-free re-implementation of the PrunableMixer classes so
utility scripts (``verify_reported_results.py``, ``analyze_structured_sparsity.py``,
``app.py``) can instantiate the model and load checkpoints without triggering
the training pipeline baked into ``self_pruning_mlp_cifar10.py``.

The class definitions are byte-compatible with the trained checkpoints in
``checkpoints/mixer_lambda_*_seed42.pt`` - same parameter names, same shapes,
same forward semantics.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Config mirror (defaults identical to the training Config in the main script)
# ---------------------------------------------------------------------------
@dataclass
class MixerConfig:
    image_size:     int   = 32
    image_channels: int   = 3
    patch_size:     int   = 4
    mixer_dim:      int   = 768
    mixer_depth:    int   = 12
    token_hidden:   int   = 256
    channel_hidden: int   = 3072
    num_classes:    int   = 10
    dropout:        float = 0.1
    gate_init:      float = -2.0


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class PrunableLinear(nn.Module):
    """Linear layer with learnable per-weight sigmoid gate.

        gates = sigmoid(gate_scores)
        W_eff = weight * gates
        y     = x @ W_eff.T + bias
    """

    def __init__(self, in_features: int, out_features: int,
                 bias: bool = True, gate_init: float = -2.0) -> None:
        super().__init__()
        self.in_features  = in_features
        self.out_features = out_features
        self.weight       = nn.Parameter(torch.empty(out_features, in_features))
        self.bias         = nn.Parameter(torch.zeros(out_features)) if bias else None
        self.gate_scores  = nn.Parameter(torch.full_like(self.weight, float(gate_init)))
        nn.init.kaiming_uniform_(self.weight, a=0.0, mode="fan_in",
                                 nonlinearity="relu")

    def gates(self) -> torch.Tensor:
        return torch.sigmoid(self.gate_scores)

    def pruned_weight(self) -> torch.Tensor:
        return self.weight * self.gates()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.pruned_weight(), self.bias)


class MlpBlock(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int,
                 dropout: float, gate_init: float) -> None:
        super().__init__()
        self.fc1   = PrunableLinear(in_dim, hidden_dim, gate_init=gate_init)
        self.act   = nn.GELU()
        self.drop1 = nn.Dropout(dropout)
        self.fc2   = PrunableLinear(hidden_dim, in_dim, gate_init=gate_init)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop2(self.fc2(self.drop1(self.act(self.fc1(x)))))


class MixerBlock(nn.Module):
    def __init__(self, num_tokens: int, dim: int, token_hidden: int,
                 channel_hidden: int, dropout: float, gate_init: float) -> None:
        super().__init__()
        self.norm1       = nn.LayerNorm(dim)
        self.token_mlp   = MlpBlock(num_tokens, token_hidden, dropout, gate_init)
        self.norm2       = nn.LayerNorm(dim)
        self.channel_mlp = MlpBlock(dim, channel_hidden, dropout, gate_init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.norm1(x).transpose(1, 2)
        y = self.token_mlp(y).transpose(1, 2)
        x = x + y
        x = x + self.channel_mlp(self.norm2(x))
        return x


class PrunableMixer(nn.Module):
    def __init__(self, cfg: MixerConfig) -> None:
        super().__init__()
        assert cfg.image_size % cfg.patch_size == 0
        self.patch_size = cfg.patch_size
        self.image_size = cfg.image_size
        self.channels   = cfg.image_channels
        self.num_tokens = (cfg.image_size // cfg.patch_size) ** 2
        patch_dim       = cfg.image_channels * cfg.patch_size * cfg.patch_size

        self.patch_embed = PrunableLinear(patch_dim, cfg.mixer_dim,
                                          gate_init=cfg.gate_init)
        self.blocks = nn.ModuleList([
            MixerBlock(self.num_tokens, cfg.mixer_dim,
                       cfg.token_hidden, cfg.channel_hidden,
                       cfg.dropout, cfg.gate_init)
            for _ in range(cfg.mixer_depth)
        ])
        self.norm       = nn.LayerNorm(cfg.mixer_dim)
        self.classifier = PrunableLinear(cfg.mixer_dim, cfg.num_classes,
                                         gate_init=cfg.gate_init)

    def _patchify(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        p = self.patch_size
        x = x.view(B, C, H // p, p, W // p, p)
        x = x.permute(0, 2, 4, 1, 3, 5).contiguous()
        return x.view(B, self.num_tokens, C * p * p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.view(-1, self.channels, self.image_size, self.image_size)
        x = self._patchify(x)
        x = self.patch_embed(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        x = x.mean(dim=1)
        return self.classifier(x)

    def prunable_layers(self) -> list[PrunableLinear]:
        return [m for m in self.modules() if isinstance(m, PrunableLinear)]


# ---------------------------------------------------------------------------
# Path annotation + helpers
# ---------------------------------------------------------------------------
def annotate_paths(net: PrunableMixer) -> None:
    """Tag each PrunableLinear with a ``_path`` attribute.

    Values: ``'patch_embed' | 'token_mix' | 'channel_mix' | 'classifier'``.
    """
    for name, m in net.named_modules():
        if not isinstance(m, PrunableLinear):
            continue
        if name == "patch_embed":
            m._path = "patch_embed"
        elif name == "classifier":
            m._path = "classifier"
        elif ".token_mlp." in name:
            m._path = "token_mix"
        elif ".channel_mlp." in name:
            m._path = "channel_mix"
        else:
            m._path = "unknown"


def load_mixer(ckpt_path: str, device: str | torch.device = "cpu",
               cfg: MixerConfig | None = None) -> PrunableMixer:
    """Instantiate a PrunableMixer and load the state-dict from ``ckpt_path``."""
    cfg = cfg or MixerConfig()
    net = PrunableMixer(cfg)
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    net.load_state_dict(state, strict=True)
    annotate_paths(net)
    net.eval()
    return net.to(device)


CIFAR10_CLASSES = (
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
)

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = (0.2470, 0.2435, 0.2616)
