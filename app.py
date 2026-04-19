"""
app.py  -  Gradio demo for the Self-Pruning PrunableMixer (CIFAR-10)
====================================================================

A single-file web UI that:

1. Lets the user pick one of three operating points (lambda = 1e-7 / 1e-6 /
   1e-5), each corresponding to a different accuracy / sparsity trade-off.
2. Accepts an uploaded image (or a sample CIFAR-10 test image).
3. Predicts the class, shows top-5 probabilities.
4. Displays the **live gate-sparsity statistics** of the loaded checkpoint
   (unstructured, row-wise, column-wise) so reviewers can see, in real time,
   that the sparsity claims in CASE_STUDY.md are a property of the shipped
   weights.

Run locally:

    pip install gradio torch torchvision
    python app.py                      # opens a local browser window

The checkpoints are large (~435 MB each) and live outside git; this script
gracefully falls back to an "untrained" model when none are present.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

from mixer_lib import (
    CIFAR10_CLASSES,
    CIFAR10_MEAN,
    CIFAR10_STD,
    MixerConfig,
    PrunableMixer,
    annotate_paths,
    load_mixer,
)

ROOT      = Path(__file__).parent
CKPT_DIR  = ROOT / "checkpoints"
LAMBDAS   = {
    "lambda = 1e-7  (max accuracy  ~83.9%,  sparsity ~20%)":  1e-7,
    "lambda = 1e-6  (balanced       ~82.2%,  sparsity ~89%)": 1e-6,
    "lambda = 1e-5  (max compression~76.2%,  sparsity ~99%)": 1e-5,
}
THRESHOLD = 1e-2

_PREPROCESS = T.Compose([
    T.Resize(32),
    T.CenterCrop(32),
    T.ToTensor(),
    T.Normalize(CIFAR10_MEAN, CIFAR10_STD),
])


def _sparsity_stats(net: PrunableMixer) -> dict:
    """Global unstructured + structured (row/col) sparsity."""
    n_pruned = n_total = 0
    dead_rows = total_rows = 0
    dead_cols = total_cols = 0
    for m in net.prunable_layers():
        g = torch.sigmoid(m.gate_scores)
        dead = g < THRESHOLD
        n_pruned   += int(dead.sum().item())
        n_total    += int(dead.numel())
        dead_rows  += int(dead.all(dim=1).sum().item())
        total_rows += int(dead.shape[0])
        dead_cols  += int(dead.all(dim=0).sum().item())
        total_cols += int(dead.shape[1])
    return {
        "unstructured": n_pruned / max(n_total, 1),
        "row":          dead_rows / max(total_rows, 1),
        "col":          dead_cols / max(total_cols, 1),
        "pruned":       n_pruned,
        "total":        n_total,
    }


_CACHE: dict[float, PrunableMixer] = {}


def _resolve(lam: float) -> Optional[PrunableMixer]:
    if lam in _CACHE:
        return _CACHE[lam]
    ckpt = CKPT_DIR / f"mixer_lambda_{lam:.0e}_seed42.pt"
    if not ckpt.exists():
        return None
    net = load_mixer(str(ckpt), device="cpu")
    _CACHE[lam] = net
    return net


def _blank_model() -> PrunableMixer:
    """Fallback when no checkpoint is present (shows the pipeline end-to-end)."""
    net = PrunableMixer(MixerConfig())
    annotate_paths(net)
    net.eval()
    return net


def predict(image: Image.Image, lam_label: str):
    lam = LAMBDAS[lam_label]
    net = _resolve(lam) or _blank_model()
    is_trained = lam in _CACHE

    if image is None:
        return (
            None,
            ("No image provided. Upload a 32x32 (or larger) image, or click one "
             "of the example thumbnails below."),
        )

    img = image.convert("RGB")
    x   = _PREPROCESS(img).unsqueeze(0)

    with torch.no_grad():
        logits = net(x)
        probs  = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()

    top5 = np.argsort(probs)[::-1][:5]
    probs_dict = {CIFAR10_CLASSES[i]: float(probs[i]) for i in top5}

    st = _sparsity_stats(net)
    status = [
        "### Live gate statistics for selected checkpoint",
        "",
        f"- **Checkpoint** : `mixer_lambda_{lam:.0e}_seed42.pt` "
        + ("- loaded" if is_trained else "- NOT FOUND, using untrained fallback"),
        f"- **Unstructured sparsity** : {st['unstructured']*100:6.2f}%  "
        f"({st['pruned']:,} of {st['total']:,} gates below {THRESHOLD})",
        f"- **Row (neuron-level) sparsity** : {st['row']*100:6.2f}%",
        f"- **Column (input-channel) sparsity** : {st['col']*100:6.2f}%",
    ]
    if not is_trained:
        status.append("")
        status.append(
            "> Checkpoints are ~435 MB each and distributed via GitHub Releases. "
            "Download them into `checkpoints/` to enable real predictions:"
        )
        status.append(
            "> `gh release download v1.0-checkpoints --dir checkpoints`"
        )
    return probs_dict, "\n".join(status)


def build_ui():
    import gradio as gr

    tradeoff_md = (
        "# Self-Pruning PrunableMixer - live demo\n"
        "\n"
        "Three checkpoints, three operating points on the accuracy-sparsity "
        "Pareto frontier. Switching between them reloads the full model (50 "
        "gated linear layers, ~57M learnable gates) and reruns the forward "
        "pass.\n"
        "\n"
        "| lambda | test accuracy | unstructured sparsity | model-size reduction |\n"
        "|:------:|:-------------:|:---------------------:|:--------------------:|\n"
        "| 1e-7   | 83.86%        | 20.0%                 | 1.25x                |\n"
        "| 1e-6   | 82.21%        | 88.9%                 | 9.01x                |\n"
        "| 1e-5   | 76.18%        | 99.2%                 | 128.57x              |\n"
    )

    default_lam = list(LAMBDAS.keys())[1]  # balanced

    with gr.Blocks(title="Self-Pruning PrunableMixer (CIFAR-10)",
                   theme=gr.themes.Soft()) as ui:
        gr.Markdown(tradeoff_md)
        with gr.Row():
            with gr.Column(scale=1):
                img_in = gr.Image(type="pil", label="Input image (CIFAR-10 class)")
                lam_dd = gr.Dropdown(list(LAMBDAS.keys()), value=default_lam,
                                     label="Operating point")
                btn    = gr.Button("Classify", variant="primary")
            with gr.Column(scale=1):
                prob_out = gr.Label(num_top_classes=5, label="Top-5 predictions")
                stat_out = gr.Markdown("Select a lambda to inspect its sparsity.")

        btn.click(predict, inputs=[img_in, lam_dd], outputs=[prob_out, stat_out])
        lam_dd.change(predict, inputs=[img_in, lam_dd], outputs=[prob_out, stat_out])
        img_in.change(predict, inputs=[img_in, lam_dd], outputs=[prob_out, stat_out])

        gr.Markdown(
            "---\n"
            "Source: [github.com/kulharshit21/self-pruning-prunable-mixer-cifar10]"
            "(https://github.com/kulharshit21/self-pruning-prunable-mixer-cifar10).  "
            "See `CASE_STUDY.md` for the full method write-up.\n"
        )

    return ui


def main() -> None:
    ui = build_ui()
    # SERVER_NAME / SERVER_PORT overridable for Spaces / Docker deployments.
    ui.launch(
        server_name=os.environ.get("GRADIO_SERVER_NAME", "127.0.0.1"),
        server_port=int(os.environ.get("GRADIO_SERVER_PORT", "7860")),
        show_error=True,
    )


if __name__ == "__main__":
    main()
