"""Concept heatmap visualization (paper Sec 3.6 / Eq. 6).

Project the inferred semantic coefficients S_hat back into the spatial domain, normalize
each concept map to [0, 1], upsample, and overlay it on the input image. Because each
column of S_hat corresponds to a named concept, every heatmap is human-interpretable.
"""

import os

import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
from tqdm.auto import tqdm

from config import CONFIG
from data_utils import clip_preprocess


def _heatmap(coeffs_k, grid, size=224):
    """One concept's coefficients (h*w,) -> normalized [0,1] heatmap upsampled to `size`."""
    m = coeffs_k.reshape(grid, grid)
    m = (m - m.min()) / (m.max() - m.min() + 1e-8)
    m = F.interpolate(m[None, None], size=size, mode="bilinear", align_corners=False)
    return m[0, 0].numpy()


def save_concept_overlays(images, S_hat, concepts, out_name="overlays.png", max_images=4):
    """Save a grid of per-concept heatmaps overlaid on the first few validation images."""
    grid, r = CONFIG["grid"], len(concepts)
    n_show = min(max_images, len(images))
    S_hat = S_hat.reshape(len(images), grid * grid, r)

    fig, axes = plt.subplots(n_show, r + 1, figsize=(2 * (r + 1), 2 * n_show))
    axes = np.atleast_2d(axes)
    for i in tqdm(range(n_show), desc="rendering overlays"):
        img = clip_preprocess(images[i])
        axes[i, 0].imshow(img)
        axes[i, 0].set_title("input", fontsize=6)
        axes[i, 0].axis("off")
        for k in range(r):
            axes[i, k + 1].imshow(img)
            axes[i, k + 1].imshow(_heatmap(S_hat[i, :, k], grid), cmap="jet", alpha=0.5)
            axes[i, k + 1].set_title(concepts[k], fontsize=5)
            axes[i, k + 1].axis("off")

    path = os.path.join(CONFIG["viz_dir"], out_name)
    plt.tight_layout()
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"[viz] saved {path}")
    return path


def plot_score_distributions(S_before, S_after, out_name="fig4_score_distributions.png"):
    """Fig 4: distribution of concept activation scores before vs after optimization.

    'Before' = CLIP-similarity initialization S; 'after' = the optimized coefficients
    S_hat. After optimization the scores disperse to match encoder activations rather
    than the initial CLIP similarities — i.e. the concept maps are driven by the model's
    internal representations, not CLIP biases.
    """
    b = S_before.flatten().numpy()
    a = S_after.flatten().numpy()
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.5))
    ax[0].hist(b, bins=60, color="tab:blue", density=True)
    ax[0].set_title("before optimization (CLIP init)")
    ax[1].hist(a, bins=60, color="tab:orange", density=True)
    ax[1].set_title("after optimization")
    for x in ax:
        x.set_xlabel("concept activation score")
    ax[0].set_ylabel("density")

    path = os.path.join(CONFIG["viz_dir"], out_name)
    plt.tight_layout()
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"[viz] saved {path}")
    return path
