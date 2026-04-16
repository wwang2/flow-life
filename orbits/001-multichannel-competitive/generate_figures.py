#!/usr/bin/env python3
"""
Generate narrative.png and results.png figures for orbit 001-multichannel-competitive.

narrative.png: 4-panel showing baseline vs multi-channel recovery from damage
results.png: metric breakdown per seed
"""
import sys
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.titleweight": "medium",
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "axes.grid": True,
    "grid.alpha": 0.15,
    "grid.linewidth": 0.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titlepad": 10.0,
    "axes.labelpad": 6.0,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "legend.frameon": False,
    "legend.borderpad": 0.3,
    "legend.handletextpad": 0.5,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "figure.constrained_layout.use": True,
})

import torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "solution"))
from substrate import MultiChannelCompetitiveLenia
import math


def generate_narrative(output_path):
    """Generate a 4-panel figure showing damage recovery.

    Top row: baseline (single-kernel standard Lenia) - damage -> no recovery
    Bottom row: our method (two-kernel gated) - damage -> recovery
    """
    grid_size = 256
    device = torch.device("cpu")

    # Our substrate
    sub = MultiChannelCompetitiveLenia(grid_size=grid_size)
    params = sub.get_default_params()

    # Create initial blob and evolve to steady state
    H, W = grid_size, grid_size
    y = np.arange(H) - 128
    x = np.arange(W) - 128
    yy, xx = np.meshgrid(y, x, indexing="ij")
    blob = np.exp(-0.5 * ((yy / 18) ** 2 + (xx / 14) ** 2))
    state = np.clip(blob + np.random.RandomState(42).randn(H, W) * 0.02, 0, 1).astype(np.float32)[np.newaxis]

    # Evolve to steady state
    t = torch.tensor(state, dtype=torch.float32, device=device)
    for i in range(300):
        t = sub.update_step(t, params)
        t = t.clamp(0, 1)
    evolved = t.detach().cpu().numpy()

    # Capture pre-damage state
    pre_mf = evolved.sum(axis=0) if evolved.ndim == 3 else evolved
    pre_mx = pre_mf.max()

    # Apply damage
    mask = pre_mf > 0.01
    n_active = mask.sum()
    side = int(math.sqrt(n_active * 0.2))
    rows, cols = np.where(mask)
    rmin, rmax = rows.min(), rows.max()
    cmin, cmax = cols.min(), cols.max()
    r0 = rmin + (rmax - rmin) // 3
    c0 = cmin + (cmax - cmin) // 3
    r1 = min(r0 + side, H)
    c1 = min(c0 + side, W)

    damaged = evolved.copy()
    damaged[:, r0:r1, c0:c1] = 0.0
    damaged_mf = damaged.sum(axis=0) if damaged.ndim == 3 else damaged

    # Recover with our substrate
    sub2 = MultiChannelCompetitiveLenia(grid_size=grid_size)
    t_r = torch.tensor(damaged, dtype=torch.float32, device=device)
    for i in range(500):
        t_r = sub2.update_step(t_r, params)
        t_r = t_r.clamp(0, 1)
    recovered = t_r.detach().cpu().numpy()
    recovered_mf = recovered.sum(axis=0) if recovered.ndim == 3 else recovered

    # Also simulate fission (longer run)
    sub3 = MultiChannelCompetitiveLenia(grid_size=grid_size)
    t_f = torch.tensor(evolved, dtype=torch.float32, device=device)
    for i in range(5000):
        t_f = sub3.update_step(t_f, params)
        t_f = t_f.clamp(0, 1)
    fissioned = t_f.detach().cpu().numpy()
    fissioned_mf = fissioned.sum(axis=0) if fissioned.ndim == 3 else fissioned

    # Create 2x2 figure
    fig, axes = plt.subplots(2, 2, figsize=(10, 10), constrained_layout=True)

    vmax = max(pre_mx, damaged_mf.max(), recovered_mf.max(), fissioned_mf.max()) + 1e-8

    ax = axes[0, 0]
    ax.imshow(pre_mf, cmap="inferno", vmin=0, vmax=vmax)
    ax.set_title("(a) Steady-state organism")
    ax.axis("off")
    ax.text(-0.12, 1.05, "(a)", transform=ax.transAxes, fontsize=14, fontweight="bold")

    ax = axes[0, 1]
    ax.imshow(damaged_mf, cmap="inferno", vmin=0, vmax=vmax)
    ax.set_title("(b) After 20% damage")
    ax.axis("off")
    ax.text(-0.12, 1.05, "(b)", transform=ax.transAxes, fontsize=14, fontweight="bold")
    # Draw damage region
    rect = plt.Rectangle((c0, r0), c1 - c0, r1 - r0, linewidth=1.5,
                          edgecolor="cyan", facecolor="none", linestyle="--")
    ax.add_patch(rect)

    ax = axes[1, 0]
    ax.imshow(recovered_mf, cmap="inferno", vmin=0, vmax=vmax)
    ax.set_title("(c) After 500-step recovery")
    ax.axis("off")
    ax.text(-0.12, 1.05, "(c)", transform=ax.transAxes, fontsize=14, fontweight="bold")

    ax = axes[1, 1]
    ax.imshow(fissioned_mf, cmap="inferno", vmin=0, vmax=vmax)
    ax.set_title("(d) Fission after 5000 steps")
    ax.axis("off")
    ax.text(-0.12, 1.05, "(d)", transform=ax.transAxes, fontsize=14, fontweight="bold")

    fig.suptitle("Multi-Channel Competitive Flow-Lenia: Self-Repair and Fission",
                 fontsize=14, fontweight="medium", y=1.02)

    plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved narrative to {output_path}")


def generate_results(output_path, seed_results=None):
    """Generate results figure with metric breakdown.

    If seed_results is None, uses placeholder data.
    """
    if seed_results is None:
        # Placeholder -- will be updated with actual eval results
        seed_results = {
            1: {"tier3": 0.0, "tier4": 0.0, "vision": 0.0, "locomotion": 0.0, "composite": 0.0},
            2: {"tier3": 0.0, "tier4": 0.0, "vision": 0.0, "locomotion": 0.0, "composite": 0.0},
            3: {"tier3": 0.0, "tier4": 0.0, "vision": 0.0, "locomotion": 0.0, "composite": 0.0},
        }

    seeds = sorted(seed_results.keys())
    metrics = ["tier3", "tier4", "vision", "locomotion", "composite"]
    labels = ["Tier 3\n(robustness)", "Tier 4\n(reproduction)", "Vision\n(VLM)", "Locomotion", "Composite"]
    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True,
                             gridspec_kw={"width_ratios": [2, 1]})

    # Left: grouped bar chart
    ax = axes[0]
    x = np.arange(len(metrics))
    width = 0.25
    for i, seed in enumerate(seeds):
        vals = [seed_results[seed][m] for m in metrics]
        bars = ax.bar(x + i * width, vals, width, label=f"Seed {seed}",
                      color=colors[i] if i < len(colors) else "gray", alpha=0.8)
        for bar, val in zip(bars, vals):
            if val > 0.01:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                        f"{val:.3f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x + width)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.0)
    ax.set_title("Metric Breakdown by Seed")
    ax.legend(loc="upper right")

    # Right: summary table
    ax2 = axes[1]
    ax2.axis("off")

    # Compute means
    means = {}
    stds = {}
    for m in metrics:
        vals = [seed_results[s][m] for s in seeds]
        means[m] = np.mean(vals)
        stds[m] = np.std(vals)

    table_data = []
    for m, label in zip(metrics, labels):
        table_data.append([label.replace("\n", " "),
                           f"{means[m]:.4f} +/- {stds[m]:.4f}"])

    table = ax2.table(cellText=table_data,
                      colLabels=["Metric", "Mean +/- Std"],
                      loc="center",
                      cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 1.5)

    # Color the composite row
    for j in range(2):
        table[len(table_data), j].set_facecolor("#E8E8F0")

    ax2.set_title("Summary Statistics")

    fig.suptitle("Evaluation Results: Multi-Channel Competitive Flow-Lenia",
                 fontsize=14, fontweight="medium", y=1.02)

    plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved results to {output_path}")


if __name__ == "__main__":
    fig_dir = os.path.join(os.path.dirname(__file__), "figures")
    os.makedirs(fig_dir, exist_ok=True)

    generate_narrative(os.path.join(fig_dir, "narrative.png"))
    generate_results(os.path.join(fig_dir, "results.png"))
