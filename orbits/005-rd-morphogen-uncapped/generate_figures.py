#!/usr/bin/env python3
"""
Generate narrative and results figures for orbit 005-rd-morphogen-uncapped.

Produces:
  - figures/narrative.png: baseline (capped orbit 003) vs uncapped V2, showing
    fission happening in the uncapped version but not the capped one.
  - figures/results.png: metric breakdown across seeds.
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "solution"))

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

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from substrate import RDMorphogenSubstrateV2


def run_simulation(substrate, params, init_state, n_steps, capture_at):
    """Run simulation and capture frames at specified steps."""
    substrate.reset()
    frames = {}
    masses = []

    if HAS_TORCH:
        device = torch.device("cpu")
        t = torch.tensor(init_state, dtype=torch.float32, device=device)
        substrate.to(device)
        with torch.no_grad():
            for step in range(n_steps):
                t = substrate.update_step(t, params)
                t = t.clamp(0.0)
                masses.append(float(t.sum().item()))
                if step in capture_at:
                    frames[step] = t.detach().cpu().numpy().copy()
    return frames, masses


def state_to_field(state):
    """Convert state to 2D field for display."""
    if state.ndim == 3:
        field = state.sum(axis=0)
    else:
        field = state
    mx = field.max()
    if mx > 1e-8:
        field = field / mx
    return field


def generate_narrative(out_path, grid_size=128):
    """Generate narrative figure showing fission in uncapped vs baseline."""
    substrate = RDMorphogenSubstrateV2(grid_size=grid_size)
    params = substrate.get_default_params()

    # Create initial blob
    rng = np.random.RandomState(42)
    H, W = grid_size, grid_size
    cy, cx = H // 2, W // 2
    y = np.arange(H) - cy
    x = np.arange(W) - cx
    yy, xx = np.meshgrid(y, x, indexing="ij")
    blob = np.exp(-0.5 * ((yy / 18.0) ** 2 + (xx / 12.0) ** 2))
    noise = rng.randn(H, W) * 0.02
    init_state = np.clip(blob + noise, 0.0, 1.0).astype(np.float32)[np.newaxis]

    # Run uncapped version
    capture_steps = [0, 500, 1000, 2000, 3000, 4000]
    frames_uncapped, masses_uncapped = run_simulation(
        substrate, params, init_state, 4001, set(capture_steps)
    )

    # Run "capped" version (simulate by adding strict mass conservation)
    params_capped = dict(params)
    params_capped["soft_drift_threshold"] = 0.01  # Very strict
    params_capped["mass_ceiling"] = 1.05  # Tight cap
    substrate2 = RDMorphogenSubstrateV2(grid_size=grid_size)
    frames_capped, masses_capped = run_simulation(
        substrate2, params_capped, init_state, 4001, set(capture_steps)
    )

    # Create figure: 2 rows x 4 cols
    # Row 1: Capped (baseline-like) at t=0, 1000, 2000, 4000
    # Row 2: Uncapped (our method) at same timesteps
    display_steps = [0, 1000, 2000, 4000]
    n_cols = len(display_steps)

    fig, axes = plt.subplots(2, n_cols, figsize=(4 * n_cols, 8),
                             constrained_layout=True)

    for col, step in enumerate(display_steps):
        # Capped row
        if step in frames_capped:
            field = state_to_field(frames_capped[step])
        else:
            field = np.zeros((grid_size, grid_size))
        axes[0, col].imshow(field, cmap="inferno", vmin=0, vmax=1)
        axes[0, col].set_title(f"T={step}", fontsize=11)
        axes[0, col].axis("off")

        # Uncapped row
        if step in frames_uncapped:
            field = state_to_field(frames_uncapped[step])
        else:
            field = np.zeros((grid_size, grid_size))
        axes[1, col].imshow(field, cmap="inferno", vmin=0, vmax=1)
        axes[1, col].set_title(f"T={step}", fontsize=11)
        axes[1, col].axis("off")

    axes[0, 0].text(-0.15, 0.5, "Capped\n(baseline)",
                    transform=axes[0, 0].transAxes,
                    fontsize=12, fontweight="medium",
                    va="center", ha="right", color="#888888")
    axes[1, 0].text(-0.15, 0.5, "Uncapped\n(ours)",
                    transform=axes[1, 0].transAxes,
                    fontsize=12, fontweight="medium",
                    va="center", ha="right", color="#4C72B0")

    fig.suptitle("RD Morphogen Flow-Lenia: Capped vs Uncapped Growth",
                 fontsize=14, fontweight="medium", y=1.02)

    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved narrative figure: {out_path}")


def generate_results(out_path, metrics=None):
    """Generate results figure with metric breakdown."""
    if metrics is None:
        # Placeholder -- will be updated with real data
        metrics = {
            "seeds": [1, 2, 3],
            "composite": [0.0, 0.0, 0.0],
            "tier3": [0.0, 0.0, 0.0],
            "tier4": [0.0, 0.0, 0.0],
            "vision": [0.0, 0.0, 0.0],
            "locomotion": [0.0, 0.0, 0.0],
        }

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)

    # Panel (a): Per-seed metric breakdown
    seeds = metrics["seeds"]
    x = np.arange(len(seeds))
    width = 0.15

    colors = {
        "tier3": "#4C72B0",
        "tier4": "#DD8452",
        "vision": "#55A868",
        "locomotion": "#8172B3",
        "composite": "#C44E52",
    }

    for i, (key, color) in enumerate(colors.items()):
        if key == "composite":
            continue
        vals = metrics[key]
        axes[0].bar(x + i * width, vals, width, label=key, color=color, alpha=0.85)

    axes[0].set_xlabel("Seed")
    axes[0].set_ylabel("Score")
    axes[0].set_title("Per-seed Component Scores")
    axes[0].set_xticks(x + 1.5 * width)
    axes[0].set_xticklabels([f"Seed {s}" for s in seeds])
    axes[0].legend(loc="upper left")
    axes[0].set_ylim(0, 1.0)
    axes[0].text(-0.12, 1.05, "(a)", transform=axes[0].transAxes,
                 fontsize=14, fontweight="bold")

    # Panel (b): Composite score with error bar
    comp = np.array(metrics["composite"])
    mean_comp = comp.mean()
    std_comp = comp.std()

    axes[1].bar(["Mean"], [mean_comp], color="#C44E52", alpha=0.85,
                yerr=[std_comp], capsize=8)
    for i, (s, c) in enumerate(zip(seeds, comp)):
        axes[1].scatter(0, c, color="#333333", s=40, zorder=5, alpha=0.7)

    axes[1].set_ylabel("Composite Score")
    axes[1].set_title("Composite Life-Likeness")
    axes[1].set_ylim(0, max(0.8, mean_comp + 2 * std_comp + 0.1))
    axes[1].axhline(0.55, color="#888888", linestyle="--", linewidth=1, alpha=0.7)
    axes[1].text(0.5, 0.56, "target = 0.55", transform=axes[1].get_xaxis_transform(),
                 fontsize=9, color="#888888", ha="center")
    axes[1].text(-0.12, 1.05, "(b)", transform=axes[1].transAxes,
                 fontsize=14, fontweight="bold")

    fig.suptitle(f"Orbit 005: RD Morphogen V2 (Uncapped) -- Metric={mean_comp:.4f} +/- {std_comp:.4f}",
                 fontsize=13, fontweight="medium", y=1.02)

    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved results figure: {out_path}")


if __name__ == "__main__":
    fig_dir = os.path.join(os.path.dirname(__file__), "figures")
    os.makedirs(fig_dir, exist_ok=True)

    print("Generating narrative figure...")
    generate_narrative(os.path.join(fig_dir, "narrative.png"), grid_size=128)

    print("Generating results figure (placeholder)...")
    generate_results(os.path.join(fig_dir, "results.png"))
