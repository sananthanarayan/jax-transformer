"""Plot the length-generalization sweep results.

Reads ``results/sweep.json`` and writes:
  * ``results/length_gen.png``       — exact-match accuracy vs operand digit count
  * ``results/per_digit_heatmap.png`` — per-digit-position error rate, one panel
                                         per variant
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib

import matplotlib.pyplot as plt
import numpy as np


COLORS = {
    "baseline":          "#d62728",  # red
    "reversed":          "#ff7f0e",  # orange
    "nope":              "#2ca02c",  # green
    "rope":              "#1f77b4",  # blue
    "abacus":            "#9467bd",  # purple
    "abacus_curriculum": "#17becf",  # cyan
}


def plot_line_chart(data: dict, output: pathlib.Path) -> None:
    cfg = data["config"]
    train_digits = cfg["train_digits"]
    eval_digits = cfg["eval_digits"]

    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    ax.axvspan(
        eval_digits[0] - 0.1, train_digits + 0.1,
        alpha=0.08, color="gray", zorder=0,
        label=f"Training distribution (≤{train_digits} digits)",
    )

    for name, v in data["variants"].items():
        xs = [int(d) for d in v["accuracies"].keys()]
        ys = [v["accuracies"][str(d)] * 100 for d in xs]
        ax.plot(
            xs, ys,
            marker="o", linewidth=2, markersize=7,
            color=COLORS.get(name, "tab:gray"),
            label=v["label"],
        )

    ax.set_xlabel("Operand digit count", fontsize=11)
    ax.set_ylabel("Exact-match accuracy (%)", fontsize=11)
    ax.set_title(
        f"Length generalization on {cfg['op']} (trained on ≤{train_digits} digits)",
        fontsize=12,
    )
    ax.set_xticks(eval_digits)
    ax.set_ylim(-2, 102)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", framealpha=0.95, fontsize=8.5)

    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150)
    print(f"wrote {output}")


def plot_heatmaps(data: dict, output: pathlib.Path) -> None:
    cfg = data["config"]
    eval_digits = cfg["eval_digits"]
    variants = list(data["variants"].items())

    max_pos = 0
    for _, v in variants:
        for d in eval_digits:
            row = v["per_position_accuracy"].get(str(d)) or []
            max_pos = max(max_pos, len(row))

    n = len(variants)
    cols = 3 if n > 2 else n
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(
        rows, cols,
        figsize=(4.0 * cols, 2.6 * rows),
        squeeze=False,
    )

    for i, (name, v) in enumerate(variants):
        ax = axes[i // cols][i % cols]
        grid = np.full((len(eval_digits), max_pos), np.nan, dtype=np.float32)
        for r, d in enumerate(eval_digits):
            row = v["per_position_accuracy"].get(str(d)) or []
            for c, val in enumerate(row):
                if val is None:
                    grid[r, c] = np.nan
                else:
                    grid[r, c] = 1.0 - float(val)   # error rate

        im = ax.imshow(
            grid, aspect="auto", origin="lower",
            cmap="Reds", vmin=0.0, vmax=1.0,
            interpolation="nearest",
        )
        ax.set_title(v["label"], fontsize=9)
        ax.set_xlabel("Answer digit position (0 = ones)", fontsize=8)
        ax.set_ylabel("Operand digit count", fontsize=8)
        ax.set_yticks(range(len(eval_digits)))
        ax.set_yticklabels(eval_digits)
        ax.set_xticks(range(max_pos))
        ax.set_xticklabels(range(max_pos))
        ax.tick_params(labelsize=8)

    # Hide unused axes
    for j in range(n, rows * cols):
        axes[j // cols][j % cols].axis("off")

    fig.suptitle(
        f"Per-digit-position error rate ({cfg['op']}, trained on ≤{cfg['train_digits']} digits)",
        fontsize=11,
    )
    cbar = fig.colorbar(im, ax=axes, shrink=0.6, pad=0.02)
    cbar.set_label("Error rate", fontsize=9)
    fig.savefig(output, dpi=150, bbox_inches="tight")
    print(f"wrote {output}")


def plot_embedding_drift(data: dict, output: pathlib.Path) -> None:
    """Bar chart of per-position L2 drift from initialization.

    Two stacked panels: top = learned absolute position embeddings (where the
    Phase 1 baseline fails — positions ≥16 stay at init), bottom = Abacus
    digit-position embeddings (where the clean abacus variant fails — positions
    beyond the training-distribution digit count stay at init).
    """
    cfg = data["config"]
    pos_emb_variants: list[tuple[str, list[float]]] = []
    digit_emb_variants: list[tuple[str, list[float]]] = []
    for name, v in data["variants"].items():
        drift = v.get("embedding_drift") or {}
        if "pos_emb" in drift:
            pos_emb_variants.append((name, drift["pos_emb"]))
        if "digit_pos_emb" in drift:
            digit_emb_variants.append((name, drift["digit_pos_emb"]))

    rows = sum(1 for x in [pos_emb_variants, digit_emb_variants] if x)
    if rows == 0:
        return

    fig, axes = plt.subplots(rows, 1, figsize=(8.5, 3.2 * rows), squeeze=False)
    row_idx = 0

    def _bar_panel(ax, variants, max_x, vline_at, panel_title, xlabel):
        n = len(variants)
        width = 0.8 / n
        positions = np.arange(max_x)
        for i, (name, drift) in enumerate(variants):
            drift_padded = drift + [0.0] * (max_x - len(drift))
            ax.bar(
                positions + (i - (n - 1) / 2) * width,
                drift_padded,
                width=width,
                color=COLORS.get(name, "tab:gray"),
                label=data["variants"][name]["label"],
            )
        ax.axvline(vline_at + 0.5, color="black", linestyle="--", linewidth=1, alpha=0.6,
                   label=f"Training-distribution boundary")
        ax.set_xticks(positions)
        ax.set_xlabel(xlabel, fontsize=10)
        ax.set_ylabel("L2 drift from init", fontsize=10)
        ax.set_title(panel_title, fontsize=11)
        ax.grid(True, alpha=0.3, axis="y")
        ax.legend(loc="upper right", fontsize=8)

    if pos_emb_variants:
        max_x = max(len(d) for _, d in pos_emb_variants)
        # Longest training-input position is roughly max_len for train_digits.
        # For 3-digit addition: "999 + 999 = 1998" = 16 chars, so positions 0..15 see grad.
        train_boundary = 15
        _bar_panel(
            axes[row_idx][0], pos_emb_variants, max_x, train_boundary,
            "Learned absolute position embedding — drift per position",
            "Position index in input sequence",
        )
        row_idx += 1

    if digit_emb_variants:
        max_x = max(len(d) for _, d in digit_emb_variants)
        # Position 0 = non-digit; positions 1..train_digits = operand digits;
        # position train_digits+1 = carry (active for addition). So train_digits+1 is the
        # rightmost position that sees gradient at clean Abacus.
        train_digits = cfg.get("train_digits", 3)
        train_boundary = train_digits + 1
        _bar_panel(
            axes[row_idx][0], digit_emb_variants, max_x, train_boundary,
            "Abacus digit-position embedding — drift per place value",
            "Digit position (0 = non-digit; 1 = ones place; 2 = tens; ...)",
        )

    fig.suptitle("Which positional embeddings actually received gradient?", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(output, dpi=150)
    print(f"wrote {output}")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser()
    repo_root = pathlib.Path(__file__).parent.parent
    p.add_argument("--input", type=pathlib.Path, default=repo_root / "results" / "sweep.json")
    p.add_argument("--line-output", type=pathlib.Path,
                   default=repo_root / "results" / "length_gen.png")
    p.add_argument("--heatmap-output", type=pathlib.Path,
                   default=repo_root / "results" / "per_digit_heatmap.png")
    p.add_argument("--drift-output", type=pathlib.Path,
                   default=repo_root / "results" / "embedding_drift.png")
    args = p.parse_args(argv)

    data = json.loads(args.input.read_text())
    plot_line_chart(data, args.line_output)
    if any("per_position_accuracy" in v for v in data["variants"].values()):
        plot_heatmaps(data, args.heatmap_output)
    if any(v.get("embedding_drift") for v in data["variants"].values()):
        plot_embedding_drift(data, args.drift_output)


if __name__ == "__main__":
    main()
