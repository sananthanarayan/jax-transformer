"""Plot the length-generalization sweep results.

Reads ``results/sweep.json`` (produced by ``scripts/run_sweep.py``) and writes
``results/length_gen.png`` — the headline chart showing exact-match accuracy
across operand digit counts for each variant, with a shaded region marking the
training distribution.
"""
from __future__ import annotations

import argparse
import json
import pathlib

import matplotlib.pyplot as plt


COLORS = {
    "baseline": "#d62728",   # red
    "reversed": "#ff7f0e",   # orange
    "nope":     "#2ca02c",   # green
}


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser()
    repo_root = pathlib.Path(__file__).parent.parent
    p.add_argument("--input", type=pathlib.Path, default=repo_root / "results" / "sweep.json")
    p.add_argument("--output", type=pathlib.Path, default=repo_root / "results" / "length_gen.png")
    args = p.parse_args(argv)

    data = json.loads(args.input.read_text())
    cfg = data["config"]
    train_digits = cfg["train_digits"]
    eval_digits = cfg["eval_digits"]

    fig, ax = plt.subplots(figsize=(7.5, 4.5))

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
            color=COLORS.get(name, "tab:blue"),
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
    ax.legend(loc="upper right", framealpha=0.95, fontsize=9)

    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=150)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
