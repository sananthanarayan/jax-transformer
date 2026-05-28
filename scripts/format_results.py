"""Print the Phase 1+2 sweep results as markdown tables for pasting into the whitepaper.

Usage:
    python scripts/format_results.py [--input results/sweep.json]

Output is sent to stdout; pipe to clipboard or redirect as needed.
"""
from __future__ import annotations

import argparse
import json
import pathlib


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser()
    repo_root = pathlib.Path(__file__).parent.parent
    p.add_argument("--input", type=pathlib.Path,
                   default=repo_root / "results" / "sweep.json")
    args = p.parse_args(argv)

    data = json.loads(args.input.read_text())
    cfg = data["config"]
    eval_digits = cfg["eval_digits"]
    variants = list(data["variants"].items())

    print("### Exact-match accuracy (%)\n")
    header = "| Variant | " + " | ".join(f"{d}d" for d in eval_digits) + " |"
    sep = "|---|" + "|".join("---:" for _ in eval_digits) + "|"
    print(header)
    print(sep)
    for name, v in variants:
        row_cells = [f"`{name}`"]
        for d in eval_digits:
            acc = v["accuracies"].get(str(d))
            if acc is None:
                row_cells.append("-")
            else:
                row_cells.append(f"{acc*100:.1f}")
        print("| " + " | ".join(row_cells) + " |")

    print("\n### Embedding-drift summary\n")
    for name, v in variants:
        drift = v.get("embedding_drift") or {}
        if not drift:
            continue
        for table_name, vals in drift.items():
            zeros = [i for i, x in enumerate(vals) if x == 0.0]
            nonzero_max_idx = max(
                [i for i, x in enumerate(vals) if x > 0.0], default=-1
            )
            print(f"- `{name}` / `{table_name}`: drift > 0 at positions 0..{nonzero_max_idx}; "
                  f"drift = 0 at positions {zeros[0] if zeros else 'none'}..{vals.__len__()-1} "
                  f"({len(zeros)} of {len(vals)} positions exactly 0).")


if __name__ == "__main__":
    main()
