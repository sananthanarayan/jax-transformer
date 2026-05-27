"""Phase 1 length-generalization sweep.

Trains three variants on 3-digit addition only:

    baseline    : learned positional embeddings, natural answer order
    reversed    : learned positional embeddings, reversed answer order
    nope        : no positional encoding,        reversed answer order

Each variant is then evaluated on operand digit counts 1..6 using sampled pairs.
Results are written to ``results/sweep.json`` for the plotting script to consume.

Run from the repo root::

    python scripts/run_sweep.py
    python scripts/run_sweep.py --epochs 4 --eval-samples 200   # quick sanity sweep
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

# Make src/ importable without `pip install -e .`
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "src"))

from addition_transformer.data import max_len_for
from addition_transformer.eval import eval_at_digits
from addition_transformer.train import train_model


VARIANTS = [
    {
        "name": "baseline",
        "label": "Learned PE + natural order",
        "pos_encoding": "learned",
        "reverse_answer": False,
    },
    {
        "name": "reversed",
        "label": "Learned PE + reversed answers",
        "pos_encoding": "learned",
        "reverse_answer": True,
    },
    {
        "name": "nope",
        "label": "NoPE + reversed answers",
        "pos_encoding": "none",
        "reverse_answer": True,
    },
]


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--op", default="addition", choices=["addition", "multiplication"])
    p.add_argument("--train-digits", type=int, default=3,
                   help="Train on operands with up to this many digits.")
    p.add_argument("--eval-max-digits", type=int, default=6,
                   help="Evaluate length generalization up to this digit count.")
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--eval-samples", type=int, default=500,
                   help="Random pairs sampled at each eval digit count.")
    p.add_argument("--output", type=pathlib.Path,
                   default=pathlib.Path(__file__).parent.parent / "results" / "sweep.json")
    args = p.parse_args(argv)

    # The model's positional capacity must fit the longest eval sequence,
    # not just the training distribution.
    model_max_len = max_len_for(args.op, args.eval_max_digits)
    eval_digits = list(range(1, args.eval_max_digits + 1))

    results: dict = {
        "config": {
            "op": args.op,
            "train_digits": args.train_digits,
            "eval_digits": eval_digits,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "seed": args.seed,
            "eval_samples": args.eval_samples,
            "model_max_len": model_max_len,
        },
        "variants": {},
    }

    t_all = time.time()
    for v in VARIANTS:
        prefix = f"[{v['name']}] "
        print(f"\n=== {v['name']}: {v['label']} ===")
        t0 = time.time()
        model = train_model(
            op=args.op,
            max_digits=args.train_digits,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            seed=args.seed,
            reverse_answer=v["reverse_answer"],
            pos_encoding=v["pos_encoding"],
            model_max_len=model_max_len,
            eval_samples=min(args.eval_samples * 2, 2000),
            log_prefix=prefix,
        )
        train_time = time.time() - t0

        accs: dict[str, float] = {}
        for d in eval_digits:
            acc = eval_at_digits(
                model,
                args.op,
                d,
                n_samples=args.eval_samples,
                reverse_answer=v["reverse_answer"],
                seed=args.seed + 1000,
            )
            accs[str(d)] = acc
            print(f"{prefix}digits={d}: acc={acc*100:.2f}%")

        results["variants"][v["name"]] = {
            "label": v["label"],
            "pos_encoding": v["pos_encoding"],
            "reverse_answer": v["reverse_answer"],
            "train_time_sec": train_time,
            "accuracies": accs,
        }

        # Write after each variant so partial runs are still useful.
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(results, indent=2))

    total = time.time() - t_all
    print(f"\n[done] total wall time = {total:.1f}s   results -> {args.output}")


if __name__ == "__main__":
    main()
