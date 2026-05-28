"""Length-generalization sweep (Phases 1 + 2).

Trains a family of variants on small-digit addition and evaluates each on
operand digit counts 1..N. By default we run six variants:

    Phase 1
      baseline           : learned PE,  natural answer order
      reversed           : learned PE,  reversed answer order
      nope               : no PE,       reversed answer order
    Phase 2
      rope               : rotary PE,                  reversed answer order
      abacus             : digit-position PE only,     reversed answer order
      abacus_curriculum  : digit-position PE + mixed-digit training (1..max)

Pass ``--phase 1`` to skip the Phase 2 variants.

Results are written to ``results/sweep.json`` (per-variant overall accuracy and
per-digit-position accuracy) and consumed by ``scripts/plot.py``.
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
from addition_transformer.eval import (
    embedding_drift,
    eval_at_digits,
    per_digit_position_accuracy,
)
from addition_transformer.train import train_model


PHASE1_VARIANTS = [
    {"name": "baseline", "label": "Learned PE + natural order",
     "pos_encoding": "learned", "reverse_answer": False, "mixed_digits": False},
    {"name": "reversed", "label": "Learned PE + reversed answers",
     "pos_encoding": "learned", "reverse_answer": True,  "mixed_digits": False},
    {"name": "nope",     "label": "NoPE + reversed answers",
     "pos_encoding": "none",    "reverse_answer": True,  "mixed_digits": False},
]
PHASE2_VARIANTS = [
    {"name": "rope",     "label": "RoPE + reversed answers",
     "pos_encoding": "rope",    "reverse_answer": True,  "mixed_digits": False},
    {"name": "abacus",   "label": "Abacus (clean) + reversed answers",
     "pos_encoding": "abacus",  "reverse_answer": True,  "mixed_digits": False},
    {"name": "abacus_curriculum",
     "label": "Abacus + length curriculum (1..max digits)",
     "pos_encoding": "abacus",  "reverse_answer": True,  "mixed_digits": True},
]


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--op", default="addition", choices=["addition", "multiplication"])
    p.add_argument("--phase", type=int, default=2, choices=[1, 2],
                   help="1 = baseline/reversed/nope only; 2 = all six variants.")
    p.add_argument("--train-digits", type=int, default=3)
    p.add_argument("--eval-max-digits", type=int, default=6)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--eval-samples", type=int, default=500)
    p.add_argument("--mixed-n-samples", type=int, default=1_000_000,
                   help="Training samples for the abacus_curriculum variant.")
    p.add_argument("--variants", default=None,
                   help="Comma-separated subset of variant names to run.")
    p.add_argument("--output", type=pathlib.Path,
                   default=pathlib.Path(__file__).parent.parent / "results" / "sweep.json")
    args = p.parse_args(argv)

    all_variants = PHASE1_VARIANTS + (PHASE2_VARIANTS if args.phase >= 2 else [])
    if args.variants:
        wanted = {v.strip() for v in args.variants.split(",")}
        all_variants = [v for v in all_variants if v["name"] in wanted]

    model_max_len = max_len_for(args.op, args.eval_max_digits)
    eval_digits = list(range(1, args.eval_max_digits + 1))

    results: dict = {
        "config": {
            "op": args.op,
            "phase": args.phase,
            "train_digits": args.train_digits,
            "eval_digits": eval_digits,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "seed": args.seed,
            "eval_samples": args.eval_samples,
            "model_max_len": model_max_len,
            "mixed_n_samples": args.mixed_n_samples,
        },
        "variants": {},
    }

    t_all = time.time()
    for v in all_variants:
        prefix = f"[{v['name']}] "
        print(f"\n=== {v['name']}: {v['label']} ===")
        t0 = time.time()
        # The curriculum variant trains on up to ``eval_max_digits - 1`` digits so it
        # still has to extrapolate by one. Other variants train on ``train_digits``.
        max_digits = (args.eval_max_digits - 1) if v["mixed_digits"] else args.train_digits
        model = train_model(
            op=args.op,
            max_digits=max_digits,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            seed=args.seed,
            reverse_answer=v["reverse_answer"],
            pos_encoding=v["pos_encoding"],
            model_max_len=model_max_len,
            eval_samples=min(args.eval_samples * 2, 2000),
            mixed_digits=v["mixed_digits"],
            mixed_min_digits=1,
            mixed_n_samples=args.mixed_n_samples,
            log_prefix=prefix,
        )
        train_time = time.time() - t0

        accs: dict[str, float] = {}
        per_pos: dict[str, list[float | None]] = {}
        for d in eval_digits:
            acc = eval_at_digits(
                model, args.op, d,
                n_samples=args.eval_samples,
                reverse_answer=v["reverse_answer"],
                seed=args.seed + 1000,
            )
            accs[str(d)] = acc
            pp_acc, _ = per_digit_position_accuracy(
                model, args.op, d,
                n_samples=args.eval_samples,
                reverse_answer=v["reverse_answer"],
                seed=args.seed + 1000,
            )
            per_pos[str(d)] = [None if (x != x) else float(x) for x in pp_acc]
            print(f"{prefix}digits={d}: overall={acc*100:.2f}%  "
                  f"per-position={['{:.0%}'.format(x) if x is not None else '-' for x in per_pos[str(d)]]}")

        drift = embedding_drift(model, seed=args.seed)
        drift_json = {k: v.tolist() for k, v in drift.items()}
        if drift_json:
            for k, vals in drift_json.items():
                zero_positions = [i for i, x in enumerate(vals) if x == 0.0]
                print(f"{prefix}{k}: {len(zero_positions)} positions never received "
                      f"gradient (indices {zero_positions})")

        results["variants"][v["name"]] = {
            "label": v["label"],
            "pos_encoding": v["pos_encoding"],
            "reverse_answer": v["reverse_answer"],
            "mixed_digits": v["mixed_digits"],
            "train_max_digits": max_digits,
            "train_time_sec": train_time,
            "accuracies": accs,
            "per_position_accuracy": per_pos,
            "embedding_drift": drift_json,
        }

        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(results, indent=2))

    total = time.time() - t_all
    print(f"\n[done] total wall time = {total:.1f}s   results -> {args.output}")


if __name__ == "__main__":
    main()
