"""Length-generalization evaluation.

Given a model trained on operands up to some digit count, measure exact-match
accuracy on operands with a specific digit count (typically larger than what
the model was trained on).
"""
from __future__ import annotations

from .data import Op, sample_pairs_at_digits
from .model import Transformer
from .train import evaluate


def eval_at_digits(
    model: Transformer,
    op: Op,
    digits: int,
    *,
    n_samples: int = 500,
    reverse_answer: bool = True,
    seed: int = 1234,
    batch_size: int = 256,
) -> float:
    """Exact-match accuracy on random (a, b) where both operands have exactly ``digits`` digits."""
    pairs = sample_pairs_at_digits(digits, n_samples, seed=seed)
    return evaluate(
        model,
        pairs,
        op,
        reverse_answer=reverse_answer,
        batch_size=batch_size,
        max_len=int(model.cfg.max_len),
    )
