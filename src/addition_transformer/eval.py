"""Length-generalization evaluation.

Given a model trained on operands up to some digit count, measure exact-match
accuracy on operands with a specific digit count (typically larger than what
the model was trained on), and per-digit-position correctness for the heatmap.
"""
from __future__ import annotations

import numpy as np

from .data import Op, render, sample_pairs_at_digits
from .model import Transformer
from .train import _build_eval_digit_positions, evaluate, greedy_generate
from .vocab import PAD_ID, VOCAB, encode


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


def per_digit_position_accuracy(
    model: Transformer,
    op: Op,
    digits: int,
    *,
    n_samples: int = 500,
    reverse_answer: bool = True,
    seed: int = 1234,
    batch_size: int = 256,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-digit-position accuracy for ``digits``-digit operands.

    Returns ``(accuracy, support)`` arrays of shape ``(max_pos,)`` where index
    ``k`` is the (k+1)-th digit position counted from the ones place if
    ``reverse_answer`` is True (i.e. position 0 = ones, 1 = tens, ...), and
    counted from the MSB otherwise. ``support[k]`` is the number of examples
    that had at least ``k+1`` answer digits; cells with ``support==0`` are NaN
    in ``accuracy``.
    """
    pairs = sample_pairs_at_digits(digits, n_samples, seed=seed)
    max_len = int(model.cfg.max_len)
    use_abacus = model.cfg.pos_encoding == "abacus"

    # Pre-render to know the maximum answer length.
    rendered = [render(int(a), int(b), op, reverse_answer=reverse_answer) for a, b in pairs]
    max_ans = max(len(ans) for _, ans in rendered)
    correct = np.zeros(max_ans, dtype=np.int64)
    support = np.zeros(max_ans, dtype=np.int64)

    for start in range(0, len(pairs), batch_size):
        chunk = pairs[start : start + batch_size]
        chunk_rendered = rendered[start : start + batch_size]
        B = len(chunk)
        prompts, plens, expected = [], [], []
        for (p, ans) in chunk_rendered:
            ids = encode(p) + [PAD_ID] * (max_len - len(p))
            prompts.append(ids)
            plens.append(len(p))
            expected.append(ans)
        prompt_ids = np.asarray(prompts, dtype=np.int32)
        plens_arr = np.asarray(plens, dtype=np.int32)
        dp = (
            _build_eval_digit_positions(prompt_ids, plens_arr, max_len, reverse_answer)
            if use_abacus
            else None
        )
        out = greedy_generate(model, prompt_ids, plens_arr, max_len, dp)
        for i in range(B):
            plen = plens[i]
            gen = out[i, plen:]
            pad_positions = np.where(gen == PAD_ID)[0]
            cut = pad_positions[0] if len(pad_positions) else len(gen)
            gen_chars = "".join(VOCAB[int(t)] for t in gen[:cut])
            exp = expected[i]
            for k in range(len(exp)):
                support[k] += 1
                if k < len(gen_chars) and gen_chars[k] == exp[k]:
                    correct[k] += 1

    with np.errstate(invalid="ignore", divide="ignore"):
        accuracy = np.where(support > 0, correct / np.maximum(support, 1), np.nan)
    return accuracy, support
