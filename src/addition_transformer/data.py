"""Synthetic arithmetic dataset for the transformer.

Each example is a string like ``"123 + 456 = 579"`` (optionally with the answer
reversed to ``"975"``), encoded with :mod:`vocab` and padded to ``max_len``.

We return three parallel int32 arrays per batch:

- ``input_ids``  : ``(B, T)``  the full sequence
- ``targets``    : ``(B, T)``  ``input_ids`` shifted left by one (next-token target)
- ``loss_mask``  : ``(B, T)``  1 where the target is an answer token, else 0

The loss mask is the key bit — we don't want the model rewarded for re-emitting
the prompt or padding.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Literal

import numpy as np

from .vocab import PAD_ID, encode

Op = Literal["addition", "multiplication"]

# Longest example with spaces, 3-digit operands, no reverse:
#   addition:        "999 + 999 = 1998"   (16 chars, 4-digit answer)
#   multiplication:  "999 * 999 = 998001" (18 chars, 6-digit answer)
# We use a fixed length of 20 to leave room for a trailing PAD/EOS slot.
MAX_LEN = 20


def max_len_for(op: Op, max_digits: int) -> int:
    """Sequence length needed for the longest example at ``max_digits``, plus a PAD/EOS slot.

    Addition results have at most ``max_digits + 1`` digits; multiplication has at
    most ``2 * max_digits``. The fixed scaffolding ``" + "`` / ``" * "`` / ``" = "``
    contributes 6 characters.
    """
    result_chars = max_digits + 1 if op == "addition" else 2 * max_digits
    return 2 * max_digits + result_chars + 6 + 1


@dataclass(frozen=True)
class Example:
    input_ids: np.ndarray        # (T,) int32
    targets: np.ndarray          # (T,) int32
    loss_mask: np.ndarray        # (T,) int32
    digit_positions: np.ndarray  # (T,) int32, used only by the Abacus variant


# Vocab IDs 1..10 are '0'..'9' (see vocab.py).
_DIGIT_LO_ID = 1
_DIGIT_HI_ID = 10


def compute_digit_positions(
    input_ids: np.ndarray, *, reverse_answer: bool, op_symbol_ids: tuple[int, ...]
) -> np.ndarray:
    """Per-token "place value" index for Abacus-style embeddings.

    Each contiguous digit run in ``input_ids`` is one number. For operands
    (the first two digit runs) the run is written most-significant-first, so the
    LSB is at the end — positions are ``[run_len, run_len-1, ..., 1]``. For the
    answer (the third digit run), if ``reverse_answer`` is True the run is
    LSB-first so positions are ``[1, 2, ..., run_len]``; otherwise it is
    MSB-first like the operands.

    Non-digit tokens (including PAD) get position 0, which the embedding table
    treats as "no positional information."
    """
    is_digit = (input_ids >= _DIGIT_LO_ID) & (input_ids <= _DIGIT_HI_ID)
    positions = np.zeros_like(input_ids, dtype=np.int32)

    runs: list[tuple[int, int]] = []
    in_run = False
    start = 0
    for i, d in enumerate(is_digit):
        if d and not in_run:
            start = i
            in_run = True
        elif not d and in_run:
            runs.append((start, i))
            in_run = False
    if in_run:
        runs.append((start, len(is_digit)))

    _ = op_symbol_ids  # currently unused; reserved for future multi-op formats
    for run_idx, (rstart, rend) in enumerate(runs):
        run_len = rend - rstart
        is_answer_run = run_idx == 2
        for j in range(run_len):
            if is_answer_run and reverse_answer:
                positions[rstart + j] = j + 1
            else:
                positions[rstart + j] = run_len - j

    return positions


def compute_digit_positions_for_decode(
    prompt_ids: np.ndarray,
    prompt_len: int,
    *,
    model_max_len: int,
    reverse_answer: bool,
    max_answer_digits: int = 16,
) -> np.ndarray:
    """Digit-position array for greedy decoding (prompt + assumed-digit answer slots).

    For Abacus we need a position for every token the model will see during
    decoding. The prompt is scanned normally; the answer region is filled
    assuming each slot will be a digit. With ``reverse_answer=True`` the answer
    is LSB-first so positions are ``[1, 2, 3, ...]``. Beyond ``max_answer_digits``
    we fall back to 0 (the embedding will at worst be wrong for positions we
    do not actually evaluate at).
    """
    if not reverse_answer:
        raise NotImplementedError(
            "Abacus + natural-order answers needs a known answer length per "
            "example; only reverse_answer=True is supported."
        )
    positions = np.zeros(model_max_len, dtype=np.int32)
    prompt_positions = compute_digit_positions(
        prompt_ids[:prompt_len], reverse_answer=False, op_symbol_ids=()
    )
    positions[:prompt_len] = prompt_positions
    n = min(max_answer_digits, model_max_len - prompt_len)
    for k in range(n):
        positions[prompt_len + k] = k + 1
    return positions


def _op_symbol(op: Op) -> str:
    return {"addition": "+", "multiplication": "*"}[op]


def _apply(op: Op, a: int, b: int) -> int:
    return a + b if op == "addition" else a * b


def render(a: int, b: int, op: Op, reverse_answer: bool = True) -> tuple[str, str]:
    """Return (prompt, answer) strings. ``prompt`` ends with ``"= "`` is *not*
    appended — we use ``"... = "`` with a space after ``=`` so the model has a
    clean position to start emitting digits."""
    sym = _op_symbol(op)
    result = _apply(op, a, b)
    prompt = f"{a} {sym} {b} = "
    answer = str(result)
    if reverse_answer:
        answer = answer[::-1]
    return prompt, answer


def encode_example(
    a: int, b: int, op: Op, *, max_len: int = MAX_LEN, reverse_answer: bool = True
) -> Example:
    prompt, answer = render(a, b, op, reverse_answer=reverse_answer)
    full = prompt + answer
    if len(full) > max_len:
        raise ValueError(
            f"Example {full!r} (len={len(full)}) exceeds max_len={max_len}"
        )

    ids = encode(full)
    pad_len = max_len - len(ids)
    ids = ids + [PAD_ID] * pad_len
    input_ids = np.asarray(ids, dtype=np.int32)

    # Next-token targets: shift left by 1, last position predicts PAD.
    targets = np.concatenate([input_ids[1:], np.array([PAD_ID], dtype=np.int32)])

    # Loss mask: 1 on positions that predict an answer token, *plus one extra*
    # position that predicts the first PAD — this teaches the model to stop.
    loss_mask = np.zeros(max_len, dtype=np.int32)
    prompt_len = len(prompt)
    ans_len = len(answer)
    start = prompt_len - 1                       # predicts first answer char
    end = start + ans_len                        # predicts first PAD (EOS)
    loss_mask[start : min(end + 1, max_len)] = 1

    digit_positions = compute_digit_positions(
        input_ids, reverse_answer=reverse_answer, op_symbol_ids=()
    )

    return Example(
        input_ids=input_ids,
        targets=targets,
        loss_mask=loss_mask,
        digit_positions=digit_positions,
    )


def generate_pairs(
    op: Op, max_digits: int = 3, seed: int = 0
) -> np.ndarray:
    """All (a, b) pairs with operands in [0, 10**max_digits), shuffled deterministically."""
    n = 10**max_digits
    rng = np.random.default_rng(seed)
    pairs = np.stack(
        np.meshgrid(np.arange(n), np.arange(n), indexing="ij"), axis=-1
    ).reshape(-1, 2)
    rng.shuffle(pairs)
    # Sanity: multiplication of 999*999 = 998001 still fits in MAX_LEN.
    _ = op
    return pairs


def sample_pairs_at_digits(
    digits: int, n: int, *, seed: int = 0
) -> np.ndarray:
    """Sample ``n`` random (a, b) pairs where both operands have exactly ``digits`` digits.

    ``digits=1`` means operands in [0, 10); for larger ``digits``, operands are in
    [10**(digits-1), 10**digits) so they are strictly ``digits``-digit numbers.
    Used for length-generalization evaluation, where the 6-digit grid (10**12 pairs)
    is too large to enumerate.
    """
    if digits < 1:
        raise ValueError(f"digits must be >= 1, got {digits}")
    lo = 0 if digits == 1 else 10 ** (digits - 1)
    hi = 10**digits
    rng = np.random.default_rng(seed)
    a = rng.integers(lo, hi, size=n)
    b = rng.integers(lo, hi, size=n)
    return np.stack([a, b], axis=1)


def build_arrays(
    pairs: np.ndarray, op: Op, *, max_len: int = MAX_LEN, reverse_answer: bool = True
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized-ish encoding: build (input_ids, targets, loss_mask, digit_positions)."""
    n = len(pairs)
    input_ids = np.zeros((n, max_len), dtype=np.int32)
    targets = np.zeros((n, max_len), dtype=np.int32)
    loss_mask = np.zeros((n, max_len), dtype=np.int32)
    digit_positions = np.zeros((n, max_len), dtype=np.int32)
    for i, (a, b) in enumerate(pairs):
        ex = encode_example(
            int(a), int(b), op, max_len=max_len, reverse_answer=reverse_answer
        )
        input_ids[i] = ex.input_ids
        targets[i] = ex.targets
        loss_mask[i] = ex.loss_mask
        digit_positions[i] = ex.digit_positions
    return input_ids, targets, loss_mask, digit_positions


def generate_mixed_pairs(
    op: Op,
    min_digits: int,
    max_digits: int,
    n_samples: int,
    *,
    seed: int = 0,
) -> np.ndarray:
    """Sample ``n_samples`` pairs with operand digit counts mixed uniformly over
    ``[min_digits, max_digits]``. Used by the Abacus length-curriculum variant
    so that every Abacus position embedding receives gradient during training.
    """
    rng = np.random.default_rng(seed)
    buckets = list(range(min_digits, max_digits + 1))
    per_bucket = max(1, n_samples // len(buckets))
    parts = []
    for d in buckets:
        parts.append(sample_pairs_at_digits(d, per_bucket, seed=seed * 1000 + d))
    pairs = np.concatenate(parts, axis=0)
    rng.shuffle(pairs)
    _ = op
    return pairs


def split(
    pairs: np.ndarray, val_frac: float = 0.05
) -> tuple[np.ndarray, np.ndarray]:
    n_val = int(len(pairs) * val_frac)
    return pairs[n_val:], pairs[:n_val]


def iterate_batches(
    input_ids: np.ndarray,
    targets: np.ndarray,
    loss_mask: np.ndarray,
    digit_positions: np.ndarray,
    batch_size: int,
    *,
    shuffle: bool = True,
    seed: int = 0,
    drop_last: bool = True,
) -> Iterator[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    n = len(input_ids)
    idx = np.arange(n)
    if shuffle:
        np.random.default_rng(seed).shuffle(idx)
    end = (n // batch_size) * batch_size if drop_last else n
    for start in range(0, end, batch_size):
        sel = idx[start : start + batch_size]
        yield input_ids[sel], targets[sel], loss_mask[sel], digit_positions[sel]
