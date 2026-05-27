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
    input_ids: np.ndarray   # (T,) int32
    targets: np.ndarray     # (T,) int32
    loss_mask: np.ndarray   # (T,) int32


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

    return Example(input_ids=input_ids, targets=targets, loss_mask=loss_mask)


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
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized-ish encoding: build (input_ids, targets, loss_mask) for all pairs."""
    n = len(pairs)
    input_ids = np.zeros((n, max_len), dtype=np.int32)
    targets = np.zeros((n, max_len), dtype=np.int32)
    loss_mask = np.zeros((n, max_len), dtype=np.int32)
    for i, (a, b) in enumerate(pairs):
        ex = encode_example(
            int(a), int(b), op, max_len=max_len, reverse_answer=reverse_answer
        )
        input_ids[i] = ex.input_ids
        targets[i] = ex.targets
        loss_mask[i] = ex.loss_mask
    return input_ids, targets, loss_mask


def split(
    pairs: np.ndarray, val_frac: float = 0.05
) -> tuple[np.ndarray, np.ndarray]:
    n_val = int(len(pairs) * val_frac)
    return pairs[n_val:], pairs[:n_val]


def iterate_batches(
    input_ids: np.ndarray,
    targets: np.ndarray,
    loss_mask: np.ndarray,
    batch_size: int,
    *,
    shuffle: bool = True,
    seed: int = 0,
    drop_last: bool = True,
) -> Iterator[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    n = len(input_ids)
    idx = np.arange(n)
    if shuffle:
        np.random.default_rng(seed).shuffle(idx)
    end = (n // batch_size) * batch_size if drop_last else n
    for start in range(0, end, batch_size):
        sel = idx[start : start + batch_size]
        yield input_ids[sel], targets[sel], loss_mask[sel]
