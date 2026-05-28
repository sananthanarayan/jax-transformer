"""Tests for the Abacus digit-position assignment."""
from __future__ import annotations

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "src"))

from addition_transformer.data import (
    compute_digit_positions,
    compute_digit_positions_for_decode,
    encode_example,
)
from addition_transformer.vocab import PAD_ID, encode


def _make_ids(text: str, pad_to: int) -> np.ndarray:
    return np.asarray(encode(text) + [PAD_ID] * (pad_to - len(text)), dtype=np.int32)


def test_natural_order_addition():
    # "123 + 456 = 579" — each digit run goes MSB->LSB so positions are run_len, ..., 1.
    ids = _make_ids("123 + 456 = 579", pad_to=20)
    pos = compute_digit_positions(ids, reverse_answer=False, op_symbol_ids=())
    expected = [3, 2, 1, 0, 0, 0, 3, 2, 1, 0, 0, 0, 3, 2, 1, 0, 0, 0, 0, 0]
    assert pos.tolist() == expected, pos.tolist()


def test_reversed_answer():
    # "123 + 45 = 861" — answer "861" is the reversed form of 168.
    # Operands stay MSB-first; answer is LSB-first so positions count up from 1.
    ids = _make_ids("123 + 45 = 861", pad_to=20)
    pos = compute_digit_positions(ids, reverse_answer=True, op_symbol_ids=())
    expected = [3, 2, 1, 0, 0, 0, 2, 1, 0, 0, 0, 1, 2, 3, 0, 0, 0, 0, 0, 0]
    assert pos.tolist() == expected, pos.tolist()


def test_single_digit_operands():
    # "7 + 4 = 11" with reversed answer "11" → both digits get LSB-first positions.
    ids = _make_ids("7 + 4 = 11", pad_to=15)
    pos = compute_digit_positions(ids, reverse_answer=True, op_symbol_ids=())
    expected = [1, 0, 0, 0, 1, 0, 0, 0, 1, 2, 0, 0, 0, 0, 0]
    assert pos.tolist() == expected, pos.tolist()


def test_six_digit_with_carry():
    # 999999 + 999999 = 1999998 (reversed: "8999991") — 7-digit answer with positions 1..7.
    ids = _make_ids("999999 + 999999 = 8999991", pad_to=28)
    pos = compute_digit_positions(ids, reverse_answer=True, op_symbol_ids=())
    expected = (
        [6, 5, 4, 3, 2, 1, 0, 0, 0]
        + [6, 5, 4, 3, 2, 1, 0, 0, 0]
        + [1, 2, 3, 4, 5, 6, 7, 0, 0, 0]
    )
    assert pos.tolist() == expected, pos.tolist()


def test_encode_example_populates_digit_positions():
    ex = encode_example(123, 456, op="addition", max_len=20, reverse_answer=True)
    # Indices of the digit tokens in "123 + 456 = 975": 0,1,2 (operand1), 6,7,8 (operand2),
    # 12,13,14 (reversed answer).
    pos = ex.digit_positions.tolist()
    assert pos[0:3] == [3, 2, 1]
    assert pos[6:9] == [3, 2, 1]
    assert pos[12:15] == [1, 2, 3]
    # Everything else is 0.
    for k, v in enumerate(pos):
        if k not in {0, 1, 2, 6, 7, 8, 12, 13, 14}:
            assert v == 0, (k, v, pos)


def test_decode_helper_fills_answer_slots():
    prompt = "123 + 456 = "
    prompt_ids = _make_ids(prompt, pad_to=26)
    full = compute_digit_positions_for_decode(
        prompt_ids,
        prompt_len=len(prompt),
        model_max_len=26,
        reverse_answer=True,
    )
    # Prompt positions: operands have their digit slots; spaces / '+' / '=' get 0.
    assert full[0:3].tolist() == [3, 2, 1]   # operand 1
    assert full[6:9].tolist() == [3, 2, 1]   # operand 2
    # Answer region starts at index 12, fills with [1, 2, ...] up to max_answer_digits.
    assert full[12] == 1
    assert full[13] == 2
    assert full[14] == 3


if __name__ == "__main__":
    test_natural_order_addition()
    test_reversed_answer()
    test_single_digit_operands()
    test_six_digit_with_carry()
    test_encode_example_populates_digit_positions()
    test_decode_helper_fills_answer_slots()
    print("OK - all digit-position tests passed")
