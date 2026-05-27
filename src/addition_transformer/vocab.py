"""Hardcoded character vocabulary for the arithmetic transformer.

Tokens (15 total):
    0  : <pad>
    1-10 : '0'..'9'
    11   : ' '
    12   : '+'
    13   : '*'
    14   : '='
"""
from __future__ import annotations

PAD_ID = 0

VOCAB: tuple[str, ...] = (
    "<pad>",
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
    " ", "+", "*", "=",
)
VOCAB_SIZE = len(VOCAB)

_STOI = {tok: i for i, tok in enumerate(VOCAB)}
_ITOS = {i: tok for i, tok in enumerate(VOCAB)}


def encode(text: str) -> list[int]:
    return [_STOI[ch] for ch in text]


def decode(ids) -> str:
    out = []
    for i in ids:
        i = int(i)
        if i == PAD_ID:
            continue
        out.append(_ITOS[i])
    return "".join(out)
