"""Sanity-check the model and data pipeline without running a full training job.

Run from the repo root::

    python scripts/smoke_test.py
"""
from __future__ import annotations

import sys
import pathlib

# Make src/ importable when running from repo root without `pip install -e .`
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "src"))

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from addition_transformer.data import (
    MAX_LEN,
    build_arrays,
    encode_example,
    generate_pairs,
    render,
)
from addition_transformer.model import Transformer, TransformerConfig, count_params
from addition_transformer.vocab import VOCAB, VOCAB_SIZE


def main() -> None:
    print(f"device = {jax.devices()[0]}")
    print(f"vocab  = {VOCAB} (size {VOCAB_SIZE})")

    # --- Data
    ex = encode_example(123, 456, op="addition", reverse_answer=True)
    p, ans = render(123, 456, op="addition", reverse_answer=True)
    print(f"\nexample: prompt={p!r}  answer={ans!r}  (= 579 reversed)")
    print(f"  input_ids = {ex.input_ids.tolist()}")
    print(f"  targets   = {ex.targets.tolist()}")
    print(f"  loss_mask = {ex.loss_mask.tolist()}")
    assert ex.input_ids.shape == (MAX_LEN,)
    assert ex.loss_mask.sum() == len(ans) + 1, (
        f"loss mask should cover {len(ans)} answer tokens + 1 EOS, got {ex.loss_mask.sum()}"
    )

    # --- Model
    cfg = TransformerConfig()
    model = Transformer(cfg, rngs=nnx.Rngs(0))
    n = count_params(model)
    print(f"\nmodel params = {n:,}  (~{n/1e6:.2f}M)")
    assert 9_000_000 < n < 12_000_000, f"unexpected param count: {n}"

    # --- Forward pass
    pairs = generate_pairs("addition", max_digits=3)[:8]
    inp, tgt, mask, dp = build_arrays(pairs, "addition")
    logits = model(jnp.asarray(inp))
    print(f"forward pass: input {inp.shape}  ->  logits {logits.shape}")
    print(f"digit_positions sample: {dp[0].tolist()}")
    assert logits.shape == (8, MAX_LEN, VOCAB_SIZE)

    # --- Loss (manual)
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    tgt_lp = jnp.take_along_axis(log_probs, jnp.asarray(tgt)[..., None], axis=-1).squeeze(-1)
    nll = -tgt_lp * jnp.asarray(mask)
    loss = nll.sum() / jnp.maximum(jnp.asarray(mask).sum(), 1)
    print(f"untrained masked loss = {float(loss):.4f}  (random baseline ~ ln({VOCAB_SIZE}) = {np.log(VOCAB_SIZE):.4f})")

    print("\nOK - smoke test passed.")


if __name__ == "__main__":
    main()
