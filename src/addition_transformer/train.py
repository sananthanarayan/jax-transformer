"""Training loop + CLI.

Run::

    python -m addition_transformer.train --op addition
    python -m addition_transformer.train --op multiplication
"""
from __future__ import annotations

import argparse
import time

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx
from tqdm import tqdm

from .data import (
    MAX_LEN,
    Op,
    build_arrays,
    compute_digit_positions_for_decode,
    generate_mixed_pairs,
    generate_pairs,
    max_len_for,
    render,
    split,
)
from .model import PosEncoding, Transformer, TransformerConfig, count_params
from .vocab import PAD_ID, VOCAB, encode


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------


def loss_fn(model: Transformer, batch):
    input_ids, targets, loss_mask, digit_positions = batch
    logits = model(input_ids, digit_positions)
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    tgt_log_probs = jnp.take_along_axis(
        log_probs, targets[..., None], axis=-1
    ).squeeze(-1)
    nll = -tgt_log_probs * loss_mask
    n = jnp.maximum(loss_mask.sum(), 1)
    return nll.sum() / n


# ---------------------------------------------------------------------------
# Eval — autoregressive greedy decode + exact match
# ---------------------------------------------------------------------------


@nnx.jit
def _step_logits(
    model: Transformer, ids: jax.Array, digit_positions: jax.Array
) -> jax.Array:
    return model(ids, digit_positions)


def greedy_generate(
    model: Transformer,
    prompt_ids: np.ndarray,
    prompt_lens: np.ndarray,
    max_len: int,
    digit_positions: np.ndarray | None = None,
) -> np.ndarray:
    """Autoregressive greedy decoding, batched over variable-length prompts.

    Inefficient (re-runs the whole forward pass each step, no KV cache) but
    simple — answers are <=8 tokens long so it's plenty fast. ``digit_positions``
    is required for Abacus models; pass zeros (or omit) otherwise.
    """
    ids = jnp.asarray(prompt_ids)
    if digit_positions is None:
        digit_positions = np.zeros_like(prompt_ids, dtype=np.int32)
    dp = jnp.asarray(digit_positions)
    prompt_lens_j = jnp.asarray(prompt_lens)
    B = ids.shape[0]
    batch_idx = jnp.arange(B)
    n_steps = max_len - int(prompt_lens.min())
    for step in range(n_steps):
        logits = _step_logits(model, ids, dp)
        read_pos = prompt_lens_j + step - 1
        write_pos = prompt_lens_j + step
        gathered = logits[batch_idx, read_pos]   # (B, V)
        next_tok = jnp.argmax(gathered, axis=-1)
        mask = write_pos < max_len
        next_tok = jnp.where(mask, next_tok, PAD_ID)
        write_pos_safe = jnp.where(mask, write_pos, 0)
        ids = ids.at[batch_idx, write_pos_safe].set(next_tok)
    return np.asarray(ids)


def _build_eval_digit_positions(
    prompt_ids: np.ndarray,
    prompt_lens: np.ndarray,
    model_max_len: int,
    reverse_answer: bool,
) -> np.ndarray:
    """Per-example digit positions for greedy decoding under Abacus."""
    B = prompt_ids.shape[0]
    out = np.zeros((B, model_max_len), dtype=np.int32)
    for i in range(B):
        out[i] = compute_digit_positions_for_decode(
            prompt_ids[i],
            int(prompt_lens[i]),
            model_max_len=model_max_len,
            reverse_answer=reverse_answer,
        )
    return out


def evaluate(
    model: Transformer,
    pairs: np.ndarray,
    op: Op,
    *,
    reverse_answer: bool,
    batch_size: int = 512,
    max_len: int | None = None,
) -> float:
    """Exact-match accuracy over (a, b) pairs."""
    if max_len is None:
        max_len = int(model.cfg.max_len)
    use_abacus = model.cfg.pos_encoding == "abacus"
    correct = 0
    total = 0
    for start in range(0, len(pairs), batch_size):
        chunk = pairs[start : start + batch_size]
        B = len(chunk)
        prompts = []
        prompt_lens = []
        expected = []
        for a, b in chunk:
            p, ans = render(int(a), int(b), op, reverse_answer=reverse_answer)
            ids = encode(p) + [PAD_ID] * (max_len - len(p))
            prompts.append(ids)
            prompt_lens.append(len(p))
            expected.append(ans)
        prompt_ids = np.asarray(prompts, dtype=np.int32)
        prompt_lens_arr = np.asarray(prompt_lens, dtype=np.int32)
        if use_abacus:
            dp = _build_eval_digit_positions(
                prompt_ids, prompt_lens_arr, max_len, reverse_answer
            )
        else:
            dp = None
        out = greedy_generate(model, prompt_ids, prompt_lens_arr, max_len, dp)
        for i in range(B):
            plen = prompt_lens[i]
            gen = out[i, plen:]
            pad_positions = np.where(gen == PAD_ID)[0]
            cut = pad_positions[0] if len(pad_positions) else len(gen)
            generated_chars = "".join(VOCAB[int(t)] for t in gen[:cut])
            if generated_chars == expected[i]:
                correct += 1
            total += 1
    return correct / max(total, 1)


# ---------------------------------------------------------------------------
# Train step
# ---------------------------------------------------------------------------


@nnx.jit
def train_step(model: Transformer, optimizer: nnx.Optimizer, batch):
    def loss(m):
        return loss_fn(m, batch)

    grad_fn = nnx.value_and_grad(loss)
    loss_val, grads = grad_fn(model)
    optimizer.update(model, grads)
    return loss_val


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def train_model(
    *,
    op: Op = "addition",
    max_digits: int = 3,
    epochs: int = 8,
    batch_size: int = 512,
    lr: float = 3e-4,
    weight_decay: float = 0.01,
    warmup_steps: int = 200,
    seed: int = 0,
    val_frac: float = 0.05,
    reverse_answer: bool = True,
    pos_encoding: PosEncoding = "learned",
    model_max_len: int | None = None,
    eval_samples: int = 2000,
    mixed_digits: bool = False,
    mixed_min_digits: int = 1,
    mixed_n_samples: int = 1_000_000,
    log_prefix: str = "",
    verbose: bool = True,
) -> Transformer:
    """Train a transformer on synthetic arithmetic and return the trained model.

    With ``mixed_digits=True`` the training set is sampled uniformly across
    operand digit counts ``[mixed_min_digits, max_digits]`` (used by the Abacus
    length-curriculum variant so every digit-position embedding gets gradient).
    """
    if model_max_len is None:
        model_max_len = max_len_for(op, max_digits)

    def log(msg: str) -> None:
        if verbose:
            print(f"{log_prefix}{msg}")

    log(f"[setup] op={op}  reverse_answer={reverse_answer}  pos_encoding={pos_encoding}  "
        f"max_digits={max_digits}  mixed_digits={mixed_digits}  "
        f"model_max_len={model_max_len}  device={jax.devices()[0]}")

    if mixed_digits:
        pairs = generate_mixed_pairs(
            op, mixed_min_digits, max_digits, mixed_n_samples, seed=seed
        )
    else:
        pairs = generate_pairs(op, max_digits=max_digits, seed=seed)
    train_pairs, val_pairs = split(pairs, val_frac=val_frac)
    log(f"[data]  train={len(train_pairs):,}  val={len(val_pairs):,}")

    train_inp, train_tgt, train_mask, train_dp = build_arrays(
        train_pairs, op, max_len=model_max_len, reverse_answer=reverse_answer
    )

    cfg = TransformerConfig(max_len=model_max_len, pos_encoding=pos_encoding)
    model = Transformer(cfg, rngs=nnx.Rngs(seed))
    n_params = count_params(model)
    log(f"[model] params={n_params/1e6:.2f}M  d_model={cfg.d_model}  layers={cfg.n_layers}")

    steps_per_epoch = len(train_pairs) // batch_size
    total_steps = steps_per_epoch * epochs
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=lr,
        warmup_steps=warmup_steps,
        decay_steps=total_steps,
        end_value=lr * 0.1,
    )
    tx = optax.adamw(schedule, weight_decay=weight_decay, b1=0.9, b2=0.95)
    optimizer = nnx.Optimizer(model, tx, wrt=nnx.Param)

    rng = np.random.default_rng(seed)
    t_start = time.time()
    for epoch in range(epochs):
        perm = rng.permutation(len(train_pairs))
        iterator = range(steps_per_epoch)
        pbar = tqdm(iterator, desc=f"{log_prefix}epoch {epoch+1}/{epochs}") if verbose else iterator
        losses = []
        for step in pbar:
            sel = perm[step * batch_size : (step + 1) * batch_size]
            batch = (
                jnp.asarray(train_inp[sel]),
                jnp.asarray(train_tgt[sel]),
                jnp.asarray(train_mask[sel]),
                jnp.asarray(train_dp[sel]),
            )
            loss_val = train_step(model, optimizer, batch)
            losses.append(float(loss_val))
            if verbose and step % 50 == 0:
                pbar.set_postfix(loss=f"{np.mean(losses[-50:]):.4f}")

        n_eval = min(eval_samples, len(val_pairs))
        sample_idx = rng.choice(len(val_pairs), size=n_eval, replace=False)
        acc = evaluate(
            model,
            val_pairs[sample_idx],
            op,
            reverse_answer=reverse_answer,
            max_len=model_max_len,
        )
        elapsed = time.time() - t_start
        log(f"[eval]  epoch {epoch+1}: exact-match acc on {n_eval} val examples = "
            f"{acc*100:.2f}%  (elapsed {elapsed:.1f}s)")

    return model


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--op", choices=["addition", "multiplication"], default="addition")
    p.add_argument("--max-digits", type=int, default=3)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup-steps", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--val-frac", type=float, default=0.05)
    p.add_argument("--no-reverse-answer", action="store_true")
    p.add_argument(
        "--pos-encoding",
        choices=["learned", "none", "rope", "abacus"],
        default="learned",
    )
    p.add_argument("--model-max-len", type=int, default=None)
    p.add_argument("--mixed-digits", action="store_true",
                   help="Sample training data uniformly across operand digit counts "
                        "[1, max_digits] (used by the Abacus length curriculum).")
    p.add_argument("--mixed-n-samples", type=int, default=1_000_000)
    p.add_argument(
        "--eval-samples",
        type=int,
        default=2000,
        help="How many val examples to use for exact-match accuracy.",
    )
    args = p.parse_args(argv)

    op: Op = args.op
    reverse = not args.no_reverse_answer

    model = train_model(
        op=op,
        max_digits=args.max_digits,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        seed=args.seed,
        val_frac=args.val_frac,
        reverse_answer=reverse,
        pos_encoding=args.pos_encoding,
        model_max_len=args.model_max_len,
        eval_samples=args.eval_samples,
        mixed_digits=args.mixed_digits,
        mixed_n_samples=args.mixed_n_samples,
    )

    # Qualitative samples from the validation split.
    print("\n[samples]")
    pairs = generate_pairs(op, max_digits=args.max_digits, seed=args.seed)
    _, val_pairs = split(pairs, val_frac=args.val_frac)
    show = val_pairs[:8]
    model_max_len = int(model.cfg.max_len)
    prompts, plens, expected = [], [], []
    for a, b in show:
        pr, ans = render(int(a), int(b), op, reverse_answer=reverse)
        prompts.append(encode(pr) + [PAD_ID] * (model_max_len - len(pr)))
        plens.append(len(pr))
        expected.append(ans)
    prompt_ids_arr = np.asarray(prompts, dtype=np.int32)
    plens_arr = np.asarray(plens, dtype=np.int32)
    dp = (
        _build_eval_digit_positions(prompt_ids_arr, plens_arr, model_max_len, reverse)
        if model.cfg.pos_encoding == "abacus"
        else None
    )
    out = greedy_generate(model, prompt_ids_arr, plens_arr, model_max_len, dp)
    for i, (a, b) in enumerate(show):
        gen = out[i, plens[i]:]
        pad_positions = np.where(gen == PAD_ID)[0]
        cut = pad_positions[0] if len(pad_positions) else len(gen)
        got = "".join(VOCAB[int(t)] for t in gen[:cut])
        ok = "OK" if got == expected[i] else "XX"
        sym = "+" if op == "addition" else "*"
        got_disp = got[::-1] if reverse else got
        exp_disp = expected[i][::-1] if reverse else expected[i]
        print(f"  [{ok}] {a} {sym} {b} = {got_disp}   (expected {exp_disp})")


if __name__ == "__main__":
    main()
