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

from .data import MAX_LEN, Op, build_arrays, generate_pairs, render, split
from .model import Transformer, TransformerConfig, count_params
from .vocab import PAD_ID, VOCAB, encode


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------


def loss_fn(model: Transformer, batch):
    input_ids, targets, loss_mask = batch
    logits = model(input_ids)
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
def _step_logits(model: Transformer, ids: jax.Array) -> jax.Array:
    return model(ids)


def greedy_generate(
    model: Transformer,
    prompt_ids: np.ndarray,
    prompt_lens: np.ndarray,
    max_len: int,
) -> np.ndarray:
    """Autoregressive greedy decoding, batched over variable-length prompts.

    Inefficient (re-runs the whole forward pass each step, no KV cache) but
    simple — answers are <=6 tokens long so it's plenty fast.
    """
    ids = jnp.asarray(prompt_ids)
    prompt_lens_j = jnp.asarray(prompt_lens)
    B = ids.shape[0]
    batch_idx = jnp.arange(B)
    n_steps = max_len - int(prompt_lens.min())
    for step in range(n_steps):
        logits = _step_logits(model, ids)
        read_pos = prompt_lens_j + step - 1
        write_pos = prompt_lens_j + step
        # logits at read_pos for each example
        gathered = logits[batch_idx, read_pos]   # (B, V)
        next_tok = jnp.argmax(gathered, axis=-1)
        # only write where we still have room
        mask = write_pos < max_len
        next_tok = jnp.where(mask, next_tok, PAD_ID)
        write_pos_safe = jnp.where(mask, write_pos, 0)
        ids = ids.at[batch_idx, write_pos_safe].set(next_tok)
    return np.asarray(ids)


def evaluate(
    model: Transformer,
    pairs: np.ndarray,
    op: Op,
    *,
    reverse_answer: bool,
    batch_size: int = 512,
    max_len: int = MAX_LEN,
) -> float:
    """Exact-match accuracy over (a, b) pairs."""
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
        out = greedy_generate(model, prompt_ids, prompt_lens_arr, max_len)
        for i in range(B):
            plen = prompt_lens[i]
            gen = out[i, plen:]
            # Stop at first PAD
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
        "--eval-samples",
        type=int,
        default=2000,
        help="How many val examples to use for exact-match accuracy.",
    )
    args = p.parse_args(argv)

    op: Op = args.op
    reverse = not args.no_reverse_answer

    print(f"[setup] op={op}  reverse_answer={reverse}  device={jax.devices()[0]}")

    # ---- Data
    pairs = generate_pairs(op, max_digits=args.max_digits, seed=args.seed)
    train_pairs, val_pairs = split(pairs, val_frac=args.val_frac)
    print(f"[data]  train={len(train_pairs):,}  val={len(val_pairs):,}")

    train_inp, train_tgt, train_mask = build_arrays(
        train_pairs, op, reverse_answer=reverse
    )

    # ---- Model + optimizer
    cfg = TransformerConfig()
    model = Transformer(cfg, rngs=nnx.Rngs(args.seed))
    n_params = count_params(model)
    print(f"[model] params={n_params/1e6:.2f}M  d_model={cfg.d_model}  layers={cfg.n_layers}")

    steps_per_epoch = len(train_pairs) // args.batch_size
    total_steps = steps_per_epoch * args.epochs
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=args.lr,
        warmup_steps=args.warmup_steps,
        decay_steps=total_steps,
        end_value=args.lr * 0.1,
    )
    tx = optax.adamw(schedule, weight_decay=args.weight_decay, b1=0.9, b2=0.95)
    optimizer = nnx.Optimizer(model, tx, wrt=nnx.Param)

    # ---- Train loop
    rng = np.random.default_rng(args.seed)
    global_step = 0
    t_start = time.time()
    for epoch in range(args.epochs):
        perm = rng.permutation(len(train_pairs))
        pbar = tqdm(range(steps_per_epoch), desc=f"epoch {epoch+1}/{args.epochs}")
        losses = []
        for step in pbar:
            sel = perm[step * args.batch_size : (step + 1) * args.batch_size]
            batch = (
                jnp.asarray(train_inp[sel]),
                jnp.asarray(train_tgt[sel]),
                jnp.asarray(train_mask[sel]),
            )
            loss_val = train_step(model, optimizer, batch)
            losses.append(float(loss_val))
            global_step += 1
            if step % 50 == 0:
                pbar.set_postfix(loss=f"{np.mean(losses[-50:]):.4f}")

        # ---- Eval
        n_eval = min(args.eval_samples, len(val_pairs))
        sample_idx = rng.choice(len(val_pairs), size=n_eval, replace=False)
        acc = evaluate(
            model,
            val_pairs[sample_idx],
            op,
            reverse_answer=reverse,
        )
        elapsed = time.time() - t_start
        print(
            f"[eval]  epoch {epoch+1}: exact-match acc on {n_eval} val examples = "
            f"{acc*100:.2f}%  (elapsed {elapsed:.1f}s)"
        )

    # ---- A few qualitative samples
    print("\n[samples]")
    show = val_pairs[: 8]
    prompts, plens, expected = [], [], []
    for a, b in show:
        pr, ans = render(int(a), int(b), op, reverse_answer=reverse)
        prompts.append(encode(pr) + [PAD_ID] * (MAX_LEN - len(pr)))
        plens.append(len(pr))
        expected.append(ans)
    out = greedy_generate(
        model,
        np.asarray(prompts, dtype=np.int32),
        np.asarray(plens, dtype=np.int32),
        MAX_LEN,
    )
    for i, (a, b) in enumerate(show):
        gen = out[i, plens[i]:]
        pad_positions = np.where(gen == PAD_ID)[0]
        cut = pad_positions[0] if len(pad_positions) else len(gen)
        got = "".join(VOCAB[int(t)] for t in gen[:cut])
        ok = "OK" if got == expected[i] else "XX"
        sym = "+" if op == "addition" else "*"
        # un-reverse for display so the human-readable form is shown
        got_disp = got[::-1] if reverse else got
        exp_disp = expected[i][::-1] if reverse else expected[i]
        print(f"  [{ok}] {a} {sym} {b} = {got_disp}   (expected {exp_disp})")


if __name__ == "__main__":
    main()
