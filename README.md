# jax-addition-transformer

A ~10M-parameter decoder-only transformer in [JAX](https://github.com/google/jax) +
[Flax NNX](https://flax.readthedocs.io/en/latest/nnx_basics.html) that learns to add
(and then multiply) up-to-3-digit numbers from scratch.

The whole thing is small enough to train in a few minutes on a single 14 GB GPU
(e.g. Colab T4 / L4) and is meant as a clean, hackable starting point for
arithmetic-reasoning experiments.

## What it does

- Tokenizes character sequences over a tiny hardcoded vocab:
  `<pad>`, `0`-`9`, ` `, `+`, `*`, `=` (15 tokens).
- Generates a synthetic dataset of problems like `"123 + 456 = 579"`, padded to a
  fixed length of 20 characters.
- Trains a decoder-only transformer autoregressively, computing the loss **only on
  the answer tokens** (the prompt and padding are masked out).
- Reports exact-match accuracy on a held-out test split.

The answer is reversed by default (`"579"` is emitted as `"975"`), which makes
multi-digit carries much easier for a small transformer to learn — this is a
well-known trick from the
[Nogueira et al. (2021)](https://arxiv.org/abs/2102.13019) line of work on
arithmetic transformers. Disable it with `--no-reverse-answer`.

## Architecture

| Hyperparameter | Value |
| --- | --- |
| `d_model` | 384 |
| `n_layers` | 6 |
| `n_heads` | 6 |
| `d_ff` | 1536 |
| `max_len` | 20 |
| `vocab_size` | 15 |
| **Total params** | **~10.6 M** |

Standard pre-norm decoder-only transformer (RMSNorm, GeLU MLPs, learned positional
embeddings, untied output head, causal self-attention).

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Train addition (default). Takes ~2-5 min on a single 14 GB GPU.
python -m addition_transformer.train --op addition

# Then try multiplication.
python -m addition_transformer.train --op multiplication
```

CPU also works for addition but is slow; expect ~30 min on a modern laptop.

## Project layout

```
src/addition_transformer/
  vocab.py        # token table, encode/decode
  data.py         # dataset generation + batching
  model.py        # Flax NNX transformer
  train.py        # training loop + eval + CLI
scripts/
  smoke_test.py   # builds the model, prints param count, runs one forward pass
```

## Why these choices?

- **Decoder-only, not encoder-decoder.** The task is small enough that an LM-style
  setup with prompt-conditioned generation is simpler and trains just as well.
- **Loss masked to the answer.** The model gets no credit for re-emitting the
  prompt, so all gradient signal goes into learning the arithmetic.
- **Reversed answer.** Predicting least-significant-digit first lets carries flow
  left-to-right, matching the natural causal direction of the transformer.
- **Synthetic full-coverage dataset.** All 10⁶ pairs `(a, b) ∈ [0, 999]²` are
  enumerated and shuffled, then split 95/5. The model still has to generalize
  across the order it sees them.

## License

MIT — see [LICENSE](LICENSE).
