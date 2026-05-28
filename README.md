# jax transformer

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sananthanarayan/jax-transformer/blob/main/notebooks/length_generalization.ipynb)

A ~10M-parameter decoder-only transformer in [JAX](https://github.com/google/jax) +
[Flax NNX](https://flax.readthedocs.io/en/latest/nnx_basics.html) that learns to add
(and then multiply) up-to-3-digit numbers from scratch — and a length-generalization
experiment showing why learned positional embeddings fall off a cliff outside the
training distribution.

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

### Reproducing the length-generalization figure

```bash
make figures        # full 6-variant sweep + both plots (~25-40 min on a 14 GB GPU)
make sweep-phase1   # just the 3 Phase 1 variants (~10-15 min)
make quick          # 2-epoch sanity sweep across all variants (~5-10 min)
make test           # unit tests for the Abacus digit-position assignment
bash scripts/reproduce.sh   # same as `make figures`, also works without make
```

The headline chart lands at `results/length_gen.png` and the per-digit-position
heatmap at `results/per_digit_heatmap.png`; raw numbers are written to
`results/sweep.json` so a forker can re-plot without retraining.

## Length generalization (the headline experiment)

Train a model on operands with ≤3 digits, then test it on operands with 1–6 digits.
Six variants are compared, all on the same architecture and training schedule
(except where noted):

| Variant | Positional encoding | Answer order | Training data |
| --- | --- | --- | --- |
| `baseline` | learned | natural (`579`) | ≤3-digit |
| `reversed` | learned | reversed (`975`) | ≤3-digit |
| `nope` | none | reversed | ≤3-digit |
| `rope` | rotary ([Su et al. 2021](https://arxiv.org/abs/2104.09864)) | reversed | ≤3-digit |
| `abacus` | digit-position only ([McLeish et al. 2024](https://arxiv.org/abs/2405.17399)) | reversed | ≤3-digit |
| `abacus_curriculum` | digit-position only | reversed | **mixed 1..5-digit** |

The expected shape:

- **`baseline`** cliffs at 4 digits. Learned PE entries at positions never used during
  training stay at random initialization (PAD positions are causally inaccessible and
  loss-masked, so no gradient reaches them), and the model has no way to extrapolate.
- **`reversed`** improves in-distribution accuracy noticeably (carries flow with the
  causal direction) but still cliffs at 4 digits for the same PE reason.
- **`nope`** has no positional embeddings at all. The model leans on the causal-mask
  asymmetry alone — which, per [Kazemnejad et al. 2023](https://arxiv.org/abs/2305.19466),
  extrapolates further than learned PE.
- **`rope`** applies position-dependent rotation to Q and K inside attention. The
  relative-position structure of RoPE generalizes to unseen offsets in principle.
- **`abacus`** replaces absolute position with *place value* — each digit token gets
  an embedding indexed by "this is the k-th digit of its number, counted from the
  ones place." Trained on 3-digit only, its place-value embeddings for positions ≥4
  never see gradient (same failure as learned PE, just reparameterized).
- **`abacus_curriculum`** uses the same place-value embedding, but trains on operands
  with mixed digit counts (1..5) so every embedding gets gradient. This is the
  variant that should actually generalize.

Run `make figures` to produce the charts for yourself; the
[Colab notebook](notebooks/length_generalization.ipynb) is the one-click version.
Three figures are written:

- `results/length_gen.png` — exact-match accuracy vs operand digit count, one curve per variant.
- `results/per_digit_heatmap.png` — per-digit-position error rate, one panel per variant
  (reveals *where* in the answer each variant fails first — typically the high-order digits).
- `results/embedding_drift.png` — L2 distance of each position embedding row from its
  initialization. A row that received no gradient during training has drift exactly 0,
  so this figure is the direct, falsifiable version of the claim that learned-PE and
  clean-Abacus fail because the relevant embedding rows *never moved from init*.

The third figure is what makes the "untrained-positions-stay-at-init" story
verifiable rather than just plausible: you can read the failure mode straight off
the JSON.

## Project layout

```
src/addition_transformer/
  vocab.py        # token table, encode/decode
  data.py         # dataset generation, sampled OOD pairs, length helpers
  model.py        # Flax NNX transformer (learned PE or NoPE)
  train.py        # training loop + greedy eval + CLI
  eval.py         # length-generalization evaluation
scripts/
  smoke_test.py   # builds the model, prints param count, runs one forward pass
  run_sweep.py    # trains the variants, writes results/sweep.json
  plot.py         # produces results/length_gen.png + results/per_digit_heatmap.png
  reproduce.sh    # install + sweep + plot, end-to-end
tests/
  test_digit_positions.py   # unit tests for the Abacus place-value assignment
notebooks/
  length_generalization.ipynb   # Colab-friendly version of the sweep
results/                          # generated figures + raw JSON
Makefile                          # make figures | quick | smoke | test | clean
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
