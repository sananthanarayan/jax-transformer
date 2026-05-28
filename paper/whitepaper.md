# Where do small transformers fail at multi-digit addition, and can we see it directly in the weights?

**A controlled comparison of positional encoding schemes on a 10M-parameter arithmetic transformer.**

*Draft — results pending the full sweep. Numerical placeholders are marked `[pending]`.*

---

## Abstract

We train a 10.6M-parameter decoder-only transformer on 3-digit addition and test its ability to generalize to operand sizes never seen at training time (1–6 digits). Six positional-encoding variants are compared on the same architecture and schedule: learned absolute, learned absolute with reversed answers, no positional encoding (NoPE), rotary (RoPE), Abacus-style place-value embeddings, and Abacus trained with a length curriculum. Beyond reporting exact-match accuracy at each digit count, we introduce a simple **embedding-drift diagnostic**: the L2 distance of each position-embedding row from its initialization. A row with drift exactly zero received no gradient during training, which directly identifies which positions the model was never exposed to. This makes the "untrained positions stay at init" story mechanistic and falsifiable, rather than something we have to infer from the loss curve.

## 1. Introduction

Length generalization on arithmetic is a sharp probe for what a transformer learns about position. The task is small enough that we can run controlled experiments on a single GPU in minutes; the rules of the task are unambiguous; and yet the model has to *systematically* extend a pattern to inputs longer than anything it saw. This is the setting in which length-generalization failures of standard transformers were first cleanly demonstrated [@anil2022], and the setting in which several recent proposals (NoPE [@kazemnejad2023], Abacus embeddings [@mcleish2024], scratchpad reasoning [@nye2021]) have been benchmarked.

Our setup is deliberately minimal:

- A 10.6M-parameter decoder-only transformer with pre-norm, GeLU MLPs, RMSNorm, and causal self-attention (no fancy attention variants).
- A 15-token character-level vocabulary (digits, space, `+`, `*`, `=`, `<pad>`).
- Synthetic data: `"{a} + {b} = {result}"`, optionally with the answer reversed (`"579"` written as `"975"` so carries flow with the causal direction [@nogueira2021]).
- Training on the full 10⁶ pairs of 3-digit operands, with held-out OOD evaluation at 1, 2, 3, 4, 5, and 6 digits.

The contributions of this report are:

1. **A clean six-variant comparison** of positional encodings on the same architecture, training schedule, and data. Most prior work compares fewer variants or varies more than one factor at a time.
2. **The embedding-drift diagnostic**, which turns the abstract claim "positions never seen at training stay at random init" into a measurable per-row L2 number that anyone can read off the model's weights.
3. **A length-curriculum control** for the Abacus variant. This isolates two distinct questions that the place-value parameterization combines: (a) does Abacus help if the relevant position embeddings receive gradient (curriculum case), and (b) does Abacus help *without* a curriculum, just by virtue of its parameterization?

The code, configuration, and the full reproducibility pipeline (`make figures`) are open-source.

## 2. Setup

### Model

| Hyperparameter | Value |
|---|---|
| Layers | 6 |
| `d_model` | 384 |
| Attention heads | 6 |
| `d_head` | 64 |
| `d_ff` | 1536 |
| Activation | GeLU |
| Normalization | RMSNorm (pre-norm) |
| Total parameters | ~10.66M |

Causal self-attention. Output head is untied from the input embedding. Default `max_len = 32`, set by `max_len_for("addition", 6)` so the model can ingest sequences up to 6-digit-operand addition without architectural changes between in- and out-of-distribution evaluation.

### Tokenization

Character-level vocabulary, 15 tokens total:

```
<pad>  '0' '1' '2' '3' '4' '5' '6' '7' '8' '9'  ' '  '+'  '*'  '='
```

The longest possible 3-digit-addition example is `"999 + 999 = 1998"` (16 characters); the longest 6-digit example is `"999999 + 999999 = 1999998"` (25 characters).

### Training

- **Optimizer**: AdamW (β₁ = 0.9, β₂ = 0.95, weight decay 0.01).
- **Schedule**: 200-step linear warmup, cosine decay to 10% of peak.
- **Peak LR**: 3 × 10⁻⁴.
- **Batch size**: 512.
- **Epochs**: 8.
- **Loss**: masked cross-entropy on answer tokens *and one terminal pad slot* (the latter teaches the model to stop). Loss on prompt tokens and on pad tokens beyond the first is masked out.

### Evaluation

For each digit count d ∈ {1, …, 6} we sample 500 random (a, b) pairs with both operands having exactly d digits (operands in [10^(d-1), 10^d), or [0, 10) for d=1). Greedy decoding from the prompt `"a + b = "`, exact-match on the entire decoded answer string. We additionally collect per-digit-position correctness over the same sample, which feeds the heatmap of Section 6.

## 3. Positional encoding variants

We hold architecture, optimizer, schedule, and data identical across variants. The single axis of variation is how (and whether) position information enters the model.

| Variant | Positional encoding | Answer order | Training data |
|---|---|---|---|
| `baseline` | learned absolute embedding | natural (`579`) | ≤3-digit |
| `reversed` | learned absolute embedding | reversed (`975`) | ≤3-digit |
| `nope` | none | reversed | ≤3-digit |
| `rope` | rotary applied to Q, K | reversed | ≤3-digit |
| `abacus` | digit-position embedding only | reversed | ≤3-digit |
| `abacus_curriculum` | digit-position embedding only | reversed | **mixed 1..5-digit** |

### 3.1 Learned absolute position embedding

The standard variant from the original transformer: a learned `Embed(max_len, d_model)` table whose row k is added to the token embedding at sequence position k. Two flavors:

- `baseline`: natural answer order. The model emits the most significant digit first.
- `reversed`: the answer is reversed at training time, so the model emits the ones digit first. Carries flow left-to-right, matching the causal direction.

The reversed flavor is the well-known "Nogueira trick" [@nogueira2021] and is included for two reasons. First, it isolates whether the gain from `reversed` over `baseline` comes from answer order (the carry-friendly direction) or from some other interaction. Second, it provides a fair baseline against which to evaluate the other reversed-answer variants (`nope`, `rope`, `abacus`).

### 3.2 NoPE

No positional encoding is added at all. The model relies only on the causal-mask asymmetry to break symmetry between positions. Despite this seeming impoverishment, NoPE has been shown [@kazemnejad2023] to extrapolate further than learned absolute PE in several settings, presumably because there is no out-of-distribution position to be confused by.

### 3.3 RoPE

Rotary position embeddings [@su2021]. We precompute a `(T, d_head/2)` table of cos/sin angles `pos · 10000^(-2k/d_head)` and rotate pairs of feature dimensions of Q and K inside `CausalSelfAttention` before computing attention scores. V is not rotated. This formulation expresses position information through *relative* offsets in the attention dot product, which in principle generalizes to unseen absolute positions as long as the relative offsets stay in distribution.

### 3.4 Abacus place-value embeddings

Following McLeish et al. [@mcleish2024], we drop absolute position information entirely and instead embed *each digit token's place value*. The procedure:

1. Identify each contiguous run of digit tokens in the input.
2. For the first two runs (operands, MSB-first by convention), assign position `run_length − index` so the ones digit gets position 1, tens get 2, hundreds get 3, etc.
3. For the third run (the answer, LSB-first under reversed answers), assign position `index + 1` directly — the leftmost digit of the reversed string is the ones digit.
4. Non-digit tokens (including pad, space, `+`, `=`) receive position 0, a distinct "no-position" embedding.

A `(max_digit_pos + 1, d_model)` learned table is added at every position in the input. With `max_digit_pos = 16` (capping at 16-digit place values), the additional parameter count is negligible relative to the 10.6M total.

Critically, the place-value embedding for position k receives gradient **only if a training example contains a digit at place value k**. For 3-digit-operand training:

- Positions 1, 2, 3 (ones, tens, hundreds) appear in every operand. ✓
- Position 4 (thousands) appears whenever the sum carries, e.g. `999 + 999 = 1998`. ✓
- Positions 5, 6, …, 16 **never appear**. ✗

This produces the same failure mode as learned absolute PE — embeddings for unseen places stay at initialization — but localized differently in the table. The next variant addresses this.

### 3.5 Abacus with a length curriculum

Same architecture and embedding scheme as `abacus`, but the training data is sampled uniformly across operand digit counts in [1, 5] rather than fixed at 3 digits. With this data distribution, every place-value embedding from position 1 to position 6 (5-digit operands + carry) receives gradient during training. The model is then evaluated at digit counts up to 6, requiring it to extrapolate by exactly one place value.

This variant separates two concerns the McLeish et al. result implicitly bundles: (a) the parameterization (place value rather than absolute position) and (b) the training distribution covering the relevant positions. If `abacus_curriculum` generalizes substantially better than `abacus`, then the parameterization advantage is real but conditional on adequate data coverage.

## 4. The embedding-drift diagnostic

Most discussion of "positions stay at random init" in the length-generalization literature appears as a *plausibility argument*: gradient cannot reach embedding rows associated with positions that the model never has to predict from. We promote this to a directly measurable quantity.

Define, for each row k of a positional embedding table E:

```
drift_k = || E_trained[k, :] − E_init[k, :] ||₂
```

where `E_init` is computed by re-instantiating the model with the same RNG seed used to train it. Because PyTorch- and JAX-style initialization is deterministic at fixed seed, `E_init` is bit-identical to the trained model's initial state. A row that received zero gradient over training has `drift_k = 0.0` exactly.

We apply this to two tables:

- `pos_emb` (learned absolute PE): we expect `drift_k > 0` for positions reachable from the loss-masked region of the longest training example, and `drift_k = 0` for higher positions.
- `digit_pos_emb` (Abacus): we expect `drift_k > 0` for place values present in the training data, and `drift_k = 0` beyond.

The diagnostic is cheap (a single L2 norm per row of one or two embeddings) and is stored alongside per-variant accuracy in `results/sweep.json`. The accompanying figure (`results/embedding_drift.png`) is, in our view, the most direct evidence of the failure mode that the length-generalization literature has so far argued for indirectly.

A 20-step CPU smoke run reproduces the expected pattern: `pos_emb` positions 16–25 have drift exactly 0.0, and `digit_pos_emb` positions 5–16 of the clean Abacus variant have drift exactly 0.0, while position 4 — the carry into thousands when summing 3-digit numbers — has drift > 0 as expected.

## 5. Experiments

For each variant we (i) train for 8 epochs on the appropriate dataset (3-digit grid for non-curriculum variants; 1M mixed-digit samples for the curriculum), (ii) evaluate exact-match accuracy at digit counts 1 through 6 with 500 sampled pairs each, (iii) compute per-digit-position accuracy over the same sample, and (iv) record per-row drift for any learned positional embedding present in the model.

The full sweep produces a single `results/sweep.json` that drives three figures:

- **Figure 1 (`length_gen.png`)** — exact-match accuracy vs operand digit count, one curve per variant. The headline result.
- **Figure 2 (`per_digit_heatmap.png`)** — per-digit-position error rate, one panel per variant. Reveals whether errors cluster at the most-significant end (typical) or are uniformly distributed (indicates a more fundamental failure).
- **Figure 3 (`embedding_drift.png`)** — per-row L2 drift from init, separated into a learned-absolute panel and an Abacus panel. Reveals which rows actually received gradient.

All artifacts are reproducible from a single command (`make figures`) on a GPU box, or via the included Colab notebook on a free T4.

## 6. Results

*[pending]* — to be populated once the full sweep is run. The structure of the result tables and the qualitative predictions below set up what we expect to see.

### 6.1 Predicted exact-match accuracy

| Variant | 1d | 2d | 3d | 4d | 5d | 6d |
|---|---:|---:|---:|---:|---:|---:|
| `baseline` | ~100% | ~100% | ~100% | ~0% | ~0% | ~0% |
| `reversed` | ~100% | ~100% | ~100% | ~0% | ~0% | ~0% |
| `nope` | [pending] | [pending] | [pending] | [pending] | [pending] | [pending] |
| `rope` | [pending] | [pending] | [pending] | [pending] | [pending] | [pending] |
| `abacus` | ~100% | ~100% | ~100% | ~0% | ~0% | ~0% |
| `abacus_curriculum` | ~100% | ~100% | ~100% | ~100% | ~100% | [pending] |

Predictions: `baseline`, `reversed`, and `abacus` all cliff at 4 digits because their respective extra-distribution position embeddings (positions 16+ for the learned variants; place values 5+ for clean Abacus) sit at random init and contribute pure noise to the residual stream. `nope` and `rope` are expected to degrade gracefully rather than cliff — for `nope`, because there is no out-of-distribution structure to misuse, and for `rope`, because the relative-offset rotation generalizes mechanically. `abacus_curriculum` should be near-ceiling out to 5 digits (the training distribution) with degradation at 6, where one place-value embedding (position 6) is still untouched.

### 6.2 Predicted per-digit-position pattern

For variants that cliff (`baseline`, `reversed`, `abacus`), we expect the heatmap to show full-red columns at digit counts ≥4 (every digit position fails). For variants that degrade gracefully (`nope`, `rope`), we expect a gradient: low-order digits (the ones place) remain mostly correct because they don't depend on long-range carries, while high-order digits fail with increasing reliability as the operand size grows. For `abacus_curriculum`, errors should be sparse and clustered at the highest place values.

### 6.3 Predicted embedding drift

`baseline` and `reversed`: `pos_emb[0:16]` should show drift in the 0.1–0.3 range (depending on optimizer and schedule); `pos_emb[16:]` should be exactly 0. `abacus`: `digit_pos_emb[0:5]` should show drift; `digit_pos_emb[5:]` should be exactly 0. `abacus_curriculum`: `digit_pos_emb[0:7]` should show drift; `digit_pos_emb[7:]` should be exactly 0.

If any of these predictions fail — for instance, if `nope` cliffs as hard as `baseline` — that is itself the interesting finding.

## 7. Discussion

The combination of the three figures, if the predictions hold, would tell a self-contained story about transformer length generalization that has been told in fragments across prior work:

1. **The failure of learned absolute PE at length generalization is mechanistic, not statistical.** It is not that the model "fails to learn" the higher positions; it is that gradient *literally never reaches* those embedding rows. The drift figure makes this directly visible.

2. **Reparameterization alone is not enough.** Abacus changes the *meaning* of the position embedding from "I am at sequence position k" to "I am the k-th place value." But if the training distribution doesn't expose every place value, the same mechanistic failure recurs. The clean-Abacus variant exists in this report precisely to make this point.

3. **The right intervention for length generalization is some combination of mechanism (NoPE/RoPE/Abacus) and data coverage (curriculum).** Neither alone seems to suffice for graceful extrapolation under our setup.

4. **Per-digit-position errors are mostly carries.** The lowest-order digit of a multi-digit sum is computable from the low-order digits of the operands alone — no carry propagation. The highest-order digit requires the entire carry chain to be right. The heatmap is therefore expected to grade red from right (low-order) to left (high-order), and the boundary creeping leftward as digit count grows is essentially a picture of how far the model can propagate carries.

## 8. Limitations

This is a small-scale, single-task, single-seed study. Specifically:

- **Scale.** 10.6M parameters is a few orders of magnitude smaller than typical research models. Results should not be assumed to transfer to billion-parameter models without verification, particularly for NoPE, where the inductive-bias arguments may behave differently with greater capacity.
- **Task.** Addition is much easier than multiplication, and multiplication is much easier than (say) modular exponentiation. We include multiplication support in the codebase but treat addition as the headline task because the failure modes are cleaner.
- **Seeds.** Each variant is trained with a single seed. Error bars across 3–5 seeds would strengthen the claims, particularly near the cliff. Future iterations should include this.
- **Decoding.** All evaluation uses greedy decoding. Beam search or temperature sampling may shift accuracy upward, particularly at the boundary of generalization.
- **Curriculum design.** Our length curriculum is the simplest possible (uniform mixture of digit counts 1–5). Curriculum design itself is a research question we do not engage with here.

## 9. Related work

**Arithmetic transformers and length generalization.** Lee et al. [@lee2023] systematically studied small transformers on arithmetic and identified the answer-order trick (predict ones-first) as critical for in-distribution accuracy. Nye et al. [@nye2021] introduced scratchpad reasoning, where the model generates intermediate carry computations explicitly; this trades inference cost for accuracy. Anil et al. [@anil2022] demonstrated that length generalization is qualitatively hard for vanilla transformers even when in-distribution accuracy is perfect.

**Positional encodings.** Vaswani et al. [@vaswani2017] introduced sinusoidal absolute PE; subsequent practice often replaced it with learned absolute PE for simplicity. Su et al. [@su2021] introduced rotary position embeddings (RoPE) that express position information as rotations in the attention dot product, with reportedly better extrapolation properties. Press et al. [@press2022] introduced ALiBi, which adds a static linear bias to attention scores based on position differences. Kazemnejad et al. [@kazemnejad2023] argued that, for decoder-only architectures, **no positional encoding at all** can extrapolate further than learned absolute PE, attributing this to the implicit position information carried by causal masking.

**Place-value embeddings.** McLeish et al. [@mcleish2024] introduced "Abacus" embeddings — a learned table indexed by a digit's distance from the ones place — and reported strong length-generalization gains for arithmetic. Our clean-Abacus / Abacus-curriculum comparison is designed to disentangle the parameterization from the data coverage in this result.

**Grokking and emergent arithmetic.** Power et al. [@power2022] showed that small transformers trained on modular arithmetic exhibit *grokking*: long after the training loss saturates, the model abruptly generalizes. Although our study uses non-modular addition and does not focus on grokking, the underlying optimization dynamics are relevant.

## 10. Conclusion

We present a small but tightly controlled comparison of six positional-encoding schemes on a 10M-parameter arithmetic transformer, and introduce an embedding-drift diagnostic that makes the dominant length-generalization failure mode directly measurable. We make qualitative predictions about each variant's behavior and provide a fully reproducible pipeline for testing them. The headline contribution is methodological: replacing reasoning about gradients with a number you can compute from the trained weights.

Future work includes (i) replicating across multiple seeds, (ii) extending to multiplication and modular operations, (iii) overlaying attention-pattern visualizations on failing examples, and (iv) probing whether NoPE's extrapolation advantage survives at larger scales.

## References

- [@anil2022] Anil, C. et al. (2022). *Exploring length generalization in large language models.* NeurIPS.
- [@kazemnejad2023] Kazemnejad, A. et al. (2023). *The Impact of Positional Encoding on Length Generalization in Transformers.* NeurIPS. arXiv:2305.19466.
- [@lee2023] Lee, N. et al. (2023). *Teaching Arithmetic to Small Transformers.* arXiv:2307.03381.
- [@mcleish2024] McLeish, S. et al. (2024). *Transformers Can Do Arithmetic with the Right Embeddings.* NeurIPS. arXiv:2405.17399.
- [@nogueira2021] Nogueira, R., Jiang, Z., Lin, J. (2021). *Investigating the Limitations of Transformers with Simple Arithmetic Tasks.* arXiv:2102.13019.
- [@nye2021] Nye, M. et al. (2021). *Show Your Work: Scratchpads for Intermediate Computation with Language Models.* arXiv:2112.00114.
- [@power2022] Power, A. et al. (2022). *Grokking: Generalization Beyond Overfitting on Small Algorithmic Datasets.* arXiv:2201.02177.
- [@press2022] Press, O., Smith, N., Lewis, M. (2022). *Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation.* ICLR. arXiv:2108.12409.
- [@su2021] Su, J. et al. (2021). *RoFormer: Enhanced Transformer with Rotary Position Embedding.* arXiv:2104.09864.
- [@vaswani2017] Vaswani, A. et al. (2017). *Attention Is All You Need.* NeurIPS.
