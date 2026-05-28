# Whitepaper finalization checklist

What needs to happen between the current draft (`whitepaper.md`) and a version
you would be comfortable sending to a recruiter, posting on a personal site, or
submitting to arXiv. Items are grouped by whether they block public sharing or
are polish.

**Status as of v0.2:** items #1, #2, #3, #5 are done (sweep ran, real numbers
in §6, predictions reconciled with reality in §7, figures embedded, voice
changed to first person). The remaining must-have is #4 (author block + abstract
polish, partially done). Everything else is polish or production.

## Must-have (blocks public sharing)

### 1. Run the full sweep and slot in the real numbers

The whole Results section (§6) is currently `[pending]`. To make this honest:

- Run `make figures` on a GPU box (Colab T4 is the easiest path; ~25–40 min for
  the full six-variant sweep). Verify `results/sweep.json` is populated.
- Replace the prediction table in §6.1 with the actual exact-match accuracies
  at digit counts 1–6 for each variant.
- Capture the per-digit-position pattern in §6.2 with one or two specific
  observations (e.g. "for NoPE at 5-digit, the ones-place is 92% accurate but
  the most-significant digit is 41%").
- Capture the actual drift numbers in §6.3, including the boundary indices
  where drift transitions from non-zero to exactly zero.

**Effort**: ~1 hour (mostly waiting on Colab). The text edits are mechanical
once the JSON exists.

### 2. Reconcile predictions with reality in the Discussion

If any prediction in §6 turns out wrong, §7 (Discussion) needs corresponding
edits. Possibilities to watch for:

- **NoPE underperforms**. At ~10M scale NoPE has been mixed in prior reports.
  If `nope` cliffs as hard as `baseline`, the framing in §7 needs to soften
  ("position information has to come from *somewhere*; the causal mask alone
  is insufficient at this scale").
- **Abacus clean beats expectation**. If `abacus` (without curriculum)
  generalizes meaningfully at 4 digits, that contradicts the drift diagnostic
  and the paper needs to investigate why — likely because position 4 (carry)
  *does* receive gradient at 3-digit training.
- **RoPE doesn't extrapolate**. If `rope` cliffs at 4 digits, the standard
  rotary formulation may need a base-frequency adjustment for short-sequence
  arithmetic; cite Liu et al. (2024, "ScalingRoPE") if so.

**Effort**: ~30 min if predictions broadly hold; 2–4 hours if a major
prediction misses and the framing needs restructuring.

### 3. Embed the figures

Once `results/*.png` are committed:

- Insert each figure after its introducing paragraph in §6 with markdown
  image syntax: `![Caption](../results/length_gen.png)`. The current text
  already references them by name, so insertion is mechanical.
- Write a one-sentence caption for each.

**Effort**: 15 min.

### 4. Author block and abstract polish

The current draft has no author line, no affiliation, no date. Add:

- Title (consider tightening the current question-form title).
- Author name and contact (GitHub link, email, or both).
- A "version" or date marker (e.g. "v0.1 — May 2026").
- Optional: a one-line acknowledgement of any collaborators or tools.

The abstract is dense (single 200-word paragraph). Consider splitting into
*setup — method — finding* sentences for easier skimming.

**Effort**: 15 min.

## Should-have (polish)

### 5. Decide on voice and rewrite if needed

The draft uses academic "we" voice. For a solo project posted to GitHub, a
direct first-person voice or technical-report passive often reads better.
Pick one and apply consistently. The decision is stylistic; the content is
unaffected.

**Effort**: 30 min if changing voice; 0 if keeping.

### 6. Tighten Related Work

§9 currently lists ten papers with one-sentence descriptions. Consider:

- Group them into 2–3 thematic clusters (positional encodings, arithmetic
  transformers, length generalization broadly) with topic sentences.
- Drop any paper that doesn't *directly* contrast with your method. Power et
  al. (grokking) is borderline — included for completeness but could be cut.
- Verify each citation by skimming the paper's abstract. The current list was
  composed from memory; check authors, years, and venues.

**Effort**: 1–2 hours.

### 7. Add a code listing for the digit-position algorithm

The Abacus place-value assignment (§3.4) is the most subtle algorithmic
contribution after the drift diagnostic. A 15-line Python listing in the body
would make the description concrete without bloating the page.

**Effort**: 20 min.

### 8. Add a small architecture diagram

A simple block diagram of the transformer with annotations showing where each
of the four positional encodings enters the model would clarify §3 at a
glance. Could be done in:

- `tikz` (LaTeX) — best quality, more setup
- `excalidraw.com` — fastest, exports as PNG
- `mermaid` (lives in the markdown) — no separate file, simplest

**Effort**: 30 min in Excalidraw; 1–2 hours in TikZ.

### 9. Multiple-seeds run

The current sweep uses a single random seed per variant. To strengthen the
near-cliff numbers, re-run with seeds 0, 1, 2 and report mean ± std at each
(variant, digit count) cell. Important for any cell where accuracy is between
20% and 80%; less so for cells at 0% or 100%.

**Effort**: 3× the sweep time (~90 min on Colab T4). Plotting code needs
small extension for error bars.

## Production decisions

### 10. Pick a citation format

The draft uses `[@key]` inline pointing at a human-readable list at the bottom.
Options for the final form:

- **Keep markdown-native** with the bracketed-key syntax. Easiest. Renders
  fine on GitHub. Loses link-to-bibliography functionality.
- **Pandoc + BibTeX**. Add `paper/refs.bib`, render via
  `pandoc --citeproc whitepaper.md --bibliography refs.bib -o whitepaper.pdf`.
  Industry-standard, supports auto-numbering and proper bibliography.
- **Hyperlinked DOIs**. Each citation links directly to the paper's arXiv or
  conference URL. Lowest friction for readers; no central bibliography.

Recommendation: **pandoc + BibTeX** if you intend a PDF; **hyperlinked DOIs**
if you intend to host only on GitHub.

**Effort**: 1–2 hours to convert.

### 11. Decide on output format and distribution

Where will this live?

- **GitHub `paper/whitepaper.md`** — already there. Lowest-friction. Renders
  in the GitHub UI. No PDF.
- **PDF on personal site / repo release**. Add a release asset or commit the
  PDF alongside the markdown. Need to pick a renderer (pandoc, LaTeX, Typst).
- **arXiv preprint**. Requires LaTeX with a specific bibliography style.
  Higher bar but gives the work a citable identity. Worth considering only
  after multi-seed runs (item #9) and a careful related-work pass (#6).
- **Blog post or Substack**. Best for general audience; would likely cut
  some of the formal scaffolding (limitations section, related work) and
  expand the discussion.

Recommendation order if you're optimizing for **recruiter appeal**:
GitHub markdown → PDF (linked from README) → blog post (linked from PDF).
arXiv only if the result is novel enough to warrant a real submission, which
is unlikely unless multi-seed numbers show something genuinely new.

**Effort**: 2–4 hours total for the GitHub + PDF combination.

### 12. README integration

Once the whitepaper has results in it:

- Add a "Whitepaper" section to the project README linking to
  `paper/whitepaper.md` (or the rendered PDF).
- Consider moving the bulk of the "Length generalization" section out of the
  README and into the whitepaper, leaving only a one-paragraph teaser + the
  headline figure in the README. Keeps the README scannable and avoids
  duplication.

**Effort**: 20 min.

## Suggested order

1. (#1) Run the sweep, slot in numbers. Block on Colab.
2. (#2) Reconcile predictions with results — if surprises, restructure §7.
3. (#3) Embed figures.
4. (#4) Author block + abstract polish.
5. (#5) Voice decision.
6. (#10) Citation format → decides whether to do #11 as PDF or markdown.
7. (#7) Code listing.
8. (#11) Render to chosen format.
9. (#12) Update README.

Items #6, #8, #9 are independent polish that can happen in parallel or be
deferred.

**Critical path to "shareable v1.0"**: items #1, #2, #3, #4, #12 (~3 hours
of focused work after the sweep completes).

**Critical path to "PDF v1.0"**: add items #5, #6 or #7, #10, #11 (~6 hours
of focused work).
