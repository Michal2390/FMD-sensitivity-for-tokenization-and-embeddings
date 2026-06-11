# Paper findings & writing guide

A plain-language map from results → claims. All numbers come from
`results/reports/sensitivity_pivot/*.csv` (final run: 2026-06-11, 5 configs ×
6 datasets × 5 perturbations + per-file paired analysis on 2 corpora).
Tables/figures regenerate via `scripts/generate_draft_tables.py` and
`scripts/generate_draft_figures.py`; the full prose draft is `../draft.tex`.

---

## The one-sentence thesis

> An FMD score is a property of the *pipeline* (representation → model →
> Fréchet): the **input representation decides which musical attributes FMD
> can perceive** - even though corpus-level style rankings stay robust - and
> raw FMD is **not comparable across embedding models**.

---

## The two-level story (the paper's core)

**Attribute level - representation gates detectability, causally.**
Velocity flattening is detected by every velocity-bearing pipeline
(MAESTRO SNR: REMI 3.16, TSD 1.35, MTF 1.18; all perm. p ≤ 0.02; POP909
stronger: 4.15 / 2.01 / 1.70). Under ABC the perturbed corpus renders to
*character-identical* notation → **FMD = 0.0000 exactly** for both
CLaMP1-ABC and CLaMP2-ABC. The same-model control (CLaMP-2 fed MTF vs ABC)
turns the response off with weights and music held fixed. Microtiming and
tempo stay below the corpus-level detection floor everywhere (quantization
SNR ≤ 0.66; tempo ≤ 0.20). Per-file paired analysis (cosine shift vs a
re-encoding retest floor, Wilcoxon + Holm): velocity moves essentially
every file in REMI/TSD/MTF (p ≈ 1e-13, both corpora), exactly zero files in
ABC; control contrast p ≈ 1e-13. Bonus dissociation: MTF quantization is
per-file significant yet corpus-level invisible - the paired view is more
sensitive than FMD itself.

**Corpus level - rankings are robust across pipelines.**
Over 15 dataset pairs, 9 of 10 configuration pairs agree significantly
(ρ = 0.51-0.91); the strongest agreement is MTF ↔ CLaMP1-ABC **ρ = 0.907**
(p < 0.001) - two incomparable spaces, one of them blind to dynamics. The
only n.s. pair is REMI ↔ TSD (ρ = 0.44): the tokenizer perturbs ranking
more than swapping model+representation. Genre separation works in every
pipeline (separation ratios up to 4.8× the noise floor).
→ "Ranks styles sensibly" does **not** certify "sees the attribute you
care about"; only a perturbation audit distinguishes the two.

**Scale - raw FMD is not cross-model comparable.**
Noise floors: MusicBERT ~1.2-2.3 (unnormalised) vs CLaMP ~0.02-0.18
(L2-normalised) - geometry, not music. All cross-config claims use
scale-invariant statistics (SNR, permutation/Wilcoxon, Spearman, CV).
Bootstrap CV: 8-20% for all five configs (ABC pipelines most stable, 8%).

---

## The methodological war story (worth a paragraph in any talk)

music21 has **no ABC export**: `score.write('abc')` silently writes the
object repr (`<music21.stream.Score 0x...>`), so an earlier iteration
embedded 36-character repr strings instead of music. The per-file
**retest check** (encode the same unperturbed file twice) exposed it
immediately: re-encodings were near-orthogonal. Fix: a deterministic
dependency-free MIDI→ABC renderer (`src/embeddings/clamp_formats.py`).
Lesson for practitioners: build a retest into every embedding pipeline -
it is nearly free and catches silent conversion failures.

---

## How this answers the supervisor's critique

| Critique (old `main`) | Resolution |
|:--|:--|
| "CLaMP-2 + REMI" mislabelled (actually ABC) | REMI/TSD → MusicBERT; CLaMP gets native MTF/ABC; CLaMP2-ABC kept as a *deliberate* same-model control |
| Velocity result explained by ABC having no velocity | Now exact and causal: ABC rendering is character-identical under velocity flattening → FMD = 0 |
| "Statistics on three values - indefensible" | 15 pairs (Spearman with p), permutation tests, per-file Wilcoxon (n=80) with Holm correction |
| Hard-coded values | every table/figure generated from CSVs |

---

## Manuscript outline (matches draft.tex)

1. Introduction - FMD is pipeline-dependent; 4 contributions.
2. Related work - FID/FAD/FMD; MusicBERT; CLaMP-1/2; MidiTok.
3. Background - Fréchet formula and why it is not scale-free.
4. Method - 5 configs (incl. control), 6 datasets, perturbations; SNR;
   permutation test; per-file paired analysis with retest floor; Spearman.
5. Results - noise floors → perturbations (+control, +replication) →
   per-file analysis → rankings → bootstrap.
6. Discussion - two-level picture; negative result (no raw cross-model
   FMD); practitioner guidance incl. the retest audit.
7. Limitations - MusicBERT token-string rendering; n ≪ d; ABC sixteenth
   grid; one perturbation strength; Lakh provenance.
8. Conclusion.

## Possible extensions (if time permits)

- Dose-response perturbation sweep (velocity 25/50/75/100%, grid 4/8/16/32)
  → detection thresholds and monotonicity.
- Sample-size sweep (n = 20-160) → minimum-n guidance for FMD.
- Canonical OctupleMIDI input for MusicBERT.
