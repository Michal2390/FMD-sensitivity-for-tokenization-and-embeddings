"""Build final_results.ipynb - a step-by-step, executable tour of the study.

The notebook reads only the result CSVs and figures (no models, no GPU), so
anyone can run it in seconds. Regenerate with:
    python scripts/build_results_notebook.py
    jupyter nbconvert --to notebook --execute --inplace final_results.ipynb
"""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "final_results.ipynb"

nb = nbf.v4.new_notebook()
cells = []


def md(text: str) -> None:
    cells.append(nbf.v4.new_markdown_cell(text.strip()))


def code(src: str) -> None:
    cells.append(nbf.v4.new_code_cell(src.strip()))


# ── 0. Title ─────────────────────────────────────────────────────────────
md("""
# What Does the Frechet Music Distance Measure?
### A step-by-step tour of the results

**FMD sensitivity to tokenization and embedding configuration** - Warsaw
University of Technology, WIMU 2026 (M. Fereniec, B. Sedzikowski).

An FMD score is produced by a *pipeline* - input representation -> embedding
model -> Frechet formula - not by the music alone. This notebook walks through
every experiment of the study and reproduces each table directly from the
result CSVs in `results/reports/sensitivity_pivot/` (nothing is typed by
hand). It needs **no models and no GPU**; it only reads CSVs and figures.

**The two-level thesis**

1. *Attribute level:* the input representation decides which musical
   attributes FMD can perceive - velocity is detected only where it is
   encoded, and is **exactly invisible** under ABC notation.
2. *Corpus level:* style rankings are nevertheless robust across pipelines
   (Spearman rho 0.51-0.91) - "ranks styles sensibly" does **not** certify
   "sees the attribute you care about".

Full paper: [`draft.tex`](draft.tex) - methodology details:
[`docs/PAPER_FINDINGS.md`](docs/PAPER_FINDINGS.md).
""")

# ── 1. Setup ─────────────────────────────────────────────────────────────
md("""
## 1. Setup

Paths and display helpers. The five configurations and their properties:

| Config | Model | Input | Velocity channel? | L2-normalised? |
|:--|:--|:--|:--:|:--:|
| `MusicBERT-REMI` | MusicBERT | REMI tokens (MidiTok) | yes | no |
| `MusicBERT-TSD` | MusicBERT | TSD tokens (MidiTok) | yes | no |
| `CLaMP2-MTF` | CLaMP-2 | MIDI Text Format | yes | yes |
| `CLaMP1-ABC` | CLaMP-1 | ABC notation | **no** | yes |
| `CLaMP2-ABC` (control) | CLaMP-2 | ABC notation | **no** | yes |

`CLaMP2-ABC` is the *same-model control*: same model as `CLaMP2-MTF`, same
input as `CLaMP1-ABC` - it disentangles the representation effect from the
model effect. Datasets: MAESTRO, POP909 and four Lakh-derived genre subsets
(classical, jazz, rock, rap), 80 files each -> C(6,2) = 15 dataset pairs.
""")
code("""
from pathlib import Path

import pandas as pd
from IPython.display import Image, display

REPORTS = Path("results/reports/sensitivity_pivot")
FIGURES = Path("results/plots/sensitivity_pivot/paper")

CONFIGS = ["MusicBERT-REMI", "MusicBERT-TSD", "CLaMP2-MTF", "CLaMP1-ABC", "CLaMP2-ABC"]
PERTS = ["no_velocity", "quantized_time", "constant_tempo", "all_combined"]

pd.set_option("display.precision", 3)
pd.set_option("display.width", 160)
""")

# ── 2. Noise floors ──────────────────────────────────────────────────────
md("""
## 2. Why raw FMD must not be compared across models (noise floors)

Split-half FMD within one dataset should be ~0 for a stable pipeline. It is
the per-pipeline **noise floor**. MusicBERT embeddings are unnormalised
(norm ~10-20) while CLaMP embeddings live on the unit sphere, and the Frechet
formula scales with the squared embedding norm - so the floors differ by one
to two **orders of magnitude for purely geometric reasons**. This is why every
cross-configuration statement in the study uses scale-invariant statistics
(SNR, permutation/Wilcoxon tests, Spearman, CV) and never raw FMD magnitudes.
""")
code("""
ss = pd.read_csv(REPORTS / "self_similarity.csv")
ss.pivot_table(index="configuration", columns="dataset", values="split_half_fmd").reindex(CONFIGS)
""")
code('display(Image(filename=str(FIGURES / "fig1_noise_floor.png")))')

# ── 3. Perturbations ─────────────────────────────────────────────────────
md("""
## 3. Perturbation sensitivity (the main experiment)

We remove one expressive attribute at a time - velocity (dynamics -> 64),
microtiming (16th-note quantization), tempo (-> 120 BPM), and all combined -
and measure FMD(original, perturbed). Each effect is reported as
**SNR = FMD / noise floor** (dimensionless, comparable across pipelines) with
a two-sample **permutation test** (200 permutations).

`detected` below means SNR >= 1 **and** permutation p < 0.05.
""")
code("""
def snr_table(csv_name: str) -> pd.DataFrame:
    pert = pd.read_csv(REPORTS / csv_name)
    label = lambda r: f"{r.snr:.2f}" + (" *" if r.significant else "")
    out = pert.assign(cell=pert.apply(label, axis=1)).pivot_table(
        index="perturbation", columns="configuration", values="cell", aggfunc="first"
    )
    return out.reindex(PERTS)[CONFIGS]

print("MAESTRO (primary study)  -  * = detected")
snr_table("perturbation_sensitivity.csv")
""")
code("""
print("POP909 (replication)  -  * = detected")
snr_table("perturbation_sensitivity_pop909.csv")
""")
md("""
**What the tables show**

- **Velocity is the only attribute detected above the noise floor, and only
  where it is represented**: REMI (SNR 3.16 / 4.15), TSD (1.35 / 2.01) and
  MTF (1.18 / 1.70) on MAESTRO / POP909, all permutation-significant.
- Under both ABC pipelines the velocity-flattened corpus renders to
  **character-identical ABC**, so FMD = 0.0000 *exactly* - the blindness is
  structural, not statistical. The `CLaMP2-ABC` column is the causal control:
  same model as `CLaMP2-MTF`, only the representation differs.
- Microtiming and tempo stay below the detection floor everywhere; the
  *combined* perturbation simply tracks velocity.
- Built-in sanity check: POP909 is already grid-quantized, so quantization is
  a no-op there and the token pipelines report FMD = 0 - the protocol reports
  nothing where nothing changed.
""")
code('display(Image(filename=str(FIGURES / "fig2_snr_heatmap.png")))')
code('display(Image(filename=str(FIGURES / "fig4_perturbation_significance.png")))')

# ── 4. Per-file paired analysis ──────────────────────────────────────────
md("""
## 4. Per-file paired analysis (Wilcoxon + retest noise floor)

FMD is a distribution-level statistic estimated at n=80 << d=768. To show the
signal is not a distribution-fitting artifact, we also measure the **cosine
shift of every individual file** between its original and perturbed
embedding, and compare it against a **retest** condition - the same
unperturbed file encoded twice. The retest distance is the per-file
encoding-noise floor (zero for every pipeline: all conversions are
deterministic). Contrasts are one-sided Wilcoxon signed-rank tests with Holm
correction.

> The retest check earns its keep: an earlier iteration unknowingly embedded
> `music21` object reprs instead of ABC (music21 has no ABC export - its
> `write('abc')` silently emits the repr). Re-encodings of the same file came
> out near-orthogonal, and the retest caught it immediately. We replaced the
> conversion with a deterministic MIDI->ABC renderer. **Recommendation: build
> a retest into any embedding pipeline - it is nearly free.**
""")
code("""
tests = pd.concat([
    pd.read_csv(REPORTS / "paired_file_tests.csv"),
    pd.read_csv(REPORTS / "paired_file_tests_pop909.csv"),
])
velocity = tests[tests.perturbation == "no_velocity"][
    ["dataset", "contrast", "configuration", "n_files", "median_shift", "p_holm", "significant"]
]
velocity.sort_values(["contrast", "dataset", "configuration"]).reset_index(drop=True)
""")
md("""
**Reading:** velocity flattening moves the embedding of essentially *every
individual file* in the velocity-bearing pipelines (Holm-adjusted
p ~ 1e-13 on both corpora) and **exactly zero files** under ABC. The
same-model control contrast (`CLaMP2-MTF` vs `CLaMP2-ABC`, within one
embedding space) is significant at p ~ 1e-13 on both corpora.

Bonus dissociation: on MAESTRO, MTF quantization shifts are per-file
significant even though the corpus-level SNR stays below 1 - the file-level
view is *more* sensitive than FMD itself.
""")

# ── 5. Rankings ──────────────────────────────────────────────────────────
md("""
## 5. Corpus-level rankings are robust across pipelines

Each configuration ranks the 15 dataset pairs by FMD; Spearman rho between
configurations is scale-invariant, so it is a sound cross-pipeline
comparison. Despite radically different attribute sensitivity, **9 of 10
configuration pairs agree significantly** (rho 0.51-0.91) - the strongest
agreement is MTF vs ABC (rho = 0.907), two pipelines in incomparable spaces,
one of which cannot see dynamics at all. The only non-significant pair is
REMI vs TSD: the tokenizer perturbs the ranking more than swapping
model+representation does.
""")
code("""
sp = pd.read_csv(REPORTS / "spearman_ranking_agreement.csv")
sp[["config_a", "config_b", "spearman_tau", "p_value"]].sort_values(
    "spearman_tau", ascending=False
).reset_index(drop=True)
""")
code('display(Image(filename=str(FIGURES / "fig5_spearman_heatmap.png")))')
code('display(Image(filename=str(FIGURES / "fig6_cross_dataset_sep.png")))')

# ── 6. Bootstrap ─────────────────────────────────────────────────────────
md("""
## 6. Bootstrap stability

50 bootstrap resamples of the MAESTRO-POP909 distance. The coefficient of
variation (CV) is scale-invariant: all five pipelines are comparably stable
(8-20%), with the ABC pipelines the most stable. The mean column is **not**
comparable across the normalisation boundary.
""")
code("""
boot = pd.read_csv(REPORTS / "bootstrap_stability.csv")
boot.assign(cv_pct=(boot.cv * 100).round(1))[
    ["configuration", "fmd_mean", "fmd_ci_lower", "fmd_ci_upper", "cv_pct"]
].set_index("configuration").reindex(CONFIGS)
""")
code('display(Image(filename=str(FIGURES / "fig3_bootstrap_cv.png")))')

# ── 7. Conclusions ───────────────────────────────────────────────────────
md("""
## 7. Conclusions and practitioner guidance

| # | Finding | Evidence |
|:-:|:--|:--|
| 1 | Input representation sets a hard, **causal** ceiling on what FMD detects | velocity SNR 1.2-4.2 (p <= 0.02) in REMI/TSD/MTF vs **exactly 0** in ABC; same-model control switches it off; per-file p ~ 1e-13; replicated on POP909 |
| 2 | Corpus-level style rankings are robust across pipelines | 9/10 pairs significant, rho 0.51-0.91 (MTF-ABC 0.907) |
| 3 | The tokenizer alone reshapes sensitivity | REMI vs TSD velocity SNR 3.16 vs 1.35; REMI-TSD is the only n.s. ranking pair |
| 4 | Raw FMD is not cross-model comparable | noise floors differ by orders of magnitude from embedding geometry alone |

**If you evaluate music generation with FMD:**

- to judge *dynamics/expression*, use a velocity-bearing representation
  (REMI/TSD tokens, MTF) - never ABC;
- ABC-based FMD is fine for *score-like stylistic* comparison and ranks
  styles consistently with the other pipelines;
- never compare raw FMD numbers across embedding models - use SNR, rank
  agreement, or CV;
- treat microtiming/tempo effects with suspicion at n ~ 80;
- **audit your pipeline with a per-file retest** (encode one file twice) -
  it is nearly free and catches silent conversion failures.

**Reproduce everything**

```bash
python scripts/run_full_study.py            # full study (checkpointed)
python scripts/generate_draft_figures.py    # paper figures from CSVs
python scripts/generate_draft_tables.py     # paper tables from CSVs
pdflatex draft.tex                          # the paper (run twice)
```
""")

nb["cells"] = cells
nb["metadata"]["kernelspec"] = {
    "display_name": "Python 3",
    "language": "python",
    "name": "python3",
}
nbf.write(nb, OUT)
print(f"Wrote {OUT} ({len(cells)} cells)")
