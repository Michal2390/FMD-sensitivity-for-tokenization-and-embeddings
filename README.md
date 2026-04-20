# FMD Sensitivity for Tokenization and Embeddings

**Advanced Music Information Retrieval Research Project**

> A systematic study of how Frechet Music Distance (FMD) responds to tokenization choices and embedding models for symbolic music.

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-red)](https://pytorch.org/)
[![Tests: 53/53](https://img.shields.io/badge/Tests-53%2F53%20✓-success)](tests/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)]()

## Quick Overview

This project implements a full experimental pipeline to analyze the impact of:
- **4 tokenization strategies**: REMI, TSD, Octuple, MIDI-Like
- **6 embedding models**: CLaMP-1, CLaMP-2, MusicBERT, MusicBERT-large, MERT, NLP-Baseline
- **4 preprocessing configs**: original, no velocity, hard quantization, combined

on **Frechet Music Distance (FMD)**, a metric for symbolic music similarity.

We evaluated **5760 FMD observations** across **6 genre pairs** (rock, jazz, electronic, country) from the **Lakh MIDI Dataset**, with **96 pipeline variants** and 10× repeated subsampling per variant. Results are validated with **bootstrap confidence intervals**, **Holm–Bonferroni correction**, **permutation tests**, and **cross-dataset replication** on MidiCaps.

## Key Findings (6-Model Analysis)

### 🔥 Embedding Model Choice Dominates FMD Variance (η² = 0.96)

With six embedding models spanning three architecture families (contrastive, masked LM, statistical), the **model choice alone explains 96% of FMD variance** — far exceeding all other factors combined.

| Source | η² | F | Interpretation |
|--------|-----|---|----------------|
| **Model (main)** | **0.9617** | **33904** | **Dominant** |
| Tokenizer × Model | 0.0034 | 40.40 | Negligible |
| Model × Preprocess | 0.0022 | 25.65 | Negligible |
| **Tokenizer (main)** | **0.0010** | 59.51 | **Negligible** |
| **Preprocessing** | **0.0010** | 60.14 | **Negligible** |

> All effects are statistically significant (p < 0.001) due to N=5760, but only the model effect has practically meaningful magnitude.

### 🔑 Key Insight: Tokenizer Sensitivity Is Model-Dependent, Not Universal

The most important finding from adding CLaMP-1 and CLaMP-2:

| Analysis | Models | η²(tokenizer) | η²(model) | Conclusion |
|----------|--------|---------------|-----------|------------|
| 2-model (CLaMP only) | CLaMP-1, CLaMP-2 | 0.101 | 0.087 | Tokenizer matters |
| 3-model | + MusicBERT | 0.020 | 0.778 | Model dominates |
| **6-model** | **+ MusicBERT-large, MERT, NLP** | **0.001** | **0.962** | **Tokenizer irrelevant** |

> **FMD sensitivity to tokenization is a property of specific models, not an inherent property of the metric.** CLaMP models (contrastive, operating on raw MIDI) are insensitive to tokenization choices, which dilutes the tokenizer effect when pooled with token-dependent models.

### 📊 Model Hierarchy by Genre Sensitivity

Cohen's d reveals a clear hierarchy of how sensitive each model is to genre differences:

| Model | Architecture | Genre Sensitivity | Cohen's d vs CLaMP-2 |
|-------|-------------|-------------------|---------------------|
| MusicBERT-large | MLM (large) | Highest | −10.02 |
| MusicBERT | MLM | High | −6.27 |
| NLP-Baseline | Sentence encoder | Medium | −1.21 |
| CLaMP-1 | Contrastive | Low | −1.55 |
| CLaMP-2 | Contrastive | Lowest | (reference) |
| MERT | Audio SSL | ⚠️ Anomalous | 0.00 (likely defective) |

> **Contrastive models (CLaMP)** produce more "unified" representations → lower FMD between genres.
> **MLM models (MusicBERT)** produce more discriminative representations → higher FMD.
> **MERT** returns constant/zero embeddings for MIDI input — requires audio, not symbolic data.

### ⚠️ Critical Insight: FMD Is Not Scale-Invariant Across Model Architectures

MusicBERT produces embeddings with **3× higher norms** than CLaMP-2 (~15 vs ~2.7), yielding systematically higher FMD values. This means:

> **Raw FMD values cannot be directly compared across embedding models of different architectures.** Normalization or model-specific baselines are required for fair comparison.

### 🔬 Generalizability Across Genre Pairs

The η² decomposition is remarkably consistent across all genre pairs (CV = 0.00 for model):

| Genre Pair | η²(model) | η²(tokenizer) | η²(preprocess) |
|------------|-----------|---------------|----------------|
| jazz ↔ country | 0.9849 | 0.0019 | 0.0009 |
| rock ↔ country | 0.9876 | 0.0007 | 0.0014 |
| rock ↔ jazz | 0.9881 | 0.0012 | 0.0013 |

### 📊 FMD Ranges by Genre Pair (6-model)

| Genre Pair | Mean FMD | Std | N |
|------------|----------|-----|---|
| jazz ↔ country | 4.163 | 4.757 | 960 |
| rock ↔ jazz | 4.751 | 5.406 | 960 |
| rock ↔ electronic | 4.759 | 6.240 | 960 |
| rock ↔ country | 4.810 | 5.528 | 960 |
| jazz ↔ electronic | 5.396 | 7.029 | 960 |
| electronic ↔ country | 5.717 | 7.375 | 960 |

### 🔒 Statistical Robustness

All main effects confirmed with multiple approaches:
- **Three-way factorial ANOVA** with interactions (N = 5760, 96 variants)
- **Bootstrap 95% CI for η²** (5000 resamples, percentile method)
- **Holm–Bonferroni correction** for multiple comparisons — all key effects survive
- **Permutation tests** (p = 0.001 for all three factors)
- **Tukey HSD** post-hoc with corrected p-values (`p_adj_holm`, `p_adj_bonf`)
- **Cross-dataset validation** (Spearman ρ = 0.975 between Lakh CD2 and MidiCaps — 3-model, 5760 observations)

### 🎯 Practical Pipeline Recommendations

| Goal | Recommended Pipeline | Rationale |
|------|---------------------|-----------|
| Finest resolution (low baseline) | REMI + CLaMP-2 | FMD ~0.05–0.10, highest eff. dim, best genre separation |
| Maximum genre separability | Any tokenizer + MusicBERT-large | Highest absolute FMD, most discriminative |
| Stable across genres | Octuple + CLaMP-1 | Lowest CV across pairs |
| ⚠️ Avoid | MIDI-Like + MusicBERT | Lowest eff. dim (11), inflated FMD |
| ⚠️ Avoid | MERT for symbolic MIDI | Returns degenerate embeddings for non-audio input |
| ⚠️ Caution | Comparing FMD across model architectures | Scale-dependent — normalize first |

---

<details>
<summary><b>📁 Previous Results (4-Model: MusicBERT, MusicBERT-large, MERT, NLP-Baseline)</b></summary>

### Embedding Model Dominates (η² = 0.94)

With 4 models (no CLaMP): model choice explained 94% of FMD variance.

| Source | η² | F | Interpretation |
|--------|-----|---|----------------|
| **Model (main)** | **0.939** | **22459** | **Dominant** |
| Tokenizer × Model | 0.004 | 34.50 | Negligible |
| Tokenizer (main) | 0.003 | 69.92 | Negligible |
| Preprocessing | 0.003 | 59.54 | Negligible |

### What Changed with 6 Models?

| Metric | 4 Models | 6 Models | Change |
|--------|----------|----------|--------|
| η²(model) | 0.939 | **0.962** | ↑ CLaMP widens spread |
| η²(tokenizer) | 0.003 | 0.001 | ↓ CLaMP ignores tokens |
| η²(tok×model) | 0.004 | 0.003 | ≈ stable |
| Total observations | 3840 | **5760** | +50% |

> Adding CLaMP-1 and CLaMP-2 (contrastive models that bypass tokenization) proved that tokenizer sensitivity is **model-specific, not universal**.

</details>

<details>
<summary><b>📁 Previous Results (2-Model Baseline: CLaMP-1 + CLaMP-2)</b></summary>

### Tokenizer × Model Interaction Dominates (η² = 0.42)

With 2 models only, the **interaction between tokenizer and model** was the single largest source of FMD variance — not either factor alone.

| Source | η² | Partial η² | Interpretation |
|--------|-----|-----------|----------------|
| **Tokenizer × Model** | **0.421** | **0.541** | Dominant interaction |
| Tokenizer (main) | 0.101 | 0.220 | Medium effect |
| Model (main) | 0.087 | 0.195 | Medium effect |
| Genre pair | 0.215 | — | Large (inherent) |
| Preprocessing | 0.020 | 0.054 | Small effect |

### Effect Sizes (Cohen's d, 2-model)

| Comparison | d | Magnitude |
|------------|---|-----------|
| Octuple vs REMI | **0.895** | **Large** |
| MIDI-Like vs REMI | 0.691 | Medium |
| CLaMP-1 vs CLaMP-2 | 0.615 | Medium |
| REMI vs TSD | −0.558 | Medium |
| MIDI-Like vs TSD | 0.016 | Negligible |

### FMD Ranges by Genre Pair (2-model)

| Genre Pair | Mean FMD | Std |
|------------|----------|-----|
| jazz ↔ country | 0.157 | 0.063 |
| rock ↔ electronic | 0.177 | 0.063 |
| rock ↔ jazz | 0.177 | 0.069 |
| rock ↔ country | 0.189 | 0.069 |
| jazz ↔ electronic | 0.241 | 0.081 |
| electronic ↔ country | 0.268 | 0.100 |

### What Changed with 3 Models?

| Metric | 2 Models | 3 Models | Change |
|--------|----------|----------|--------|
| η²(model) | 0.087 | **0.778** | ↑ 9× — MusicBERT on a different scale |
| η²(tokenizer) | 0.101 | 0.020 | ↓ diluted by model dominance |
| η²(tok×model) | 0.421 | 0.051 | ↓ model main effect absorbs variance |
| η²(preprocess) | 0.020 | 0.002 | ↓ negligible in both |
| Cross-dataset ρ | 0.921 (2-model) | **0.975** (3-model) | ↑ stronger with MusicBERT |
| Total observations | 1920 | 2880 | +50% (48 variants vs 32) |

> Adding MusicBERT revealed that **FMD is fundamentally scale-sensitive** — the dominant "interaction" in the 2-model analysis was partially a proxy for architectural differences that become explicit with a third, differently-scaled model.

</details>

---

## New: Normalized FMD (nFMD) — Scale-Invariant Metric

We propose **Normalized FMD (nFMD)** to address the key finding that raw FMD is not comparable across embedding models of different architectures. Three normalization strategies are implemented in `src/metrics/fmd.py`:

| Method | Formula | Intuition |
|--------|---------|-----------|
| **Trace** (default) | nFMD = FMD / (Tr(Σ₁) + Tr(Σ₂)) | Normalizes by total embedding variance |
| **Norm** | nFMD = FMD / (‖μ₁‖ + ‖μ₂‖)² | Compensates for quadratic mean-norm scaling |
| **Z-score** | nFMD = (FMD − μ_baseline) / σ_baseline | Calibrates against within-genre baseline |

### Validation: Normalization Eliminates Scale Artefact

| Model | Raw FMD (split test) | nFMD_trace | nFMD_norm | Mean ‖emb‖ |
|-------|---------------------|------------|-----------|------------|
| CLaMP-1 | 0.0172 | 0.001409 | 0.000081 | 7.71 |
| CLaMP-2 | 0.0115 | 0.002683 | 0.000549 | 2.72 |
| MusicBERT | 0.1477 | 0.001662 | 0.000209 | 14.84 |

> **Raw FMD varies 12.8× across models; after trace-normalization, only 1.9×.**
> This confirms that the dominant η²(model) = 0.78 is largely a scale artefact, and nFMD enables fair cross-model comparison.

## Embedding Models (6 Models, 3 Architecture Families)

The pipeline uses **6 embedding models** spanning three distinct architecture families:

| Model | Type | Architecture | Embedding Dim | Domain |
|-------|------|-------------|---------------|--------|
| **CLaMP-1** | Contrastive | CLIP-style (MIDI→text) | 768 | Symbolic |
| **CLaMP-2** | Contrastive | CLIP-style (MIDI→text) | 768 | Symbolic |
| MusicBERT | Masked LM | BERT (symbolic tokens) | 768 | Symbolic |
| MusicBERT-large | Masked LM | BERT-large (symbolic tokens) | 1024 | Symbolic |
| **MERT** | Self-supervised | Wav2Vec2-style (audio) | 768 | **Audio** |
| **NLP-Baseline** | Sentence encoder | MPNet (general text) | 768 | **Text (control)** |

With 6 models: **96 pipeline variants** × 6 genre pairs × 10 repeats = **5760 FMD observations**.

The inclusion of contrastive (CLaMP), masked LM (MusicBERT), audio SSL (MERT), and text (NLP-Baseline) models enables testing whether FMD sensitivity patterns are architecture-dependent.

## New: Sample-Size Sensitivity (Power Analysis)

We investigated how η² estimates stabilize as a function of per-cell sample size (number of repeated subsamplings per variant×pair combination).

### Key Finding: η² Is Remarkably Stable Even at Small Sample Sizes

| Repeats/cell | η²(model) | η²(tokenizer) | η²(preprocess) | Power(model) | Power(tokenizer) | Power(preprocess) |
|-------------|-----------|---------------|----------------|-------------|-----------------|-------------------|
| 2 | 0.780 ± 0.007 | 0.020 ± 0.001 | 0.002 ± 0.000 | 100% | 100% | 0% |
| 5 | 0.779 ± 0.002 | 0.020 ± 0.000 | 0.002 ± 0.000 | 100% | 100% | 0% |
| 10 | 0.778 ± 0.000 | 0.020 ± 0.000 | 0.002 ± 0.000 | 100% | 100% | 0% |

> **Model and tokenizer effects achieve 100% power even with just 2 repeats per cell.**
> Preprocessing effect (η² = 0.002) is genuinely negligible — it never reaches significance regardless of sample size.
> This validates our experimental design: 10 repeats provide more than adequate statistical power.

Generated plots: `sample_size_stability.{png,pdf}`, `sample_size_power.{png,pdf}`, `sample_size_combined.{png,pdf}`.

---

## Experimental Design

```
                    ┌─────────────────────────────────┐
                    │     Lakh MIDI Dataset            │
                    │  rock(119) jazz(119) electr(118) │
                    │         country(119)             │
                    └────────────┬────────────────────┘
                                 │
                    ┌────────────▼────────────────────┐
                    │      Preprocessing (×4)          │
                    │  original | no_vel | quant | both│
                    └────────────┬────────────────────┘
                                 │
                    ┌────────────▼────────────────────┐
                    │      Tokenization (×4)           │
                    │  REMI | TSD | Octuple | MIDI-Like│
                    └────────────┬────────────────────┘
                                 │
                    ┌────────────▼────────────────────┐
                    │    Embedding Model (×6)           │
                    │  CLaMP-1 | CLaMP-2 | MusicBERT   │
                    │  MusicBERT-large | MERT | NLP    │
                    └────────────┬────────────────────┘
                                 │
                    ┌────────────▼────────────────────┐
                    │   FMD Computation (×6 pairs)     │
                    │   10× repeated subsampling       │
                    │   = 5760 total observations      │
                    └──────────────────────────────────┘
```

## Project Structure

```
├── src/                           # Source code
│   ├── preprocessing/             # MIDI preprocessing
│   ├── tokenization/              # 4 tokenizers (MidiTok)
│   ├── embeddings/                # CLaMP-1/2, MusicBERT, MERT, NLP extraction + cache
│   ├── metrics/                   # FMD (Frechet Music Distance)
│   ├── experiments/               # Analysis, bootstrap CI & publication plots
│   └── utils/                     # Config and helpers
│
├── scripts/                       # Experiment runners, demos, data tools
├── tests/                         # 53+ unit tests
├── configs/                       # YAML configuration
├── data/                          # Lakh MIDI subsets
├── results/
│   ├── reports/lakh/              # Single-pair analysis (32 variants)
│   ├── reports/lakh_multi/        # Multi-genre analysis (5760 obs, 6 models)
│   └── plots/paper/               # Publication-ready figures
├── docs/                          # Documentation & weekly summaries
├── scripts/                       # Data generation & plotting
└── main.py                        # Entry point (multiple modes)
```

## Quick Start

### Installation
```bash
pip install -r requirements.txt
pip install -e .
```

### Run modes
```bash
python main.py --mode quick          # Sanity demo + quick benchmark
python main.py --mode paper          # Full paper pipeline
python main.py --mode paper-full     # Extended with special pairs
python main.py --mode paper-plots    # Generate publication plots
python main.py --mode fetch-data     # Download Lakh MIDI subsets
python main.py --mode tests          # Run test suite
```

### Multi-Genre Analysis
```bash
python scripts/run_multi_genre_analysis.py   # 4 genres × 96 variants × 10 repeats (6 models)
```

### Sample-Size Sensitivity (Power Analysis)
```bash
python scripts/run_sample_size_ablation.py   # η² stability & power curves
```
Produces ANOVA tables, interaction plots, η² heatmaps, and violin plots in `results/reports/lakh_multi/` and `results/plots/paper/`.

### Cross-Dataset Validation (3-Model)
```bash
python scripts/run_cross_dataset_validation.py --source midicaps # MidiCaps (3-model, ~95 min)
python scripts/run_cross_dataset_validation.py                   # All sources (CD1 + MidiCaps)
python scripts/run_cross_dataset_validation.py --source cd1      # Tagtraum CD1 only
python main.py --mode cross-validate                     # Via main entry point
python main.py --mode cross-validate --cv-source cd1     # Specific source
```
Validates generalizability by repeating the sensitivity analysis on independent data sources:
- **Lakh + Tagtraum CD1** — same MIDI files, different genre annotator (cross-annotation)
- **MidiCaps** — independent MIDI dataset with genre tags (cross-dataset)

**3-model results (CLaMP-1, CLaMP-2, MusicBERT):**
- Spearman ρ = **0.975** (Lakh CD2 vs MidiCaps) — near-perfect ranking agreement
- η²(model) = 0.778 (Lakh) vs 0.805 (MidiCaps) — consistent dominance
- 5760 total observations across both sources

Produces η² comparison, ranking agreement (Spearman ρ), tokenizer×model heatmaps, and a comprehensive report in `results/reports/cross_validation/`.

### Starter Data
```bash
python scripts/generate_starter_midis.py --count-per-dataset 120 --bars 16
```

### Run Tests
```bash
pytest tests/ -v
pytest tests/test_paper_pipeline.py -v
```

### Run Demos
```bash
python scripts/demo_embeddings.py --demo all
python scripts/demo_fmd.py --demo basic
python scripts/run_ablation_study.py
```

## Weeks Overview

| Week | Topic | Status | Tests |
|------|-------|--------|-------|
| 1 | Initialization & config | ✅ | — |
| 2 | Preprocessing & tokenization | ✅ | 47 |
| 3 | CLaMP embeddings & cache | ✅ | 29 |
| 4 | FMD metric & single-pair analysis | ✅ | 24 |
| 5 | Ablation study & sensitivity | ✅ | 13 |
| 6 | Multi-genre analysis & ANOVA | ✅ | — |
| 7 | Cross-dataset validation (CD1 + MidiCaps) | ✅ | 9 |
| 8 | MusicBERT, bootstrap CI, interaction mechanism, multiple comparison correction | ✅ | — |
| 9 | Normalized FMD (nFMD), MIDI-BERT + MERT models, sample-size power analysis | ✅ | — |
| 10 | **6-model analysis: CLaMP-1, CLaMP-2 added — tokenizer sensitivity is model-dependent** | ✅ | — |

**👉 [See detailed summaries →](docs/weekly_summaries/)**

## Generated Outputs

### Reports

| File | Description |
|------|-------------|
| [`ANALYSIS_REPORT.md`](results/reports/lakh/ANALYSIS_REPORT.md) | Single-pair (rock vs jazz) full analysis |
| [`MULTI_GENRE_REPORT.md`](results/reports/lakh_multi/MULTI_GENRE_REPORT.md) | Multi-genre 6-model analysis with 3-way ANOVA |
| [`INTERACTION_MECHANISM_REPORT.md`](results/reports/lakh_multi/INTERACTION_MECHANISM_REPORT.md) | Tok×Model interaction mechanism (PCA, t-SNE, norms) |
| [`CROSS_VALIDATION_REPORT.md`](results/reports/cross_validation/CROSS_VALIDATION_REPORT.md) | Cross-dataset generalizability (3-model, ρ=0.975) |
| `multi_genre_fmd.csv` | Raw 5760 FMD observations |
| `eta_sq_per_pair.csv` | η² consistency across pairs |
| `interaction_cell_stats.csv` | Per-cell embedding diagnostics |
| `cross_dataset_fmd.csv` | Combined FMD from all sources |
| `sample_size_ablation.csv` | η² and power at varying sample sizes |
| `sample_size_summary.csv` | Aggregated sample-size sensitivity |

### Publication Plots

| Plot | Description |
|------|-------------|
| `multi_summary_4panel` | Overview: violin + heatmap + η² + interaction |
| `multi_fmd_violin_tok_model` | FMD distributions by tokenizer and model |
| `multi_eta_sq_bootstrap_ci` | **Forest plot: η² with 95% bootstrap CI** |
| `multi_eta_sq_heatmap` | η² stability across genre pairs |
| `multi_eta_sq_comparison` | Factor importance comparison |
| `multi_fmd_by_pair` | FMD distributions per genre pair |
| `multi_interaction_per_pair` | Tokenizer×model interaction per pair |
| `interaction_pca_curves` | **PCA explained variance per tok×model cell** |
| `interaction_token_length` | **Token sequence length distributions** |
| `interaction_eff_dim_heatmap` | **Effective dimensionality heatmap** |
| `interaction_tsne_best_worst` | **t-SNE: best vs worst tok×model cell** |
| `cross_summary_4panel` | Cross-dataset validation overview |
| `cross_eta_sq_comparison` | η² comparison across data sources |
| `cross_tok_model_heatmaps` | Tokenizer×model means per source |
| `cross_ranking_agreement` | Pipeline ranking Spearman ρ |
| `cross_fmd_violin_by_source` | FMD distributions per source |
| `sample_size_stability` | **η² stability vs sample size (CI ribbon)** |
| `sample_size_power` | **Statistical power curves per factor** |
| `sample_size_combined` | **Combined stability + power 2-panel figure** |

## Documentation

- **[Weekly Summaries](docs/weekly_summaries/)** — Detailed progress reports
- **[How to Run](docs/INSTRUKCJA_URUCHAMIANIA.md)** — Execution guide
- **[Design Proposal](docs/references/DESIGN_PROPOSAL.md)** — Original research plan

## References

1. Retkowski, J., Stępniak, J., Modrzejewski, M. (2025). *Frechet Music Distance: A Metric For Generative Symbolic Music Evaluation.*
2. Fradet, N., et al. (2024). *MidiTok: A Python Package for MIDI File Tokenization.*
3. Wu, Y., et al. (2023). *CLaMP: Contrastive Language-Music Pre-training.*

## Academic Context

**Project**: Frechet Music Distance Sensitivity  
**Institution**: Warsaw University of Technology, EITI  
**Course**: WIMU (Music Information Retrieval)  
**Duration**: February–June 2026

---

**Status**: ✅ Complete (Weeks 1–10) | **Last Updated**: 20.04.2026
