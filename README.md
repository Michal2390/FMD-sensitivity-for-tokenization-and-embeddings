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
- **2 embedding models**: CLaMP-1, CLaMP-2
- **4 preprocessing configs**: original, no velocity, hard quantization, combined

on **Frechet Music Distance (FMD)**, a metric for symbolic music similarity.

We evaluated **1920 FMD observations** across **6 genre pairs** (rock, jazz, electronic, country) from the **Lakh MIDI Dataset**, with 10× repeated subsampling per variant.

## Key Findings

### 🔥 Tokenizer × Model Interaction Dominates (η² = 0.42)

The single largest source of FMD variance is the **interaction between tokenizer and model choice** — not either factor alone. This means the optimal tokenizer depends on which embedding model you use.

| Source | η² | Partial η² | Interpretation |
|--------|-----|-----------|----------------|
| **Tokenizer × Model** | **0.421** | **0.541** | Dominant interaction |
| Tokenizer (main) | 0.101 | 0.220 | Medium effect |
| Model (main) | 0.087 | 0.195 | Medium effect |
| Genre pair | 0.215 | — | Large (inherent) |
| Preprocessing | 0.020 | 0.054 | Small effect |

### 📊 FMD Ranges by Genre Pair

| Genre Pair | Mean FMD | Std |
|------------|----------|-----|
| jazz ↔ country | 0.157 | 0.063 |
| rock ↔ electronic | 0.177 | 0.063 |
| rock ↔ jazz | 0.177 | 0.069 |
| rock ↔ country | 0.189 | 0.069 |
| jazz ↔ electronic | 0.241 | 0.081 |
| electronic ↔ country | 0.268 | 0.100 |

### 🎯 Practical Pipeline Recommendations

| Goal | Recommended Pipeline | Rationale |
|------|---------------------|-----------|
| Maximum genre separability | REMI + CLaMP-1 | Highest absolute FMD values |
| Finest resolution (low baseline) | REMI + CLaMP-2 | FMD ~0.05–0.10, high sensitivity |
| Stable across genres | Octuple + CLaMP-1 | Lowest CV across pairs |
| ⚠️ Avoid | Octuple + CLaMP-2 | Anomalously high FMD, poor discrimination |

### 📈 Effect Sizes (Cohen's d)

| Comparison | d | Magnitude |
|------------|---|-----------|
| Octuple vs REMI | **0.895** | **Large** |
| MIDI-Like vs REMI | 0.691 | Medium |
| CLaMP-1 vs CLaMP-2 | 0.615 | Medium |
| REMI vs TSD | −0.558 | Medium |
| MIDI-Like vs TSD | 0.016 | Negligible |

> **Key insight**: MIDI-Like and TSD produce nearly identical FMD distributions (d ≈ 0), while Octuple and REMI represent opposite extremes.

### 🔬 Statistical Validation

All main effects confirmed with:
- **Three-way factorial ANOVA** with interactions (N = 1920)
- **Permutation tests** (p < 0.001 for all main effects)
- **Tukey HSD** post-hoc comparisons
- **Per-pair η² consistency** analysis across 6 genre pairs

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
                    │    Embedding Model (×2)           │
                    │     CLaMP-1  |  CLaMP-2           │
                    └────────────┬────────────────────┘
                                 │
                    ┌────────────▼────────────────────┐
                    │   FMD Computation (×6 pairs)     │
                    │   10× repeated subsampling       │
                    │   = 1920 total observations      │
                    └──────────────────────────────────┘
```

## Project Structure

```
├── src/                           # Source code
│   ├── preprocessing/             # MIDI preprocessing
│   ├── tokenization/              # 4 tokenizers (MidiTok)
│   ├── embeddings/                # CLaMP-1/2 extraction + cache
│   ├── metrics/                   # FMD (Frechet Music Distance)
│   ├── experiments/               # Analysis & publication plots
│   └── utils/                     # Config and helpers
│
├── tests/                         # 53+ unit tests
├── configs/                       # YAML configuration
├── data/                          # Lakh MIDI subsets
├── results/
│   ├── reports/lakh/              # Single-pair analysis (32 variants)
│   ├── reports/lakh_multi/        # Multi-genre analysis (1920 obs)
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
python run_multi_genre_analysis.py   # 4 genres × 32 variants × 10 repeats
```
Produces ANOVA tables, interaction plots, η² heatmaps, and violin plots in `results/reports/lakh_multi/` and `results/plots/paper/`.

### Cross-Dataset Validation
```bash
python run_cross_dataset_validation.py                  # All sources (CD1 + MidiCaps)
python run_cross_dataset_validation.py --source cd1     # Tagtraum CD1 only
python run_cross_dataset_validation.py --source midicaps # MidiCaps only
python main.py --mode cross-validate                     # Via main entry point
python main.py --mode cross-validate --cv-source cd1     # Specific source
```
Validates generalizability by repeating the sensitivity analysis on independent data sources:
- **Lakh + Tagtraum CD1** — same MIDI files, different genre annotator (cross-annotation)
- **MidiCaps** — independent MIDI dataset with genre tags (cross-dataset)

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
python demo_embeddings.py --demo all
python demo_fmd.py --demo basic
python run_ablation_study.py
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

**👉 [See detailed summaries →](docs/weekly_summaries/)**

## Generated Outputs

### Reports

| File | Description |
|------|-------------|
| [`ANALYSIS_REPORT.md`](results/reports/lakh/ANALYSIS_REPORT.md) | Single-pair (rock vs jazz) full analysis |
| `multi_genre_fmd.csv` | Raw 1920 FMD observations |
| `eta_sq_per_pair.csv` | η² consistency across pairs |
| [`CROSS_VALIDATION_REPORT.md`](results/reports/cross_validation/CROSS_VALIDATION_REPORT.md) | Cross-dataset generalizability analysis |
| `cross_dataset_fmd.csv` | Combined FMD from all sources |

### Publication Plots

| Plot | Description |
|------|-------------|
| `multi_summary_4panel` | Overview: violin + heatmap + η² + interaction |
| `multi_fmd_violin_tok_model` | FMD distributions by tokenizer and model |
| `multi_eta_sq_heatmap` | η² stability across genre pairs |
| `multi_eta_sq_comparison` | Factor importance comparison |
| `multi_fmd_by_pair` | FMD distributions per genre pair |
| `multi_interaction_per_pair` | Tokenizer×model interaction per pair |
| `cross_summary_4panel` | Cross-dataset validation overview |
| `cross_eta_sq_comparison` | η² comparison across data sources |
| `cross_tok_model_heatmaps` | Tokenizer×model means per source |
| `cross_ranking_agreement` | Pipeline ranking Spearman ρ |
| `cross_fmd_violin_by_source` | FMD distributions per source |

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

**Status**: ✅ Complete (Weeks 1–6) | **Last Updated**: 19.04.2026
