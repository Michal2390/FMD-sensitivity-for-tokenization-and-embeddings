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
- **Preprocessing modifications**: velocity removal and hard quantization

on **Frechet Music Distance (FMD)**, a metric for symbolic music similarity.

## Project Structure

```
├── docs/                          # Documentation
│   ├── weekly_summaries/          # Weekly summaries (Weeks 1-5)
│   ├── setup/                     # Setup instructions
│   ├── references/                # Reference materials
│   └── private/                   # Private analysis (not in repo)
│
├── src/                           # Source code
│   ├── preprocessing/             # MIDI preprocessing
│   ├── tokenization/              # 4 tokenizers
│   ├── embeddings/                # CLaMP embeddings
│   ├── metrics/                   # FMD metric
│   └── utils/                     # Config and helpers
│
├── tests/                         # Test suite
├── configs/                       # Configuration
├── data/                          # Data
└── results/                       # Results
```

## Quick Start

### Installation
```bash
pip install -r requirements.txt
pip install -e .
```

### One-click run (PyCharm/IDE)
Run `main.py` with the **Run** button. The default mode is `quick`:
- sanity demo,
- quick paper benchmark,
- report generation in `results/reports/paper/`.

You can also run modes from CLI:
```bash
python main.py --mode quick
python main.py --mode paper
python main.py --mode paper-full
python main.py --mode paper-plots
python main.py --mode fetch-data
python main.py --mode tests
python main.py --mode full
```

Each mode prints stage-level progress (percentage) and a final elapsed-time message.

### Starter Data and Special Genre Comparisons
For article-oriented experiments, you can generate large, balanced, and more diverse MIDI sets (procedurally, locally):

```bash
python scripts/generate_starter_midis.py --count-per-dataset 120 --bars 16
```

Source of those files: **local synthetic generation** via `scripts/generate_starter_midis.py` (not downloaded from external audio/MIDI repositories).

After `paper-full`, special-pair reports (e.g., `jazz vs rock`, `classical vs pop`, `rap vs pop`) and publication plots are generated automatically. You can also run plotting as a separate step:

```bash
python scripts/generate_publication_plots.py
```

Genre alias mapping is stored in `configs/genre_mapping.yaml`.

Additional `paper` pipeline features:
- `fallback_mode: synthetic|strict|hard_strict` (`hard_strict` stops the run immediately),
- split analysis into `all` and `real-only`,
- bootstrap CI for FMD (`mean`, `std`, and confidence interval).

Embeddings are computed in inference mode (no neural network training). With `embeddings.device: auto`, CUDA is used when available (e.g., RTX 4070 Ti).

### Run Tests
```bash
pytest tests/ -v
pytest tests/test_paper_pipeline.py -v
```

### Run Demo
```bash
python demo_embeddings.py --demo all
python demo_fmd.py --demo basic
python run_ablation_study.py
```

## Weeks Overview

| Week | Topic | Status | Tests |
|------|-------|--------|-------|
| 1 | Initialization | ✅ | - |
| 2 | Preprocessing and Tokenization | ✅ | 47 |
| 3 | CLaMP Embeddings | ✅ | 29 |
| 4 | FMD Metric | ✅ | 24 |
| 5 | Ablation Study | ✅ | 13 |

**👉 [See detailed summaries →](docs/weekly_summaries/)**

## Main Components

- **Preprocessing**: MIDI loading, velocity removal, quantization
- **Tokenization**: REMI, TSD, Octuple, MIDI-Like (4 schemes)
- **Embeddings**: CLaMP-1/2 with dual-level cache
- **Metrics**: Frechet Music Distance with ranking and stability
- **Experiments**: Ablation study for sensitivity analysis

## Documentation

- **[Weekly Summaries](docs/weekly_summaries/)** - Detailed progress reports
- **[How to Run](docs/INSTRUKCJA_URUCHAMIANIA.md)** - Execution guide
- **[Setup](docs/setup/)** - Configuration and installation
- **[References](docs/references/)** - Design proposal and papers

## Key Results

✅ **53+ unit tests** (CI-verified core suite)  
✅ **4000+ lines of code**  
✅ **88% code coverage** (FMD module)  
✅ **Stable rankings** across configurations (consistency metric in the report)  
✅ **Paper outputs generated automatically**:
- `results/reports/paper/paper_results.json`
- `results/reports/paper/pairwise_fmd.csv`
- `results/reports/paper/special_pair_fmd.csv`
- `results/reports/paper/special_pair_summary.csv`
- `results/reports/paper/special_pair_top_variants.csv`
- `results/reports/paper/pairwise_fmd_all.csv`
- `results/reports/paper/pairwise_fmd_real_only.csv`
- `results/reports/paper/variant_delta_tokenizer.csv`
- `results/reports/paper/variant_delta_model.csv`
- `results/reports/paper/variant_delta_tokenizer_real_only.csv`
- `results/reports/paper/variant_delta_model_real_only.csv`
- `results/reports/paper/paper_summary.md`

✅ **Publication plots** (`results/plots/paper/`):
- `fig_pairwise_heatmap.png`
- `fig_special_pairs_boxplot.png`
- `fig_special_pairs_distinguishability.png`
- `fig_top_variants_per_pair.png`
- `fig_ranking_stability.png`
- `fig_rank_consistency.png`

## Academic Context

**Project**: Frechet Music Distance Sensitivity  
**Institution**: Warsaw University of Technology, EITI  
**Course**: WIMU (Music Information Retrieval)  
**Duration**: February-June 2026

---

**Status**: ✅ Complete (Weeks 1-5) | **Last Updated**: 31.03.2026

