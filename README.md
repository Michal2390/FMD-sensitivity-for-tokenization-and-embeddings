# FMD Sensitivity for Tokenization and Embeddings

**Advanced Music Information Retrieval Research Project**

> Systematyczne badanie wrażliwości metryki Frechet Music Distance (FMD) na wybór tokenizacji i modelu embeddingów dla symbolicznej muzyki

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-red)](https://pytorch.org/)
[![Tests: 53/53](https://img.shields.io/badge/Tests-53%2F53%20✓-success)](tests/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)]()

## 📋 Szybki przegląd

Projekt implementuje pełny pipeline do badania wpływu:
- **4 strategii tokenizacji**: REMI, TSD, Octuple, MIDI-Like
- **2 modeli embeddingów**: CLaMP-1, CLaMP-2  
- **Preprocessing modifications**: Usuwanie velocity, hard quantization

Na metrykę **Frechet Music Distance** - nową metrykę do oceny podobieństwa muzyki symbolicznej.

## 🏗️ Struktura projektu

```
├── docs/                          # 📚 Dokumentacja
│   ├── weekly_summaries/          # Podsumowania tygodni (Weeks 1-5)
│   ├── setup/                     # Instrukcje konfiguracji
│   ├── references/                # Materiały referencyjne
│   └── private/                   # Private analysis (nie w repo)
│
├── src/                           # 💻 Kod źródłowy
│   ├── preprocessing/             # MIDI preprocessing
│   ├── tokenization/              # 4 tokenizatory
│   ├── embeddings/                # CLaMP embeddings
│   ├── metrics/                   # Metryka FMD
│   └── utils/                     # Config & helpers
│
├── tests/                         # ✅ 53 testy (100% pass)
├── configs/                       # ⚙️ Konfiguracja
├── data/                          # 📊 Dane
└── results/                       # 📈 Wyniki
```

## 🚀 Quick Start

### Instalacja
```bash
pip install -r requirements.txt
pip install -e .
```

### One-click run (PyCharm/IDE)
Uruchom `main.py` przyciskiem **Run**. Domyslny tryb to `quick`:
- sanity demo,
- szybki benchmark paperowy,
- zapis raportow do `results/reports/paper/`.

Mozesz tez uruchomic tryby z CLI:
```bash
python main.py --mode quick
python main.py --mode paper
python main.py --mode paper-full
python main.py --mode tests
python main.py --mode full
```

### Uruchomienie testow
```bash
pytest tests/ -v
pytest tests/test_paper_pipeline.py -v
```

### Uruchomienie demo
```bash
python demo_embeddings.py --demo all
python demo_fmd.py --demo basic
python run_ablation_study.py
```

## 📊 Weeks Overview

| Week | Topic | Status | Tests |
|------|-------|--------|-------|
| 1 | Initialization | ✅ | - |
| 2 | Preprocessing & Tokenization | ✅ | 47 |
| 3 | CLaMP Embeddings | ✅ | 29 |
| 4 | FMD Metric | ✅ | 24 |
| 5 | Ablation Study | ✅ | 13 |

**👉 [See detailed summaries →](docs/weekly_summaries/)**

## 🔬 Main Components

- **Preprocessing**: MIDI loading, velocity removal, quantization
- **Tokenization**: REMI, TSD, Octuple, MIDI-Like (4 schemes)
- **Embeddings**: CLaMP-1/2 with dual-level cache
- **Metrics**: Frechet Music Distance with ranking & stability
- **Experiments**: Ablation study for sensitivity analysis

## 📚 Documentation

- **[Weekly Summaries](docs/weekly_summaries/)** - Detailed progress reports
- **[How to Run](docs/INSTRUKCJA_URUCHAMIANIA.md)** - Execution guide
- **[Setup](docs/setup/)** - Configuration & installation
- **[References](docs/references/)** - Design proposal & papers

## 📈 Key Results

✅ **53+ unit tests** (CI-verified core suite)  
✅ **4000+ lines of code**  
✅ **88% code coverage** (FMD module)  
✅ **Stable rankings** across configurations (consistency metric in report)  
✅ **Paper outputs generated automatically**:
- `results/reports/paper/paper_results.json`
- `results/reports/paper/pairwise_fmd.csv`
- `results/reports/paper/paper_summary.md`

## 🎓 Academic Context

**Project**: Wrażliwość Frechet Music Distance  
**Institution**: Politechnika Warszawska, EITI  
**Course**: WIMU (Wyszukiwanie Informacji Muzycznych)  
**Duration**: February-June 2026

---

**Status**: ✅ Complete (Weeks 1-5) | **Last Updated**: 31.03.2026

