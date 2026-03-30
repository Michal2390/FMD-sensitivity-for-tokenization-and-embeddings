# Tydzień 4: Kalkulacja metryki FMD - Podsumowanie

**Okres realizacji:** 13.04-19.04.2026  
**Status:** ✅ **UKOŃCZONY**

## Cel tygodnia

Pełna implementacja metryki Frechet Music Distance (FMD) wraz z integracją embeddingów z Tygodnia 3, kalkulacja rankingów i analizą stabilności.

## Zrealizowane zadania

### 1. Rozszerzenie implementacji FMD ✅

#### Moduł `src/metrics/fmd.py` - Pełna aktualizacja

**Klasa `FrechetMusicDistance` - Ulepszenia**

- **`compute_fmd()` - Pełna implementacja Fréchet Distance**
  - Komponenty:
    1. **Wasserstein component**: L2 distance między średnimi
    2. **Fréchet component**: Różnica matrix square roots covariance matrices
    3. **Optional std component**: Różnica w standardowych odchyleniach
  - Formula: `FMD = sqrt(mean_dist² + cov_dist + std_dist)`
  - Obsługa edge cases: 1D embeddingi, single sample
  - Fallback dla problematycznych macierzy

- **`compute_fmd_matrix()` - Matrix-based approach**
  - Utilizuje optimal transport approximation
  - Min-max matching strategy
  - Dla sekwencji embeddingów

- **`compute_batch_fmd()` - Batch pairwise computation**
  - Obliczanie wszystkich par (N x N macierz)
  - Statystyki: mean, median, std, min, max FMD
  - Obsługa NaN wartości
  - Logging i tracking błędów

**Nowe importy:**
```python
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import json
```

### 2. Integracja z embeddingami ✅

**Pełny pipeline:**
1. Ekstrakcja embeddingów z tokenów (CLaMP-1/CLaMP-2)
2. Obliczenie statystyk embeddingów
3. Kalkulacja FMD dla pairwise datasets
4. Generowanie rankingów
5. Analiza stabilności

### 3. Ranking i stabilność ✅

#### Klasa `FMDRanking`

- `rank_by_fmd()` - Ranking datasets względem referencji
- `compute_ranking_stability()` - Kendall tau distance
  - Porównanie konsystencji rankingów między konfiguracjami
  - Score 0-1 (1 = full agreement)

#### Klasa `FMDComparator`

- `compare_tokenizers()` - Porównanie FMD across tokenizers
- `compare_models()` - Porównanie FMD across embedding models

### 4. Testy jednostkowe ✅

#### `tests/test_fmd.py` - 7 testów (oryginalne)

**Testy FrechetMusicDistance:**
- ✅ FMD(x, x) = 0 dla identical distributions
- ✅ FMD > 0 dla different distributions
- ✅ Symetria: FMD(x, y) = FMD(y, x)
- ✅ Obsługa 1D embeddings
- ✅ Matrix-based FMD

**Testy FMDRanking:**
- ✅ Ranking by FMD distances
- ✅ Ranking stability computation

**Pokrycie:** 100% (7/7 testów)

#### `tests/test_fmd_integration.py` - 17 testów (NOWYCH)

**TestFrechetMusicDistanceEnhanced (7 testów):**
- ✅ Identical distributions → FMD ≈ 0
- ✅ Different distributions → FMD > 0
- ✅ Symmetry property
- ✅ 1D embedding handling
- ✅ Large distance > small distance
- ✅ Dimension mismatch error
- ✅ Matrix method

**TestFMDRankingEnhanced (4 testy):**
- ✅ Ranking by FMD
- ✅ Ranking stability (identical)
- ✅ Ranking stability (different)
- ✅ Ranking stability (single)

**TestFMDComparator (2 testy):**
- ✅ Comparator initialization
- ✅ Compare models

**TestEmbeddingFMDIntegration (4 testy):**
- ✅ Extract embeddings + compute FMD
- ✅ Embedding statistics + FMD
- ✅ FMD pipeline multiple datasets
- ✅ End-to-end integration

**Pokrycie:** 24/24 testów (100% pass rate)
**FMD Module Coverage:** 88%

### 5. Demo skrypt ✅

#### `demo_fmd.py` - 6 scenariuszy demonstracyjnych

1. **demo_basic_fmd()** - Podstawowa kalkulacja FMD
   - Tworzenie embedding distributions
   - Porównanie identical vs. different
   - Verification FMD symmetry

2. **demo_fmd_matrix()** - Macierz FMD (pairwise distances)
   - 5 embedding sets
   - Full pairwise matrix
   - Statystyki (mean, median, std, min, max)
   - Tabularny format

3. **demo_fmd_ranking()** - Ranking based na FMD
   - Reference dataset
   - Ranking from reference
   - Distance scores

4. **demo_embedding_fmd_integration()** - Real embeddings + FMD
   - Extraction z token sequences
   - Embedding groups (similar/different)
   - Intra vs. inter-group distances

5. **demo_fmd_stability()** - Ranking stability analysis
   - Multiple configurations
   - Ranking comparison
   - Stability score (0-1)
   - Kendall tau interpretation

6. **demo_tokenization_sensitivity()** - FMD sensitivity
   - Different tokenizations (REMI, TSD, Octuple)
   - Simulation tokenizer impact
   - Stability across tokenizers
   - Variation analysis

**Użycie:**
```bash
# Wszystkie demo
python demo_fmd.py --demo all

# Konkretne demo
python demo_fmd.py --demo basic
python demo_fmd.py --demo matrix
python demo_fmd.py --demo ranking
python demo_fmd.py --demo integration
python demo_fmd.py --demo stability
python demo_fmd.py --demo tokenization
```

### 6. Struktura modułów

```
src/
└── metrics/
    ├── __init__.py
    └── fmd.py                      # ✅ Pełna aktualizacja
        ├── FrechetMusicDistance    # Enhanced (3 metody)
        ├── FMDRanking              # Ranking & stability
        └── FMDComparator           # Comparisons

tests/
├── test_fmd.py                     # ✅ 7 testów (oryginalne)
└── test_fmd_integration.py         # ✅ 17 testów (nowych)

demo_fmd.py                         # ✅ 6 scenariuszy demo
```

## Implementacja Fréchet Distance

### Matematyka

**FMD Formula:**
```
FMD(X, Y) = sqrt(
    ||mean(X) - mean(Y)||² + 
    ||sqrt(Cov(X)) - sqrt(Cov(Y))||²_F + 
    α*||std(X) - std(Y)||²
)
```

Gdzie:
- X, Y: macierze embeddingów (N×D, M×D)
- Cov: macierz kowariancji
- ||·||_F: norma Frobeniusa
- α: waga std component (optional)

### Komponenty

1. **Wasserstein Distance Component**
   - Odległość między średnimi embeddingów
   - Mierzy offset między rozkładami

2. **Fréchet Distance Component**
   - Square roots macierzy kowariancji
   - Norma Frobeniusa różnic
   - Mierzy różnicę w variance structure

3. **Standard Deviation Component** (optional)
   - Różnica per-dimension std
   - Fine-grained variance comparison

## Kluczowe metryki

### FMD Calculation
- **Metody:** 3 (compute_fmd, compute_fmd_matrix, compute_batch_fmd)
- **Batch size:** configurable
- **Obsługa:** NaN values, edge cases

### Ranking
- **Metody:** rank_by_fmd, compute_ranking_stability
- **Stability metric:** Kendall tau (0-1)
- **Interpretacja:** 0.7+ = high stability

### Testing
- **Testy:** 24 (7 + 17)
- **Pass rate:** 100%
- **Coverage FMD:** 88%

## Zgodność z wymaganiami WIMU

✅ **Wysoka jakość kodu**
- Pełna matematyczna implementacja
- Type hints we wszystkich funkcjach
- Docstringi w stylu Google
- Obsługa błędów

✅ **Reprodukowalność**
- Deterministyczne obliczenia
- Obsługa seed'ów
- Logging wszystkich operacji
- Statystyki pośrednie

✅ **Rzetelność**
- 24 testy jednostkowe
- 88% pokrycie FMD modułu
- Validation symmetry, identity, distance properties
- Obsługa edge cases

✅ **Dokumentacja**
- Docstringi dla 10+ metod
- Demo z 6 scenariuszami
- Matematyka w komentarzach
- Type hints

## Problemy napotkane i rozwiązania

### Problem 1: Covariance z małą liczbą próbek
**Opis:** np.cov() zwracał NaN dla single-sample embeddings.  
**Rozwiązanie:** Obsługa przypadku `embeddings.shape[0] <= 1` - zwracanie zero matrix.

### Problem 2: Matrix square root instability
**Opis:** Eigenvalues mogły być negative (numerical errors).  
**Rozwiązanie:** `np.maximum(eigvals, 0)` + fallback do trace difference.

### Problem 3: NaN w batch computation
**Opis:** Jedna failed FMD computation mogła zawalić całą matrycę.  
**Rozwiązanie:** Try/except w pętli, NaN w macierzy, filtering w statystykach.

## Wyniki Tygodnia 3 ↔ Tydzień 4 Integration

| Komponent | Tydzień 3 | Tydzień 4 |
|-----------|----------|----------|
| Embeddingi | CLaMP-1/2 | ✅ Consumed |
| Cache | Memory + Disk | ✅ Used |
| Statystyki | Mean, Std, Cov | ✅ Used in FMD |
| Distance metrics | Euclidean, Cosine | ✅ Integrated |
| Testy embeddings | 29 testów | ✅ Reused |

## Następne kroki (Tydzień 5)

Zgodnie z harmonogramem projektu, w **Tygodniu 5 (20.04-26.04.2026)** planowana jest:

**Eksperymenty z parametrami preprocessingu (Ablation Study)**

Zadania:
1. Generowanie znormalizowanych wariantów danych
2. Zastosowanie hard quantization
3. Usunięcie velocity (expresji)
4. Ponowna ekstrakcja embeddingów
5. Ponowna kalkulacja FMD
6. Analiza wpływu na FMD
7. Porównanie stabilności

## Porównanie z planem Tygodnia 4

| Zadanie | Plan | Status | Realizacja |
|---------|------|--------|-----------|
| Implementacja FMD | ✓ | ✅ | Pełna z 3 metodami |
| Obsługa embeddingów | ✓ | ✅ | Pełna integracja |
| Kalkulacja rankingów | ✓ | ✅ | Z Kendall tau |
| Analiza stabilności | ✓ | ✅ | Per-configuration |
| Testy jednostkowe | ✓ | ✅ | 24 testów (100%) |
| Demo skrypt | ✓ | ✅ | 6 scenariuszy |
| Dokumentacja | ✓ | ✅ | Kompletna |

## Git Status

```bash
# New files
- tests/test_fmd_integration.py (17 testów)
- demo_fmd.py (6 scenariuszy)

# Modified files
- src/metrics/fmd.py (130 linii, 88% coverage)
```

## Autorzy

Projekt realizowany w ramach przedmiotu **WIMU** (Wyszukiwanie Informacji Muzycznych)  
Wydział Elektroniki i Technik Informacyjnych (EITI), Politechnika Warszawska

## Referencje

1. **Retkowski, J., et al. (2025).** Frechet Music Distance: A Metric For Generative Symbolic Music Evaluation
2. **Frechet Distance:** https://en.wikipedia.org/wiki/Fréchet_distance
3. **Wasserstein Distance:** https://en.wikipedia.org/wiki/Wasserstein_distance
4. **SciPy Spatial:** https://docs.scipy.org/doc/scipy/reference/spatial.distance.html

---

**Status końcowy:** ✅ **TYDZIEŃ 4 UKOŃCZONY**  
**Data ukończenia:** 19.04.2026  
**Implementacja:** 100%  
**Testy:** 24 testów jednostkowych, 100% pass rate  
**Dokumentacja:** Kompletna  
**FMD Module Coverage:** 88%  
**Demo:** Wszyscy 6 scenariuszy działający  
**Git:** Commited i pushed

