> ⛔ **NIEAKTUALNE / NIE CYTOWAĆ.** Dokument z wcześniejszego runu zawiera błąd
> metodologiczny: porównuje **surowe FMD między modelami** („60–80× bardziej spójny",
> „730× więcej ekspresji"). To **artefakty skali** — embeddingi CLaMP są L2-normalizowane
> (małe FMD), MusicBERT nie (duże FMD), więc surowe wartości nie są porównywalne między
> modelami. Aktualne, niezmiennicze skalowo wyniki: [`PAPER_FINDINGS.md`](PAPER_FINDINGS.md),
> zaktualizowany `README.md` oraz `draft.tex`. Zachowane wyłącznie historycznie.

---

# Analiza Wyników Sensitivity Pivot Analysis

**Data**: 2026-06-09  
**Run**: Sensitivity Profiling with Perturbation Analysis

---

## 📊 Podsumowanie Wykonanych Testów

Przeprowadzono kompleksową analizę wrażliwości (sensitivity analysis) dla 4 konfiguracji tokenizacji i embedingu:
- **MusicBERT-REMI** (basowany na REMI tokenizacji)
- **MusicBERT-TSD** (basowany na TSD tokenizacji)
- **CLaMP2-MTF** (CLaMP v2 z Multi-Task Framework)
- **CLaMP1-ABC** (CLaMP v1 z ABC tokenizacją)

### Testowane Parametry:
✅ **Self-similarity** (split-half test) - Dwie datasety: maestro i pop909  
✅ **Cross-dataset FMD** - Porównanie maestro vs pop909  
✅ **Perturbation sensitivity** - Testowanie wrażliwości na modyfikacje MIDI:
   - Usunięcie velocity (dynamiki)
   - Kwantyzacja czasu (usunięcie mikrotiming)
   - Stała tempo (usunięcie zmian tempa)
   - Kombinacja wszystkich perturbacji
✅ **Bootstrap stability** - 50-krotne resampling z 95% CI

---

## 🎯 KLUCZOWE ODKRYCIA

### 1. **CLaMP Modele Znacznie Lepsze Niż MusicBERT**

#### Self-Similarity (Split-Half Test)
Split-half FMD powinno być **jak najniższe** (reprezentuje wewnętrzną konsystencję):

| Konfiguracja | Maestro | Pop909 | Średnia |
|---|---|---|---|
| **MusicBERT-REMI** | 2.12 | 1.57 | **1.85** ⚠️ |
| **MusicBERT-TSD** | 2.31 | 2.32 | **2.31** ⚠️ |
| **CLaMP2-MTF** | 0.036 | 0.029 | **0.032** ✅ |
| **CLaMP1-ABC** | 0.035 | 0.022 | **0.029** ✅ |

**Wniosek**: CLaMP modele są ~**60-80x bardziej konsystentne** w ramach tej samej datasety!

#### Cross-Dataset Distance (Maestro vs Pop909)
FMD powinno być rozsądne - pokazuje separację między datasetami:

| Konfiguracja | FMD |
|---|---|
| MusicBERT-REMI | 2.933 |
| MusicBERT-TSD | 3.235 |
| **CLaMP2-MTF** | **0.035** |
| **CLaMP1-ABC** | **0.011** |

**Obserwacja**: CLaMP znajduje bardzo mało różnic między maestro i pop909! To może oznaczać:
- Bardzo solidne, datasetowe-niezależne reprezentacje, ALBO
- Potencjalną utratę wyrażalności (mniej wrażliwości na rzeczywiste różnice stylistyczne)

---

### 2. **Wrażliwość na Perturbacje - MusicBERT Bardziej Ekspresyjny**

#### MusicBERT-REMI - Wrażliwość na Perturbacje:
| Perturbacja | FMD | Wniosek |
|---|---|---|
| **Bez velocity** | 7.32 | 🔴 **WYSOKA** - dynamika ma OGROMNY wpływ |
| Kwantyzacja czasu | 0.104 | Mikrotiming ma mały wpływ |
| Stała tempo | 0.148 | Rubato ma mały wpływ |
| Wszystkie razem | 7.63 | Złączenie zbliżone do samej velocity |

#### CLaMP1-ABC - Wrażliwość na Perturbacje:
| Perturbacja | FMD | Wniosek |
|---|---|---|
| Bez velocity | 0.0103 | ✅ **NISKA** - nie koduje dynamiki |
| Kwantyzacja czasu | 0.0116 | Nie koduje mikrotimingu |
| Stała tempo | 0.0114 | Nie koduje tempa rubato |
| Wszystkie razem | 0.0105 | Perturbacje prawie nie zmieniają embedingu |

#### CLaMP2-MTF - Pośrednia wrażliwość:
| Perturbacja | FMD | Wniosek |
|---|---|---|
| Bez velocity | 0.046 | ✅ Trochę wrażliwy na dynamikę |
| Kwantyzacja czasu | 0.026 | |
| Stała tempo | 0.005 | Najmniej wrażliwy na tempo |
| Wszystkie razem | 0.061 | |

**Interpretacja**:
- **MusicBERT-REMI**: Koduje **dużo ekspresji** - szczególnie dynamiki
- **CLaMP**: Koduje głównie **nutyzację i strukturę** - MNIEJ ekspresji
- **MusicBERT-TSD**: Pośrednia wrażliwość (3.13 na brak velocity)

---

### 3. **Bootstrap Stability - Stabilność Szacunków**

Cross-dataset FMD z 95% przedziałami ufności (50 resamples):

| Konfiguracja | Średnia | Std Dev | CI | CV |
|---|---|---|---|---|
| MusicBERT-REMI | 3.51 | 0.531 | [2.66, 4.64] | 15.1% ⚠️ |
| MusicBERT-TSD | 3.92 | 0.791 | [2.74, 5.47] | 20.2% ⚠️⚠️ |
| **CLaMP2-MTF** | **0.048** | 0.0067 | [0.039, 0.060] | **14.0%** ✅ |
| **CLaMP1-ABC** | **0.019** | 0.0025 | [0.015, 0.023] | **12.8%** ✅ |

**Wniosek**: CLaMP ma **niższy współczynnik zmienności (CV)** i bardziej sztywne przedziały ufności
- To oznacza: CLaMP daje bardziej **wiarygodne, powtarzalne wyniki**
- MusicBERT bardziej zmiennych między próbami

---

## 📈 Rekomendacje Dla Badań

### ✅ Kiedy użyć CLaMP:
1. Potrzebujesz **wysokiej spójności wewnątrz-datasetowej**
2. Szukasz **reprezentacji niezależnych od datasetu**
3. Chcesz **stabilnych, powtarzalnych wyników**
4. Fokus na **abstrakcyjnej strukturze muzycznej** (nie ekspresji)

### ✅ Kiedy użyć MusicBERT:
1. **Ekspresja muzyczna** jest ważna (dynamika, rubato, tempo)
2. Potrzebujesz więcej **wyrażającej mocy** embedingów
3. Szukasz **różnorodności** w reprezentacji
4. OK więcej zmienności, ale bardziej "bogata" reprezentacja

---

## 🔍 Szczegółowa Analiza Każdej Metryki

### A. Self-Similarity (Split-Half Consistency)
```
Czym to jest: Dzielę datasety na dwie połowy, licze embedingi
              dla każdej, i porównuję je. Niska wartość = wysokie
              dopasowanie = konsystencja.
              
Interpretacja:
- MusicBERT: ~1.5-2.3  -> Dość zmienne
- CLaMP:     ~0.022-0.036 -> Bardzo spójne
```

### B. Cross-Dataset FMD (Dataset Independence)
```
Czym to jest: Licze Fréchet Mean Distance między embedingami
              z maestro vs pop909.
              
Interpretacja:
- MusicBERT: ~2.9-3.2 -> Wyraźnie widzi różnicę
- CLaMP1/2:  ~0.01-0.035 -> Prawie nie widzi różnicy
```

**⚠️ PARADOKS**: CLaMP lepiej jako ogólny model, ale Mniej wrażliwy na różnice między datasetami.

### C. Perturbation Sensitivity
```
Czym to jest: Modyfikuję MIDI (usuwam dynamikę, kvantyzuję, 
              stała tempo) i patrzę jak zmienia się embedding.
              
Wysokie wartości = model koduje tę cechę
Niskie wartości = model ignoruje tę cechę
```

**Kluczowa Obserwacja**:
- **MusicBERT koduje ekspresję** (velocity ma ogromny wpływ)
- **CLaMP koduje strukturę** (prawie nie zmienia się przy perturbacjach)

### D. Bootstrap Stability
```
Czym to jest: 50x resampluję dane, licze FMD, patrzę na 
              rozkład wyników. Niskie CV = stabilne.
              
Interpretacja:
- MusicBERT CV: 15-20% -> Zmienne
- CLaMP CV:    12-14% -> Stabilne
```

---

## 🎓 Wnioski Naukowe

### Hipoteza 1: ✅ POTWIERDZONO
**CLaMP embedingi mają wyższą spójność wewnątrz-gruppową**
- Split-half FMD: CLaMP ~60-80x niższy
- Bootstrap CV: CLaMP ~12-14% vs MusicBERT ~15-20%

### Hipoteza 2: ⚠️ CZĘŚCIOWO POTWIERDZONO
**CLaMP lepiej generalizuje między datasetami**
- Cross-dataset FMD bardzo niski (~0.01-0.035)
- ALE: Może to oznaczać utratę ekspresji, nie lepszą generalizację

### Hipoteza 3: ✅ POTWIERDZONO
**MusicBERT koduje więcej aspektów ekspresji**
- Usunięcie velocity zmienia embedingi o 7.3 (REMI) vs 0.01 (CLaMP1-ABC)
- MusicBERT ~730x bardziej wrażliwy na dynamikę

---

## 📋 Metryki Dokładne

### Self-Similarity:
- **MusicBERT-REMI** (Maestro): 2.1237
- **MusicBERT-REMI** (Pop909): 1.5723
- **MusicBERT-TSD** (Maestro): 2.3083
- **MusicBERT-TSD** (Pop909): 2.3230
- **CLaMP2-MTF** (Maestro): 0.0356
- **CLaMP2-MTF** (Pop909): 0.0286
- **CLaMP1-ABC** (Maestro): 0.0354
- **CLaMP1-ABC** (Pop909): 0.0223

### Perturbation Sensitivity:

**MusicBERT-REMI:**
- no_velocity: 7.3166
- quantized_time: 0.1043
- constant_tempo: 0.1483
- all_combined: 7.6272

**CLaMP1-ABC:**
- no_velocity: 0.0103
- quantized_time: 0.0116
- constant_tempo: 0.0114
- all_combined: 0.0105

---

## 📁 Pliki Wyników

- `cross_dataset_fmd.csv` - Porównanie maestro vs pop909
- `perturbation_sensitivity.csv` - Testy perturbacji
- `self_similarity.csv` - Split-half consistency
- `bootstrap_stability.csv` - Bootstrap CI i CV
- `sensitivity_pivot_summary.json` - Pełny podsumowanie

---

## 🔬 Znaczenie Dla Projektu

### Dla publikacji:
1. **CLaMP** pokazuje wyłączną konsystencję - idealny do production-grade systemów
2. **MusicBERT** pokazuje bogactwo ekspresji - ważne dla artistic applications
3. **Perturbation test** potwierdza że token representation matters

### Dla dalszych badań:
- Zbadać hybrid approach (MusicBERT + CLaMP features?)
- Test na większych datasetach
- Cross-validation na nowych gatunkach muzyki

---

**Koniec Raportu**

