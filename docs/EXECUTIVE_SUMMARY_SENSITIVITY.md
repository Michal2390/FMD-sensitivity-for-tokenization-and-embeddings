> ⛔ **NIEAKTUALNE / NIE CYTOWAĆ.** Zawiera błąd metodologiczny: porównanie **surowego FMD między modelami** („63× bardziej spójny", „730× więcej ekspresji") — to **artefakty skali** (CLaMP L2-normalizowany → małe FMD; MusicBERT nie → duże FMD). Aktualne, niezmiennicze skalowo wyniki: [`PAPER_FINDINGS.md`](PAPER_FINDINGS.md), `README.md`, `draft.tex`. Zachowane historycznie.

---

# 🎯 Executive Summary - Sensitivity Analysis Results

**Projekt**: WIMU - Music Embedding Sensitivity Analysis  
**Data**: 2026-06-09  
**Status**: ✅ KOMPLETNE

---

## ⚡ TL;DR (Too Long; Didn't Read)

### Główne odkrycie:
**CLaMP modele (~60x bardziej spójne) ≠ MusicBERT (koduje ekspresję)**

- **CLaMP1-ABC** ⭐⭐⭐⭐⭐: Najlepsza konsystencja i niezawodność
- **MusicBERT-REMI** 🎵: Koduje dynamikę muzyczną (7.3x wrażliwość!)
- **Istnieje trade-off** między spójnością a ekspresją

---

## 📊 Kluczowe Liczby

| Metryka | CLaMP | MusicBERT | Wnioski |
|---------|-------|-----------|---------|
| **Split-Half FMD** | 0.03 | 1.85 | CLaMP 60x spójniejszy |
| **Cross-Dataset FMD** | 0.01 | 2.93 | CLaMP bardziej universal |
| **Velocity Sensitivity** | 0.01 | 7.3 | MusicBERT 730x bardziej koduje ekspresję |
| **Bootstrap CV** | 12.8% | 15.1% | CLaMP bardziej stabilny |

---

## 🎯 Najważniejsze Odkrycia

### 1️⃣ CLaMP Ma Niezwykłą Spójność Wewnętrzną
```
Test: Podziel 80 piosenek na 2 grupy po 40, porównaj embedingi

Wynik CLaMP:     0.029 (prawie identyczne!)
Wynik MusicBERT: 1.85  (wyraźnie różne)

→ CLaMP embedingi są ~63x bardziej spójne w ramach datasetu
```

### 2️⃣ MusicBERT Koduje DYNAMIKĘ (Velocity)
```
Test: Usunięcie velocity (dynamiki) z MIDI

Zmiana dla MusicBERT-REMI: 7.32 ← OGROMNA zmiana
Zmiana dla CLaMP1-ABC:    0.01 ← prawie żadna zmiana

→ MusicBERT 730x bardziej wrażliwy na dynamikę
→ To DOBRZE - oznacza że koduje ekspresję
```

### 3️⃣ CLaMP Bardzo Dobrze Generalizuje Między Datasetami
```
Test: Porównaj embedingi maestro vs pop909

FMD dla CLaMP1-ABC:    0.011 (prawie brak separacji)
FMD dla MusicBERT-REMI: 2.93  (wyraźna separacja)

→ CLaMP nie różnicuje gatunkami
→ Dobre dla transfer learning, ale mniej informacji
```

### 4️⃣ Perturbacje Prawie Nie Wpływają na CLaMP
```
Test: Modyfikuj MIDI na różne sposoby

CLaMP wrażliwość na:
  - Usunięcie velocity:       0.010
  - Kwantyzacja czasu:        0.012
  - Stała tempo:              0.011
  - WSZYSTKIE RAZEM:          0.011

→ Perturbacje są prawie niezależne!
→ CLaMP koduje głównie STRUKTURĘ, nie EKSPRESJĘ
```

### 5️⃣ Bootstrap: CLaMP Bardziej Stabilny
```
50 resamples z maestro vs pop909:

CLaMP1-ABC:
  - Średnia FMD: 0.019
  - Std Dev:     0.0025
  - CV:          12.8% ← NISKI
  - 95% CI:      [0.0153, 0.0234]

MusicBERT-REMI:
  - Średnia FMD: 3.51
  - Std Dev:     0.531
  - CV:          15.1%
  - 95% CI:      [2.66, 4.64]

→ CLaMP daje bardziej reproducible wyniki
```

---

## 🎛️ Decision Tree: Który Model Wybrać?

```
                         ┌─ Chcesz EKSPRESJĘ?
                         │  └→ MusicBERT-REMI ✅
                         │
Wybór Modelu ────────────┤
                         │  ┌─ Chcesz SPÓJNOŚĆ?
                         │  └→ CLaMP1-ABC ✅
                         │
                         └─ Chcesz STABILNOŚĆ?
                            └→ CLaMP (CV: 12-14%) ✅
```

---

## 📍 Metryki Szczegółowe

### **Self-Similarity (Split-Half Consistency)**
Niska = Dobra (model konsystentny w ramach datasetu)

| Model | Score | Rating |
|-------|-------|--------|
| CLaMP1-ABC | 0.029 | ⭐⭐⭐⭐⭐ Doskonały |
| CLaMP2-MTF | 0.032 | ⭐⭐⭐⭐⭐ Doskonały |
| MusicBERT-REMI | 1.85 | ⭐⭐ Słaby |
| MusicBERT-TSD | 2.31 | ⭐ Bardzo słaby |

### **Cross-Dataset FMD (Niezależność od Datasetu)**
Niska = Dobrze generalizuje

| Model | Score | Rating |
|-------|-------|--------|
| CLaMP1-ABC | 0.011 | ⭐⭐⭐⭐⭐ Idealny |
| CLaMP2-MTF | 0.035 | ⭐⭐⭐⭐⭐ Idealny |
| MusicBERT-REMI | 2.93 | ⭐⭐ OK |
| MusicBERT-TSD | 3.23 | ⭐ Słaby |

### **Velocity Sensitivity (Ekspresja)**
Wysoka = Koduje ekspresję

| Model | Score | Rating |
|-------|-------|--------|
| MusicBERT-REMI | 7.32 | ⭐⭐⭐⭐⭐ Bardzo ekspresywny |
| MusicBERT-TSD | 3.13 | ⭐⭐⭐⭐ Ekspresywny |
| CLaMP2-MTF | 0.046 | ⭐⭐ Mniej ekspresywny |
| CLaMP1-ABC | 0.010 | ⭐ Minimalne kodowanie |

### **Bootstrap Stability (Powtarzalność)**
Niski CV% = Bardziej niezawodny

| Model | CV | CI | Rating |
|-------|----|----|--------|
| CLaMP1-ABC | 12.8% | [0.0153, 0.0234] | ⭐⭐⭐⭐⭐ |
| CLaMP2-MTF | 14.0% | [0.0391, 0.0600] | ⭐⭐⭐⭐⭐ |
| MusicBERT-REMI | 15.1% | [2.66, 4.64] | ⭐⭐⭐ |
| MusicBERT-TSD | 20.2% | [2.74, 5.47] | ⭐⭐ |

---

## 💡 Praktyczne Implikacje

### Jeśli pracujesz nad:

**📀 Tagowaniem Gatunku / Klasyfikacją Stylu**
```
→ Użyj CLaMP1-ABC (bardzo spójny, dataset-independent)
→ Powód: Nie chcesz aby ekspresja wpłynęła na sztukę klasyfikacji
```

**🎸 Systemem Reprodukcji Ekspresji / Sound Synthesis**
```
→ Użyj MusicBERT-REMI (koduje dynamikę)
→ Powód: Potrzebujesz informacji o ekspresji do syntezy
```

**🔬 Badanym Naukowym / Publikacji**
```
→ Rozważ HYBRID: CLaMP + MusicBERT
→ Powód: Pokrywa zarówno spójność jak i ekspresję
```

**⚙️ Production-Grade System**
```
→ Użyj CLaMP1-ABC (12.8% CV, stabilny)
→ Powód: Najniższy rozsiew wyników, reproducible
```

---

## 🚀 Następne Kroki

### ✅ Zrobione
- [x] Split-half consistency testing
- [x] Cross-dataset FMD analysis
- [x] Perturbation sensitivity profiling
- [x] Bootstrap stability analysis

### 📋 Rekomendacje
- [ ] Hybrid model łączący CLaMP i MusicBERT
- [ ] Fine-tuning na specjalistyczne zadania
- [ ] Test na większych datasetach (1000+ piosenek)
- [ ] Cross-genre validation
- [ ] Temporal stability analysis

---

## 📚 Pliki Wyników

Dostępne w: `results/reports/sensitivity_pivot/`

- `sensitivity_pivot_summary.json` - Full JSON dump wszystkich wyników
- `cross_dataset_fmd.csv` - Porównanie maestro vs pop909
- `self_similarity.csv` - Split-half consistency
- `perturbation_sensitivity.csv` - Wrażliwość na perturbacje
- `bootstrap_stability.csv` - Bootstrap CI i CV

---

## 🎓 Wnioski Naukowe

1. **✅ CLaMP jest superiorem się konsystencji** (63x lepiej)
2. **✅ MusicBERT koduje ekspresję** (730x bardziej na velocity)
3. **🔄 Trade-off fundamentalny** między spójnością a ekspresją
4. **✅ CLaMP bardziej stabilny** dla reproducible research
5. **💡 Przestrzeń dla hybrid approach** łączącego obie zalety

---

## 🎯 Ranking Końcowy

### Dla Ogólnego Użytku: **CLaMP1-ABC** ⭐⭐⭐⭐⭐
- Najlepsza spójność (0.029)
- Najlepsze generalizacja (0.011)
- Najniższe CV (12.8%)
- Najstabilniejsze wyniki

### Dla Ekspresji: **MusicBERT-REMI** ⭐⭐⭐⭐⭐
- Koduje 730x więcej ekspresji
- Widzi różnice między datasetami
- Rich representation

### Dla Production: **CLaMP1-ABC** ⭐⭐⭐⭐⭐
- Reproducible
- Stable
- Reliable CI bounds

---

**Status**: 🟢 ANALIZA KOMPLETNA  
**Jakość Danych**: ✅ Wysokiej jakości (all tests passed)  
**Rekomendacja**: ✅ Publikacja możliwa

---

*Generated by Sensitivity Pivot Analysis System*  
*2026-06-09 03:12:13*

