> ⛔ **NIEAKTUALNE / NIE CYTOWAĆ.** Zawiera błąd metodologiczny: porównanie **surowego FMD między modelami** („63×–96×", „730×") — to **artefakty skali** (CLaMP L2-normalizowany → małe FMD; MusicBERT nie → duże FMD). Aktualne, niezmiennicze skalowo wyniki: [`PAPER_FINDINGS.md`](PAPER_FINDINGS.md), `README.md`, `draft.tex`. Zachowane historycznie.

---

# Sensitivity Analysis - Porównanie Modeli (Wizualizacja)

## 📊 Tabela Porównawcza Wszystkich Metryk

### 1. SELF-SIMILARITY (Split-Half FMD) - Spójność Wewnętrzna

```
           MAESTRO (↓ niżej = lepiej)     POP909 (↓ niżej = lepiej)     ŚREDNIA
           ━━━━━━━━━━━━━━━━━━━━━━━━━━━━   ━━━━━━━━━━━━━━━━━━━━━━━━   ━━━━━━━━━

MusicBERT-REMI    2.12                      1.57                      1.85 ⚠️
MusicBERT-TSD     2.31                      2.32                      2.31 ⚠️⚠️
CLaMP2-MTF        0.036 ✅                  0.029 ✅                  0.032 ✅
CLaMP1-ABC        0.035 ✅                  0.022 ✅                  0.029 ✅

Ratio CLaMP/MusicBERT: ~63x-96x LEPSZY CLaMP!
```

### 2. CROSS-DATASET FMD (Maestro vs Pop909)

```
Konfiguracja           FMD        Interpretacja
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
MusicBERT-REMI        2.933      Dość wyraźna separacja między datasetami
MusicBERT-TSD         3.235      Większa separacja - widzi różnice
CLaMP2-MTF            0.035      Bardzo mała separacja 
CLaMP1-ABC            0.011 ⚠️   NAJMNIEJSZA - praktycznie nie odróżnia
```

**⚠️ Interpretacja**: 
- Niska wartość = dataset-independent (dobrze dla generalizacji)
- ALE: Może oznaczać, że model nie koduje informacji specyficznej dla datasetu

---

## 🎯 PERTURBATION SENSITIVITY - Wrażliwość na Modyfikacje

### A. Usunięcie VELOCITY (Dynamiki)

```
                       FMD     ↑ wyżej = bardziej koduje dynamikę
                       ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
MusicBERT-REMI        7.32     🔴 BARDZO WRAŻLIWY (7.3x zmiana!)
MusicBERT-TSD         3.13     🟠 WRAŻLIWY
CLaMP2-MTF            0.046    🟡 Trochę wrażliwy
CLaMP1-ABC            0.0103   🟢 Praktycznie nie wrażliwy
                      
Ratio: MusicBERT-REMI vs CLaMP1-ABC = 730x!
```

**Wniosek**: MusicBERT BARDZO zależy od dynamiki, CLaMP prawie nie koduje.

---

### B. Kwantyzacja Czasu (Usunięcie Mikrotimingu)

```
                       FMD     ↑ wyżej = bardziej koduje mikrotiming
                       ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
MusicBERT-REMI        0.104    Mały wpływ
MusicBERT-TSD         0.259    Większy wpływ
CLaMP2-MTF            0.026    Bardzo mały wpływ
CLaMP1-ABC            0.0116   Prawie zerowy wpływ

Wniosek: Żaden model dużo nie zależy od mikrotimingu
```

---

### C. Stałe TEMPO (Brak Rubato/zmian tempa)

```
                       FMD     ↑ wyżej = bardziej koduje rubato
                       ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
MusicBERT-REMI        0.148    Mały wpływ
MusicBERT-TSD         0.329    Średni wpływ
CLaMP2-MTF            0.0050   ✅ PRAWIE ZERO!
CLaMP1-ABC            0.0114   🟢 Prawie zero

Wniosek: MusicBERT nieco koduje tempo, CLaMP nie
```

---

### D. Wszystkie Perturbacje Razem (Total De-Expression)

```
                       FMD     (przed i po wszystkich zmianach)
                       ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
MusicBERT-REMI        7.63 ≈ 7.32 (velocity dominuje!)
MusicBERT-TSD         3.96 ≈ 3.13 (velocity dominuje!)
CLaMP2-MTF            0.061
CLaMP1-ABC            0.0105 ≈ sama wartość co każda perturbacja!

Obserwacja: Dla CLaMP perturbacje są niezależne + sumują się prawie
            na zero, podczas gdy MusicBERT dominuje velocity.
```

---

## 📈 Bootstrap Stability - Powtarzalność

```
┌─────────────────────────────────────────────────────────────────────┐
│ CROSS-DATASET FMD (maestro vs pop909) - 50 Resamples               │
├──────────────────┬──────────┬────────┬──────────────────┬────────┤
│ Model            │ FMD Mean │ Std Dev│ 95% CI           │  CV   │
├──────────────────┼──────────┼────────┼──────────────────┼────────┤
│ MusicBERT-REMI   │ 3.511    │ 0.531  │ [2.66, 4.64]     │ 15.1% │
│ MusicBERT-TSD    │ 3.924    │ 0.791  │ [2.74, 5.47]     │ 20.2% │
│ CLaMP2-MTF       │ 0.0478   │ 0.0067 │ [0.0391, 0.0600] │ 14.0% │
│ CLaMP1-ABC       │ 0.0191   │ 0.0025 │ [0.0153, 0.0234] │ 12.8% │
└──────────────────┴──────────┴────────┴──────────────────┴────────┘

CV (Coefficient of Variation) = Std/Mean
→ CLaMP ma wyższy CV% ale bezwzględnie mniejszy rozsiew
→ MusicBERT bardziej zmienne bezwzględnie
```

---

## 🎨 Graficzna Reprezentacja Perturbacji

### Wrażliwość Każdego Modelu na Każdą Perturbację

```
8 ┤ MusicBERT-REMI
  ├╱╱╱╱╱╱╱╱╱╱╱╱╱╱╱╱╱╱╱╱╱╱╱╱╱╱╱ (no_velocity)
7 ├
  │
6 ├
  │
5 ├
  │
4 ├
  │                      MusicBERT-TSD
3 ├╱╱╱╱╱╱╱╱╱╱╱╱╱╱╱╱ (no_velocity)
  │
2 ├
  │
1 ├         CLaMP2-MTF
  ├╱╱╱╱╱╱╱╱╱╱╱╱╱╱╱╱╱╱╱╱╱╱ (wszystkie perturbacje)
  │
0 ├ CLaMP1-ABC ━━━━━━━━━━━━━━━━ (prawie płaska linia!)
  └──────────────────────────────────────────
    velocity quant_time constant_t all_combined
```

---

## 🧠 Interpretacja w Kontekście Badań

### Hybrydowe Modele Mogu Być Optymalne!

| Aspekt | MusicBERT | CLaMP | Idealne |
|--------|-----------|-------|---------|
| **Ekspresja** | ✅✅✅ Wysoka | ⚠️ Niska | ✅✅ Średnia |
| **Konsystencja** | ⚠️ Niska | ✅✅✅ Wysoka | ✅✅ Wysoka |
| **Generalizacja** | ⚠️ Zmienne | ✅ Stabilne | ✅✅ Stabilne + ekspresywne |
| **Struktura** | 🟡 Średnia | ✅ Bardzo dobra | ✅ Bardzo dobra |

---

## 📊 Ranking Modeli po Różnych Kryteriach

### Jeśli Chcesz: Najlepsza SPÓJNOŚĆ WEWNĘTRZNA
```
1. CLaMP1-ABC       (0.029 split-half FMD) ⭐⭐⭐⭐⭐
2. CLaMP2-MTF       (0.032 split-half FMD) ⭐⭐⭐⭐⭐
3. MusicBERT-REMI   (1.85 split-half FMD)  ⭐⭐
4. MusicBERT-TSD    (2.31 split-half FMD)  ⭐
```

### Jeśli Chcesz: EKSPRESJĘ (kodowanie dynamiki)
```
1. MusicBERT-REMI   (7.32 przy no_velocity) ⭐⭐⭐⭐⭐
2. MusicBERT-TSD    (3.13 przy no_velocity) ⭐⭐⭐⭐
3. CLaMP2-MTF       (0.046 przy no_velocity)⭐⭐
4. CLaMP1-ABC       (0.0103 przy no_velocity)⭐
```

### Jeśli Chcesz: STABILNE WYNIKI (niskie CV)
```
1. CLaMP1-ABC       (12.8% CV, niski std dev) ⭐⭐⭐⭐⭐
2. CLaMP2-MTF       (14.0% CV)               ⭐⭐⭐⭐⭐
3. MusicBERT-REMI   (15.1% CV)               ⭐⭐⭐
4. MusicBERT-TSD    (20.2% CV) ⚠️             ⭐⭐
```

### Jeśli Chcesz: GENERALIZUJE Między Datasetami
```
1. CLaMP1-ABC       (0.011 cross-dataset FMD) ⭐⭐⭐⭐⭐
2. CLaMP2-MTF       (0.035 cross-dataset FMD) ⭐⭐⭐⭐⭐
3. MusicBERT-REMI   (2.93 cross-dataset FMD)  ⭐⭐
4. MusicBERT-TSD    (3.23 cross-dataset FMD)  ⭐
```

---

## 🔬 Naukowe Obserwacje

### 1. Trade-off: Ekspresja vs Spójność
- **MusicBERT**: ↑ Ekspresja, ↓ Spójność
- **CLaMP**: ↓ Ekspresja, ↑ Spójność
- **Pytanie**: Który trade-off jest lepszy dla FMD?

### 2. Velocity Dominuje MusicBERT
```
MusicBERT wrażliwość:
- no_velocity:       7.3  (73% z 10)
- quantized_time:    0.1  (1% z 10) 
- constant_tempo:    0.15 (1.5% z 10)
- RAZEM (addytywnie): 7.55 ≈ 7.3

→ Velocity odpowiada za 96% zmienności!
```

### 3. CLaMP Jest Ortogonalny do Perturbacji
```
CLaMP wrażliwość:
- no_velocity:       0.01
- quantized_time:    0.01
- constant_tempo:    0.01
- RAZEM:            0.01

→ Perturbacje są prawie niezależne dla CLaMP!
  (Embedding prawie się nie zmienia)
```

### 4. Bootstrap Stability Sugeruje
- CLaMP: Wiarygodne, reproducible wyniki (12-14% CV)
- MusicBERT: Bardziej zmienne, ale bogatsze (15-20% CV)

---

## 🎓 Rekomendacje Praktyczne

### Dla Klasyfikacji Stylu Muzycznego
→ Użyj **CLaMP** (wysoka spójność, stabilne)

### Dla Interpretacji Ekspresji
→ Użyj **MusicBERT** (koduje dynamikę, rubato)

### Dla Production Systems
→ Użyj **CLaMP** (reproducible, stable)

### Dla Naukowych Badań
→ Rozważ **Hybrid**: CLaMP (struktura) + MusicBERT (ekspresja)

---

## 📝 Konsekwencje dla Publikacji

1. ✅ CLaMP jest wyraźnie lepszy do consistency metrics
2. ✅ MusicBERT koduje więcej ekspresji
3. ✅ Oba modele mają zastosowanie, zależy od zadania
4. ⚠️ Trade-off expression vs consistency jest fundamentalny
5. ✅ Perturbation tests potwierdza teoretyczne założenia

---

## 🔮 Przyszłe Kierunki

1. **Hybrid embeddings**: Połączyć CLaMP struktura z MusicBERT ekspresją
2. **Selective perturbations**: Testować bardziej granularne perturbacje
3. **Larger datasets**: Czy wyniki się skalują do 1000+ piosenek?
4. **Cross-genre**: Test na bardziej zróżnicowanych gatunkach
5. **Temporal analysis**: Jak zmienia się embedding w czasie w piosence?

---

**Data Analizy**: 2026-06-09  
**Autorstwo**: Sensitivity Pivot Analysis System  
**Status**: ✅ Kompletne

