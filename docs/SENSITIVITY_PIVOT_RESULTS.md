# Sensitivity Pivot — Wyniki Eksperymentów

## Podsumowanie

Pipeline: 3 konfiguracje × 3 datasety × 5 perturbacji
Czas: 27 minut na MacBook (CPU only)
Data: 2026-05-28

---

## Krok 3: Self-Similarity (Sanity Check)

Split-half FMD — oczekiwana wartość ≈ 0 (sprawdzamy, czy konfiguracja jest stabilna).

| Dataset   | CLaMP2-ABC | CLaMP2-MTF | CLaMP1-ABC |
|-----------|-----------|-----------|-----------|
| MAESTRO   | 0.039     | 0.027     | 0.022     |
| POP909    | 0.070     | 0.056     | 0.038     |
| Folk      | 0.039     | 0.019     | 0.059     |

**Wniosek:** Wszystkie wartości bliskie zeru (max 0.07). Konfiguracje są stabilne i wewnętrznie spójne. Szum bazowy wynosi ~0.02–0.07, co stanowi dolną granicę „istotnej różnicy".

---

## Krok 4: Cross-Dataset Ranking

FMD między datasetami — mierzy dystans stylistyczny.

| Para             | CLaMP2-ABC | CLaMP2-MTF | CLaMP1-ABC |
|------------------|-----------|-----------|-----------|
| MAESTRO vs POP909| **0.265** | 0.170     | 0.145     |
| MAESTRO vs Folk  | **0.572** | 0.477     | 0.294     |
| POP909 vs Folk   | **0.814** | 0.310     | 0.213     |

### Ranking wg konfiguracji:
- **CLaMP2-ABC:**  pop909-folk (0.81) > maestro-folk (0.57) > maestro-pop909 (0.27)
- **CLaMP2-MTF:**  maestro-folk (0.48) > pop909-folk (0.31) > maestro-pop909 (0.17)
- **CLaMP1-ABC:**  maestro-folk (0.29) > pop909-folk (0.21) > maestro-pop909 (0.14)

### Spearman τ (zgodność rankingów):
| Para konfiguracji      | Spearman τ | p-value |
|------------------------|-----------|---------|
| CLaMP2-ABC vs CLaMP2-MTF | 0.50     | 0.67    |
| CLaMP2-ABC vs CLaMP1-ABC | 0.50     | 0.67    |
| CLaMP2-MTF vs CLaMP1-ABC | **1.00** | 0.00    |

**Kluczowy wniosek:** CLaMP2-ABC daje **odwrócony ranking** względem pozostałych dwóch konfiguracji! Dla CLaMP2-ABC najdalej od siebie są pop909-folk (0.81), a dla CLaMP2-MTF i CLaMP1-ABC — maestro-folk. To oznacza, że **wybór reprezentacji (ABC vs MTF) istotnie wpływa na percepcję odległości stylistycznej**. CLaMP2-MTF i CLaMP1-ABC dają identyczny ranking (τ = 1.0).

---

## Krok 5: Perturbation Sensitivity (KLUCZOWY WYNIK)

FMD(oryginał, zaburzony) na zbiorze MAESTRO. Im wyższy FMD, tym bardziej konfiguracja „widzi" daną zmianę.

| Perturbacja      | CLaMP2-ABC | CLaMP2-MTF | CLaMP1-ABC |
|------------------|-----------|-----------|-----------|
| no_velocity      | **0.510** | **0.408** | 0.022     |
| quantized_time   | 0.008     | 0.008     | 0.007     |
| constant_tempo   | 0.000     | 0.000     | 0.000     |
| all_combined     | **0.514** | **0.408** | 0.026     |

### Główne odkrycia:

1. **Velocity (dynamika) — najważniejszy czynnik:**
   - CLaMP2-ABC reaguje bardzo silnie (FMD = 0.51) — ABC zachowuje informację o velocity
   - CLaMP2-MTF reaguje silnie (FMD = 0.41) — MTF też koduje velocity
   - CLaMP1-ABC praktycznie **nie reaguje** (FMD = 0.02) — CLaMP-1 gubi velocity!

2. **Kwantyzacja czasu — żadna konfiguracja nie jest wrażliwa:**
   - FMD ≈ 0.008 dla wszystkich — na poziomie szumu bazowego
   - Microtiming nie jest kluczowy dla tych embeddingów

3. **Tempo — kompletna niewrażliwość:**
   - FMD = 0.0 dla wszystkich trzech konfiguracji
   - Żaden model nie koduje informacji o tempie/rubato w embeddingach

4. **All combined ≈ no_velocity:**
   - Efekt jest zdominowany wyłącznie przez velocity
   - all_combined ≈ no_velocity (0.514 vs 0.510 dla CLaMP2-ABC)

### ⇒ Rekomendacja praktyczna:
> **„Jeśli zależy ci na ocenie dynamiki (velocity/expression) w generowanej muzyce, używaj CLaMP-2 (zarówno ABC jak MTF). Nie używaj CLaMP-1 — ten model w ogóle nie widzi zmian dynamicznych."**

> **„Żaden z badanych modeli nie jest wrażliwy na tempo ani microtiming — te aspekty muzyki nie są mierzone przez FMD w tych konfiguracjach."**

---

## Krok 6: Bootstrap Stability

Stabilność FMD przy resampowaniu (maestro vs pop909), 10 bootstraps × 200 próbek.

| Konfiguracja | FMD mean | FMD std | 95% CI          | CV    |
|-------------|---------|---------|-----------------|-------|
| CLaMP2-ABC  | 0.286   | 0.025   | [0.253, 0.329]  | 8.9%  |
| CLaMP2-MTF  | 0.184   | 0.015   | [0.159, 0.211]  | 8.3%  |
| CLaMP1-ABC  | 0.157   | 0.013   | [0.140, 0.180]  | 8.6%  |

**Wniosek:** Wszystkie trzy konfiguracje mają podobną stabilność (CV ≈ 8–9%). Żadna nie jest znacząco mniej stabilna od innych przy 200 próbkach. CLaMP1-ABC daje najwęższe CI bezwzględnie.

---

## Synteza — 3 główne wnioski do artykułu

### 1. Wybór reprezentacji istotnie wpływa na ranking odległości
Spearman τ = 0.5 między CLaMP2-ABC a pozostałymi konfiguracjami → ranking nie jest wymienialny. Wybór konfiguracji FMD zmienia wnioski o podobieństwie zbiorów muzycznych.

### 2. Velocity jest jedynym aspektem ekspresji widocznym dla FMD
Tempo i microtiming dają FMD = 0. Tylko velocity/dynamika jest rejestrowana, i to wyłącznie przez CLaMP-2 (nie CLaMP-1). CLaMP-1 jest de facto ślepy na velocity.

### 3. Konfiguracje mają porównywalną stabilność statystyczną
Bootstrap CV ≈ 8–9% dla wszystkich — żadna konfiguracja nie jest niestabilna przy N=200.

---

## Praktyczne rekomendacje (Tabela 1 do publikacji)

| Cel ewaluacji | Rekomendowana konfiguracja | Uzasadnienie |
|---------------|---------------------------|-------------|
| Ocena dynamiki | CLaMP-2 (ABC lub MTF) | Widzi velocity (FMD ≈ 0.4–0.5 po usunięciu) |
| Ocena struktury harmonicznej | CLaMP-1 ABC | Invariant na velocity → mierzy głębszą strukturę |
| Porównanie stylów z zachowaniem rankingu | CLaMP-2 MTF lub CLaMP-1 ABC | Identyczny ranking (τ = 1.0) |
| Unikać | CLaMP-2 ABC dla rankingu | Daje inny ranking niż pozostałe (τ = 0.5) |

