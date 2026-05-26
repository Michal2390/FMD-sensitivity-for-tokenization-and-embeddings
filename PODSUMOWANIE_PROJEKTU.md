# Podsumowanie projektu — FMD Sensitivity for Tokenization and Embeddings

Krótko: badanie czułości Frechet Music Distance (FMD) względem wyboru tokenizera i modelu embeddingów dla muzyki symbolicznej. Dokument zawiera streszczenie, najważniejsze tabelki z README oraz wybrane wykresy (najsilniejsze sygnały).

---

## Szybkie informacje

- Cel: Zbadać wpływ tokenizacji, modelu embeddingów i preprocessingu na wartość FMD.
- Tokenizery: REMI, TSD, Octuple, MIDI-Like
- Modele embeddingów (6): CLaMP-1, CLaMP-2, MusicBERT, MusicBERT-large, MERT, NLP-Baseline
- Preprocessingi: original, no velocity, hard quantization, combined
- Eksperyment: 96 wariantów × 6 par gatunków × 10 powtórzeń = 5760 obserwacji (Lakh MIDI, 6 par gatunków)
- Walidacja statystyczna: bootstrap CI, Holm–Bonferroni, permutation tests, Tukey HSD, cross-dataset (MidiCaps)

---

## Najważniejsze wnioski (skrócone)

- Wybór modelu embeddingów dominuje wariancję FMD (η² ≈ 0.96 w analizie 6-modelowej).
- Wrażliwość na tokenizację zależy od modelu (np. MusicBERT jest silnie wrażliwy, CLaMP mniej).
- Surowe wartości FMD są skalowo zależne od architektury modelu (norma wektorów różni się 3×), więc bez normalizacji porównania międzymodelowe są mylące.
- Normalizacja (nFMD — trace / norm / z‑score) znacznie redukuje efekt skali i uwidacznia wpływ tokenizera.
- MERT (audio SSL) zwraca degenerowane lub nieodpowiednie embeddingi dla symbolicznego MIDI — unikać dla tego zadania.

---

## Kluczowe tabelki (z README)

### Efekty (6-model analysis) — η² i interpretacja

| Source | η² | F | Interpretation |
|--------|-----:|----:|----------------|
| Model (main) | 0.9617 | 33904 | Dominant |
| Tokenizer × Model | 0.0034 | 40.40 | Negligible |
| Model × Preprocess | 0.0022 | 25.65 | Negligible |
| Tokenizer (main) | 0.0010 | 59.51 | Negligible |
| Preprocessing | 0.0010 | 60.14 | Negligible |

**Wniosek:** Model dominuje — tokenizer i preprocessing mają marginalne znaczenie.


### Hierarchia modeli wg czułości na rozróżnianie gatunków (Cohen's d vs CLaMP-2)

| Model | Architecture | Cohen's d vs CLaMP‑2 | Interpretation |
|-------|-------------|---------------------:|-----------------|
| MusicBERT-large | MLM (large) | −10.02 | Highest sensitivity |
| MusicBERT | MLM | −6.27 | High |
| NLP‑Baseline | Sentence encoder | −1.21 | Medium |
| CLaMP-1 | Contrastive | −1.55 | Low |
| CLaMP-2 | Contrastive | (reference) | Lowest |
| MERT | Audio SSL | 0.00 | ⚠️ Anomalous / defective on symbolic MIDI |

**Wniosek:** MusicBERT > NLP-Baseline > CLaMP; MERT niezdatny do MIDI.


### Zakresy FMD wg par gatunków (6-model)

| Genre Pair | Mean FMD | Std | N |
|------------|---------:|----:|---:|
| jazz ↔ country | 4.163 | 4.757 | 960 |
| rock ↔ jazz | 4.751 | 5.406 | 960 |
| rock ↔ electronic | 4.759 | 6.240 | 960 |
| rock ↔ country | 4.810 | 5.528 | 960 |
| jazz ↔ electronic | 5.396 | 7.029 | 960 |
| electronic ↔ country | 5.717 | 7.375 | 960 |

**Wniosek:** Różnice między parami (~37%) są mniejsze niż między modelami (~960%).


### Rekomendacje praktyczne (skrót)

| Cel | Rekomendowany pipeline | Rationale |
|-----|-----------------------|-----------|
| Najdrobniejsza rozdzielczość (niski baseline) | REMI + CLaMP-2 | Niskie nFMD, wysoka efektywna dimenzja, dobra separacja gatunków |
| Maksymalna separowalność gatunków | dowolny tokenizer + MusicBERT-large | Najwyższe absolutne FMD |
| Ocena wrażliwości tokenizera | REMI lub TSD + MusicBERT | η²(tok)=0.36 po nFMD — tokenizer ma znaczenie |
| Stabilność między gatunkami | Octuple + CLaMP-1 | Najniższe CV między parami |
| Porównania między modelami | Użyć nFMD_trace | Surowe FMD różnią się ~12.8× między modelami; nFMD zmniejsza do ~1.9× |
| Unikać | MIDI‑Like + MusicBERT; MERT dla symbolic MIDI |

**Wniosek:** Wybór modelu determinuje wydajność; tokenizer ma znaczenie tylko dla MusicBERT.


### Wpływ normalizacji (nFMD) — skrót

| Factor | η²(raw FMD) | η²(nFMD_trace) | η²(nFMD_norm) |
|--------|------------:|---------------:|--------------:|
| model | 0.9617 | 0.7079 | 0.6534 |
| tokenizer | 0.0010 | 0.0142 | 0.0414 |
| pair (genre) | 0.0067 | 0.0959 | 0.0066 |

**Wniosek:** nFMD_trace zmniejsza dominację modelu z 96% do 71% — tokenizer staje się widoczny.


### Wrażliwość tokenizera per model (nFMD_trace)

| Model | η²(tokenizer) | η²(preprocess) | Interpretacja |
|-------|--------------:|---------------:|---------------|
| MusicBERT | 0.3588 | 0.0111 | 🔴 Bardzo wrażliwy na tokenizer |
| MusicBERT-large | 0.1851 | 0.0241 | 🟡 Umiarkowanie wrażliwy |
| CLaMP-2 | 0.0839 | 0.0008 | 🟡 Mało wrażliwy |
| CLaMP-1 | 0.0605 | 0.0002 | 🟢 Niska wrażliwość |
| NLP-Baseline | 0.0184 | 0.1344 | 🟢 Wrażliwy raczej na preprocessing |
| MERT | 0.0055 | 0.0240 | 🟢 Niewrażliwy (audio-based, ignoruje tokeny) |

**Wniosek:** MusicBERT (36%) >> CLaMP-2 (8%) — architektura text-based wynosi.


---

## Najważniejsze wykresy (wersja czytelna)

Poniższe wykresy pokazują główne wnioski w przystępny sposób:

### 1) Co naprawdę wpływa na FMD?

![η² decomposition](results/plots/simple/01_eta_squared_effects.png)

Model dominuje z ~96% wariancji. Tokenizer i preprocessing to <1% — ale po normalizacji (nFMD_trace) tokenizer staje się widoczny!


### 2) Ranking modeli — który embedduje najlepiej?

![Model hierarchy](results/plots/simple/02_model_hierarchy.png)

MusicBERT-large prowadzi (FMD ≈ 11.7), potem MusicBERT (≈ 8.1). CLaMP-2 i CLaMP-1 znacznie poniżej (~2.5). MERT nie nadaje się do MIDI.


### 3) Różnice między parami gatunków

![Genre pair comparison](results/plots/simple/03_genre_pair_comparison.png)

Wszystkie pary są porównywalne (FMD 4.2–5.7). Dobór pary ma mały wpływ w stosunku do modelu.


### 4) Czułość tokenizera — zależy od modelu!

![Tokenizer sensitivity](results/plots/simple/04_tokenizer_sensitivity.png)

MusicBERT (η²=0.36) jest bardzo czuły na wybór tokenizera. CLaMP (~0.06) ignoruje go prawie całkowicie.


### 5) Jak normalizacja zmienia obraz?

![Normalization impact](results/plots/simple/05_normalization_impact.png)

Raw FMD kamufluje tokenizer (η²=0.001). nFMD_trace go ujawnia (η²=0.014). Dlatego normalizacja jest ważna!


### 6) Porównanie tokenizatorów (w kontekście nFMD)

![Tokenizer comparison](results/plots/simple/06_tokenizer_comparison.png)

REMI i TSD dają najniższe FMD (wąskie zakresy). MIDI-Like trochę wyżej. Ale efekt zależy od modelu (zob. wykres 4).


---


---

## Gdzie szukać pełnych wyników

- Raport nFMD: `results/reports/lakh_multi/NFMD_ANALYSIS_REPORT.md` (generowany przez `scripts/run_nfmd_analysis.py`).
- Pliki z wykresami: `results/plots/paper/` (wybrane powyżej).
- Dane tabelaryczne: `results/reports/lakh/variant_summary.csv`, `results/reports/lakh_multi/`.

---

## Szybki start (uruchomienie lokalne)

PowerShell (w katalogu projektu):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements-ci.txt
pip install -e .
python main.py --mode demo        # krótki test
python scripts\run_multi_genre_analysis.py   # pełne multi-genre (dłużej)
python scripts\run_nfmd_analysis.py          # nFMD analysis (~3h)
```

Pełna instrukcja uruchamiania: `docs/INSTRUKCJA_URUCHAMIANIA.md`.

---

## Proponowane następne kroki (porządkowanie repo)

1. Wyeksportować najważniejsze wykresy do `results/summary/` i zaktualizować odnośniki w tym pliku.
2. Dodać krótki skrypt generujący `docs/PODSUMOWANIE_PROJEKTU.md` automatycznie z najnowszych artefaktów (statyczny snapshot).
3. Oznaczyć modele, które zwracają degenerowane embeddingi (MERT) i dodać test jednostkowy sprawdzający wariancję embeddingów dla inputu MIDI.

---

Plik wygenerowany automatycznie: `PODSUMOWANIE_PROJEKTU.md` w katalogu głównym repozytorium — jeśli chcesz, mogę usunąć starą wersję w `docs/` lub zaktualizować odnośnienia w README.

---

##  Normalizacja FMD (nFMD) — szczegóły i zasada działania

Celem normalizacji jest usunięcie artefaktu skali wynikającego z różnej normy wektorów embeddingów dla różnych architektur, tak aby porównania między modelami odzwierciedlały faktyczne różnice w rozkładach, a nie tylko różnice w amplitudzie wektorów.

Zaimplementowane metody (w `src/metrics/fmd.py` — funkcja `compute_nfmd` i klasa `NormalizedFMD`):

1) Trace (nFMD_trace)

- Wzór: nFMD_trace = FMD / (Tr(Σ1) + Tr(Σ2))
- Intuicja: dzieli surowe FMD przez sumę śladów kowariancji (całkowitą wariancję) obu rozkładów embeddingów, kompensując różnice w wewnętrznym rozrzucie (scale/variance).
- Zastosowanie: rekomendowane do porównań międzymodelowych (użyte jako domyślna normalizacja w analizach porównawczych).

2) Norm (nFMD_norm)

- Wzór: nFMD_norm = FMD / (||μ1|| + ||μ2||)^2
- Intuicja: normalizuje przez kwadrat sumy norm średnich wektorów, co kompensuje fakt, że składnik ||μ1 - μ2||^2 skaluje kwadratowo z normami wektorów.
- Zastosowanie: przydatne, gdy problem wynika głównie z różnic w średnich (mean-scale).

3) Z-score (nFMD_z)

- Wzór: nFMD_z = (FMD - μ_baseline) / σ_baseline
- Intuicja: kalibruje FMD względem baseline'u (np. rozkładu FMD dla podziału tego samego gatunku), zwracając ile odchyleń standardowych wynosi obserwowana różnica.
- Wymaga: wcześniejszego obliczenia μ_baseline i σ_baseline (np. z wewnątrzgatunkowych porównań).

Aspekty implementacyjne i bezpieczeństwo numeryczne:

- Przy obliczaniu pierwiastka macierzowego (sqrtm) i pracą z kowariancjami stosowana jest niewielka regularizacja (epsilon) dodawana do diagonalnych elementów, by uniknąć niestabilności numerycznych.
- Jeśli mianownik normalizacji jest ekstremalnie mały, implementacja bezpiecznie zwraca 0.0 zamiast powodować błąd dzielenia przez zero.
- W przypadku problemów z obliczeniem sqrtm stosowany jest fallback oparty na wartości własnych (eigen-decomposition), co daje stabilne wyniki.

Rekomendacje praktyczne:

- Do porównań między różnymi architekturami embeddingów: stosować nFMD_trace.
- Do analizy wpływu tokenizera wewnątrz pojedynczego modelu: nFMD_norm lub nFMD_trace (oba ujawnią różne aspekty).
- Do kalibracji i interpretacji względnych efektów względem modelowego baseline'u: nFMD_z.

Więcej: implementacja i testy znajdują się w `src/metrics/fmd.py` (metody: `compute_nfmd`, `NormalizedFMD`, `compute_fmd_components`).