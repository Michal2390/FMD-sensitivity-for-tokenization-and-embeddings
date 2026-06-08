# Sensitivity Pivot — Wyniki (prawdziwe datasety, 2026-06-08)

## Źródła danych

| Dataset | Plików | Źródło |
|---------|--------|--------|
| **maestro** | 1276 | [MAESTRO v3](https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0/maestro-v3.0.0-midi.zip) |
| **pop909** | 2898 | [POP909-Dataset](https://github.com/music-x-lab/POP909-Dataset) |
| **folk** | 1034 | [Nottingham Dataset](https://github.com/jukedeck/nottingham-dataset) |
| **midicaps_classical** | 300 | [MidiCaps](https://huggingface.co/datasets/amaai-lab/MidiCaps) — tag `classical` |

W eksperymencie: **80 losowych plików** per dataset (seed=42).

## Konfiguracje

- **CLaMP2-MTF** — MIDI → MTF (`mido`) → M3 patches
- **CLaMP2-REMI** — MidiTok REMI → text patches
- **CLaMP1-ABC** — MIDI → ABC (`music21`) → bar patches

---

## Krok 3: Self-Similarity

| Dataset | CLaMP2-MTF | CLaMP2-REMI | CLaMP1-ABC |
|---------|-----------|------------|-----------|
| MAESTRO | 0.017 | 0.010 | 0.029 |
| POP909 | 0.024 | 0.010 | 0.033 |
| Folk | 0.094 | 0.011 | 0.026 |
| MidiCaps classical | 0.129 | 0.039 | 0.034 |

Wszystkie < 0.15 — konfiguracje stabilne.

---

## Krok 4: Cross-Dataset Ranking (6 par)

| Para | CLaMP2-MTF | CLaMP2-REMI | CLaMP1-ABC |
|------|-----------|------------|-----------|
| MAESTRO–POP909 | 0.073 | 0.051 | 0.017 |
| MAESTRO–Folk | **0.729** | 0.066 | 0.015 |
| MAESTRO–Classical | 0.272 | 0.077 | 0.015 |
| POP909–Folk | **0.624** | 0.102 | 0.016 |
| POP909–Classical | 0.219 | 0.089 | 0.015 |
| Folk–Classical | 0.472 | 0.084 | 0.014 |

### Spearman τ (n=6)

| Para konfiguracji | τ |
|-------------------|---|
| CLaMP2-MTF vs CLaMP2-REMI | 0.26 |
| CLaMP2-MTF vs CLaMP1-ABC | −0.37 |
| CLaMP2-REMI vs CLaMP1-ABC | −0.09 |

MTF widzi duże dystanse stylistyczne (MAESTRO–Folk 0.73); ABC/REMI spłaszczają ranking (~0.015).

---

## Krok 5: Perturbation Sensitivity (MAESTRO)

| Perturbacja | CLaMP2-MTF | CLaMP2-REMI | CLaMP1-ABC |
|-------------|-----------|------------|-----------|
| no_velocity | **0.063** | 0.053 | 0.020 |
| quantized_time | **0.118** | 0.000 | 0.014 |
| constant_tempo | 0.000 | 0.000 | 0.015 |
| all_combined | **0.207** | 0.055 | 0.016 |

**Wnioski:**
- MTF najsilniej reaguje na velocity i kwantyzację czasu na prawdziwym MAESTRO
- REMI reaguje na velocity, nie na timing
- ABC praktycznie niewrażliwe na wszystkie perturbacje (strata informacji w konwersji)

---

## Krok 6: Bootstrap (MAESTRO vs POP909, n=50)

| Konfiguracja | FMD mean ± std | 95% CI |
|-------------|---------------|--------|
| CLaMP2-MTF | 0.080 ± 0.004 | [0.073, 0.088] |
| CLaMP2-REMI | 0.055 ± 0.003 | [0.050, 0.062] |
| CLaMP1-ABC | 0.027 ± 0.003 | [0.021, 0.031] |

---

## Synteza

1. Na prawdziwych danych MTF daje wyraźniejszą separację stylistyczną i wrażliwość na perturbacje.
2. MAESTRO–Folk FMD = 0.73 (MTF) potwierdza duży kontrast klasyczny vs folk.
3. Kwantyzacja czasu na MAESTRO jest wykrywalna przez MTF (FMD=0.12) — na syntetykach było ~0.02.
4. Rekomendacja: **CLaMP2+MTF** do oceny MIDI z dynamiką i timingiem; **CLaMP1+ABC** tylko do struktury partytury.

**Czas uruchomienia:** ~116 min (CPU, 80 plików/dataset).
