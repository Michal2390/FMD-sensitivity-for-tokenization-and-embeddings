# Instrukcja uruchamiania (sensitivity pivot, czerwiec 2026)

Aktualna wersja projektu skupia się na **paper-grade sensitivity profiling** - profilowaniu wrażliwości FMD na tokenizację i konfigurację embeddingów bez mieszania tokenizerów z natywnymi wejściami CLaMP.

Poprawne konfiguracje pivotu:
- `MusicBERT-REMI`
- `MusicBERT-TSD`
- `CLaMP2-MTF`
- `CLaMP1-ABC`

Nie używaj starych wyników `CLaMP2-REMI`: CLaMP-2 nie konsumuje tokenów REMI jako natywnego wejścia.

Główny punkt wejścia: `main.py --mode sensitivity`  
Konfiguracja: `configs/sensitivity_pivot.yaml`

---

## 1. Wymagania

- Python **3.10+** (zalecane 3.11)
- PowerShell (Windows) lub bash (Linux/macOS)
- ~15 GB wolnego miejsca na dysku (datasety + modele HuggingFace)
- Połączenie z internetem przy pierwszym uruchomieniu
- **CPU:** pełny pipeline ~2 h | **GPU:** znacznie szybciej (opcjonalnie)

---

## 2. Instalacja środowiska

W katalogu projektu:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

Szybki smoke test (bez ciężkich modeli):

```powershell
python main.py --mode demo
pytest tests\ -q -k "not integration"
```

> Jeśli PowerShell blokuje aktywację venv:  
> `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`

---

## 3. Przygotowanie danych (4 datasety)

Pivot wymaga **prawdziwych** plików MIDI — nie używaj syntetycznych placeholderów.

### Krok A — MAESTRO + POP909 (automatyczne pobieranie)

```powershell
python main.py --mode fetch-data --datasets maestro pop909
```

Oczekiwany wynik:
- `data/raw/maestro/` — ~1276 plików `.midi`
- `data/raw/pop909/` — ~2898 plików `.mid`

### Krok B — Folk (Nottingham)

```powershell
python scripts\download_folk_dataset.py
```

Oczekiwany wynik: `data/raw/folk/` — ~1000+ plików `.mid`

### Krok C — MidiCaps classical (ręcznie)

1. Pobierz archiwum z [HuggingFace MidiCaps](https://huggingface.co/datasets/amaai-lab/MidiCaps) (`midicaps.tar.gz`)
2. Wypakuj do `data/raw/midicaps/`
3. Wygeneruj podzbiór classical:

```powershell
python -c "import sys; sys.path.insert(0,'src'); from utils.config import load_config; from data.midicaps_loader import MidiCapsGenreLoader; c=load_config('configs/config.yaml'); c['cross_validation']['midicaps']['genres']=['classical']; MidiCapsGenreLoader(c).populate_raw_datasets()"
```

Oczekiwany wynik: `data/raw/midicaps_classical/` — ~300 plików `.mid`

### Weryfikacja

```powershell
python -c "from pathlib import Path; ds=['maestro','pop909','folk','midicaps_classical']; [print(f'{d}: {len(list(Path(f\"data/raw/{d}\").rglob(\"*.mid*\")))}') for d in ds]"
```

Każdy dataset powinien mieć **≥ 80** plików (limit w configu).

---

## 4. Uruchomienie głównego eksperymentu

### Pełny pipeline (wszystkie kroki + wykresy)

```powershell
python main.py --mode sensitivity
```

Kroki wykonywane automatycznie:
1. Self-similarity sanity check
2. Cross-dataset ranking (6 par datasetów)
3. Perturbation sensitivity (5 perturbacji x poprawne konfiguracje)
4. Bootstrap stability
5. Generowanie wykresów

### Pojedyncze kroki (opcjonalnie)

```powershell
python main.py --mode sensitivity --sensitivity-step self-similarity
python main.py --mode sensitivity --sensitivity-step ranking
python main.py --mode sensitivity --sensitivity-step perturbation
python main.py --mode sensitivity --sensitivity-step bootstrap
python main.py --mode sensitivity --sensitivity-step plots
```

### Tylko wykresy (gdy CSV już istnieją)

```powershell
python main.py --mode sensitivity --sensitivity-step plots
```

---

## 5. Wyniki

### Raporty CSV/JSON

```
results/reports/sensitivity_pivot/
├── self_similarity.csv
├── cross_dataset_fmd.csv
├── spearman_ranking_agreement.csv
├── perturbation_sensitivity.csv
├── bootstrap_stability.csv
└── sensitivity_pivot_summary.json
```

### Wykresy

```
results/plots/sensitivity_pivot/
├── perturbation_heatmap.png
├── cross_dataset_bar.png
├── bootstrap_stability.png
└── self_similarity.png
```

### Opis wyników (PL)

`docs/SENSITIVITY_PIVOT_RESULTS.md`

---

## 6. Inne tryby (legacy / pomocnicze)

| Tryb | Komenda | Opis |
|:-----|:--------|:-----|
| Demo | `python main.py --mode demo` | Szybki test FMD |
| Testy | `python main.py --mode tests` | Pełny pytest |
| Pełny CI | `python main.py --mode full` | Testy + sensitivity pivot |
| Paper (stary) | `python main.py --mode paper` | Multi-model benchmark |
| Lakh (stary) | `python main.py --mode lakh` | Walidacja Lakh MIDI |
| Per-song | `python main.py --mode song --midi-file ścieżka.mid` | Analiza jednego utworu |

---

## 7. Typowe problemy

| Problem | Rozwiązanie |
|:--------|:------------|
| `python` nie działa | Użyj `py` zamiast `python` |
| Brak GPU / CUDA | Pipeline działa na CPU (wolniej) |
| CLaMP nie ładuje się | Sprawdź `pip install transformers torch mido music21` |
| Pusty dataset | Uruchom ponownie krok 3 (fetch-data / folk / midicaps) |
| `music21` błąd konwersji ABC | Upewnij się, że pliki MIDI są poprawne (nie puste) |
| Długi czas | Zmniejsz `max_files_per_dataset` w `configs/sensitivity_pivot.yaml` |

---

## 8. Czego NIE używać

Usunięte z repo (nieaktualne):

- `scripts/run_experiment.py` — stuby exp1–exp5
- `scripts/generate_pivot_plots.py` — stare etykiety (CLaMP2-ABC)
- `scripts/generate_starter_midis.py` — syntetyczne datasety
- `scripts/run_nfmd_analysis.py` — odrzucony kierunek nFMD
- Wykresy `fig1_*.png` … `fig6_*.png` — błędne etykiety

---

## 9. Szybka ściąga (copy-paste)

```powershell
# Od zera do wyników:
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -e .
python main.py --mode fetch-data --datasets maestro pop909
python scripts\download_folk_dataset.py
# (ręcznie: midicaps.tar.gz → data/raw/midicaps/ → komenda z kroku C)
python main.py --mode sensitivity
```

**Czas:** ~116 min na CPU przy 80 plikach/dataset (wyniki z 2026-06-08).
