# Instrukcja uruchamiania (czerwiec 2026)

Projekt bada wrażliwość Frechet Music Distance (FMD) na wybór reprezentacji
wejściowej (tokenizery MidiTok vs natywne MTF/ABC) i modelu embeddingów.

Konfiguracje badania (5):

| Konfiguracja | Model | Wejście |
|:-------------|:------|:--------|
| `MusicBERT-REMI` | MusicBERT | tokeny REMI (MidiTok) |
| `MusicBERT-TSD`  | MusicBERT | tokeny TSD (MidiTok) |
| `CLaMP2-MTF`     | CLaMP-2   | MIDI Text Format |
| `CLaMP1-ABC`     | CLaMP-1   | notacja ABC (własny renderer) |
| `CLaMP2-ABC`     | CLaMP-2   | notacja ABC - kontrola same-model |

Główny punkt wejścia: `main.py` - Konfiguracja: `configs/sensitivity_pivot.yaml`
Przewodnik po wynikach: `final_results.ipynb` (root repo)

---

## 1. Wymagania

- Python **3.10+** (zalecane 3.11)
- ~15 GB miejsca (datasety + modele HuggingFace przy pierwszym uruchomieniu)
- **GPU:** pełne badanie ~3-4 h | **CPU:** odpowiednio dłużej

## 2. Instalacja

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
python main.py --mode tests
```

> Jeśli PowerShell blokuje aktywację venv:
> `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`

## 3. Dane (6 datasetów po >=80 plikow MIDI)

Badanie używa: `maestro`, `pop909` oraz czterech podzbiorów gatunkowych
wyprowadzonych z Lakh MIDI: `classical`, `jazz`, `rock`, `rap`
(katalogi w `data/raw/<nazwa>/`).

```powershell
python main.py --mode fetch-data --datasets maestro pop909
```

Podzbiory gatunkowe przygotowuje `src/data/lakh_genre_loader.py` /
`midicaps_loader.py` (wymagają lokalnych archiwów Lakh/MidiCaps).
Weryfikacja liczebności:

```powershell
python -c "from pathlib import Path; ds=['maestro','pop909','classical','jazz','rock','rap']; [print(d, len(list(Path('data/raw',d).rglob('*.mid*')))) for d in ds]"
```

## 4. Uruchomienie badania

### Pełne badanie (wszystkie kroki, oba korpusy perturbacji)

```powershell
python scripts\run_full_study.py
```

Kroki: self-similarity -> cross-dataset ranking (15 par, Spearman) ->
perturbacje MAESTRO -> perturbacje POP909 (replikacja) -> bootstrap ->
analiza per-plik (paired + retest) na obu korpusach.
Wyniki sa checkpointowane po kazdej konfiguracji - przerwany run nie traci
ukonczonych komorek.

> Dluga sesja na Windows: uruchom jako proces odlaczony, np.
> `Start-Process python -ArgumentList "-u","scripts\run_full_study.py" -RedirectStandardOutput logs\run.log`

### Pojedyncze kroki

```powershell
python main.py --mode sensitivity                              # kroki 3-7 (maestro)
python main.py --mode sensitivity --sensitivity-step self-similarity
python main.py --mode sensitivity --sensitivity-step ranking
python main.py --mode sensitivity --sensitivity-step perturbation
python main.py --mode sensitivity --sensitivity-step paired
python main.py --mode sensitivity --sensitivity-step bootstrap
python main.py --mode sensitivity --sensitivity-step plots
```

### Tabele i figury do artykulu (z gotowych CSV, bez modeli)

```powershell
python scripts\generate_draft_figures.py
python scripts\generate_draft_tables.py
```

## 5. Wyniki

```
results/reports/sensitivity_pivot/
├── self_similarity.csv                  # progi szumu (split-half), 6 datasetow x 5 konfiguracji
├── cross_dataset_fmd.csv                # 15 par datasetow x 5 konfiguracji
├── spearman_ranking_agreement.csv       # zgodnosc rankingow (n=15, p-value)
├── perturbation_sensitivity.csv         # FMD, SNR, CI, permutacyjne p (maestro; *_pop909 = replikacja)
├── paired_file_shifts.csv               # przesuniecia per-plik + retest (maestro; *_pop909)
├── paired_file_tests.csv                # kontrasty Wilcoxona, korekta Holma (maestro; *_pop909)
├── bootstrap_stability.csv              # srednia, CI, CV
└── tables/*.tex                         # tabele LaTeX generowane do draft.tex

results/plots/sensitivity_pivot/         # wykresy robocze pipeline'u
results/plots/sensitivity_pivot/paper/   # figury publikacyjne (fig1-fig6)
```

Interpretacja wynikow: `final_results.ipynb` (krok po kroku) oraz
`docs/PAPER_FINDINGS.md` (mapa wynikow -> twierdzen artykulu).
Artykul: `draft.tex` (kompilacja: `pdflatex draft.tex` z korzenia repo).

## 6. Tryby pomocnicze

| Tryb | Komenda | Opis |
|:-----|:--------|:-----|
| Demo | `python main.py --mode demo` | szybki test FMD (bez modeli) |
| Testy | `python main.py --mode tests` | pelny pytest |
| Pelny CI | `python main.py --mode full` | testy + sensitivity |
| Fetch data | `python main.py --mode fetch-data` | pobranie datasetow |
| Lint | `python main.py --mode lint` | black + flake8 |

## 7. Typowe problemy

| Problem | Rozwiazanie |
|:--------|:------------|
| `python` nie dziala | uzyj `py` zamiast `python` |
| brak GPU / CUDA | pipeline dziala na CPU (wolniej) |
| CLaMP nie laduje sie | `pip install -r requirements.txt` (wymagane m.in. `unidecode`, `mido`) |
| pusty dataset | sprawdz `data/raw/<nazwa>/` i krok 3 |
| dlugi czas | zmniejsz `max_files_per_dataset` w `configs/sensitivity_pivot.yaml` |
| przerwany dlugi run | wyniki sa checkpointowane; uruchom ponownie brakujacy krok |
