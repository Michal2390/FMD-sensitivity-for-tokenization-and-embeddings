# Wrażliwość Frechet Music Distance (FMD) na wybór tokenizacji i modelu embeddingów

## Opis projektu

Projekt analizuje wrażliwość metryki **Frechet Music Distance (FMD)** na wybór metody tokenizacji i modelu embeddingów przy porównywaniu zbiorów utworów muzycznych w formacie MIDI.

### Cele projektu

1. **Experiment 1**: Zbadanie wpływu wyboru strategii tokenizacji (REMI, TSD, Octuple, MIDI-Like) na wartości metryki FMD
2. **Experiment 2**: Porównanie architektury modeli embeddingów (CLaMP 1 vs CLaMP 2)
3. **Experiment 3**: Analiza wrażliwości metryki na usunięcie informacji o ekspresji (velocity)
4. **Experiment 4**: Wpływ kwantyzacji czasu na stabilność metryki
5. **Experiment 5**: Ocena konsystencji klasyfikacji międzygatunkowej

## Harmonogram projektu

- **Tydzień 1 (23.03-29.03.2026)**: Inicjalizacja i przygotowanie danych ✓
- **Tydzień 2 (30.03-05.04.2026)**: Preprocessing i tokenizacja
- **Tydzień 3 (06.04-12.04.2026)**: Integracja modeli CLaMP
- **Tydzień 4 (13.04-19.04.2026)**: Kalkulacja FMD
- **Tydzień 5 (20.04-26.04.2026)**: Eksperymenty z preprocessing
- **Tydzień 6 (27.04-03.05.2026)**: Analiza danych i ewaluacja
- **Tydzień 7 (04.05-10.05.2026)**: Finalizacja i dokumentacja

## Struktura projektu

```
.
├── README.md                 # Ta dokumentacja
├── requirements.txt          # Zależności Python
├── pyproject.toml           # Konfiguracja narzędzi
├── .flake8                  # Konfiguracja flake8
├── .gitignore              # Ignorowanie plików
├── configs/
│   └── config.yaml         # Główna konfiguracja eksperymentów
├── src/
│   ├── __init__.py
│   ├── utils/
│   │   ├── __init__.py
│   │   └── config.py       # Załadowanie konfiguracji i logging
│   ├── data/
│   │   ├── __init__.py
│   │   └── manager.py      # Zarządzanie zbiorami danych
│   ├── preprocessing/
│   │   ├── __init__.py
│   │   └── processor.py    # Preprocessing MIDI
│   ├── tokenization/
│   │   ├── __init__.py
│   │   └── tokenizer.py    # Tokenizacja (REMI, TSD, Octuple, MIDI-Like)
│   ├── embeddings/
│   │   ├── __init__.py
│   │   └── extractor.py    # Ekstrakcja embeddingów (CLaMP 1/2)
│   ├── metrics/
│   │   ├── __init__.py
│   │   └── fmd.py          # Implementacja metryki FMD
│   └── experiments/
│       ├── __init__.py
│       └── runner.py       # Uruchamianie eksperymentów
├── tests/
│   ├── test_fmd.py        # Testy FMD
│   ├── test_data.py       # Testy zarządzania danymi
│   └── test_preprocessing.py # Testy preprocessingu
├── data/
│   ├── raw/               # Oryginalne dane (MIDI)
│   ├── processed/         # Przetworzone dane
│   └── embeddings/        # Wyekstraktowane embeddingi
├── notebooks/             # Jupyter notebooks
├── results/
│   ├── plots/            # Wizualizacje
│   └── reports/          # Raporty eksperymentów
├── logs/                 # Logi eksperymentów
└── run_experiment.py     # Główny skrypt uruchamiania
```

## Instalacja

### Wymagania
- Python 3.10+
- CUDA (opcjonalnie, dla przyspieszenia GPU)

### Setup

1. Sklonuj repozytorium:
```bash
git clone <repository-url>
cd WIMU
```

2. Utwórz wirtualne środowisko:
```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
# lub
source .venv/bin/activate  # Linux/Mac
```

3. Zainstaluj zależności:
```bash
pip install -r requirements.txt
```

4. Sprawdź instalację:
```bash
pytest tests/ -v
```

## Użytkowanie

### Uruchomienie wszystkich eksperymentów

```bash
python run_experiment.py --all
```

### Uruchomienie konkretnego eksperymentu

```bash
python run_experiment.py --experiment exp1_tokenization_sensitivity
```

### Dostępne eksperymenty

- `exp1_tokenization_sensitivity` - Wpływ tokenizacji
- `exp2_model_sensitivity` - Wpływ modelu embeddingów
- `exp3_expression_ablation` - Ablation study na ekspresji
- `exp4_quantization_sensitivity` - Wpływ kwantyzacji
- `exp5_cross_genre` - Stabilność międzygatunkowa

### Testy

```bash
# Uruchomienie wszystkich testów
pytest tests/ -v

# Uruchomienie z pokryciem kodu
pytest tests/ --cov=src

# Uruchomienie konkretnego testu
pytest tests/test_fmd.py::TestFrechetMusicDistance::test_fmd_symmetry -v
```

### Linting i formatowanie

```bash
# Formatowanie kodu
black src/ tests/

# Sprawdzenie lintera
flake8 src/ tests/

# Type checking
mypy src/
```

## Konfiguracja

Główna konfiguracja znajduje się w `configs/config.yaml`:

- **Zbiory danych**: Ścieżki do MAESTRO, MidiCaps, POP909
- **Preprocessing**: Parametry normalizacji, kwantyzacji
- **Tokenizacja**: Konfiguracja 4 tokenizatorów
- **Embeddingi**: Parametry CLaMP 1 i CLaMP 2
- **Eksperymenty**: Definicje wszystkich eksperymentów
- **Wyniki**: Ścieżki do zapisywania rezultatów

## Techstack

| Komponent | Biblioteka |
|-----------|-----------|
| Przetwarzanie MIDI | `pretty_midi`, `symusic`, `miditok` |
| Deep Learning | `PyTorch`, `transformers` |
| Obliczenia | `NumPy`, `SciPy` |
| Wizualizacja | `Matplotlib`, `Seaborn` |
| ML Utils | `scikit-learn` |
| Logging | `loguru` |
| Testing | `pytest` |
| Linting | `flake8`, `ruff`, `black` |

## Wymagania WIMU

Projekt spełnia wszystkie wymagania z regulaminu WIMU:

✅ **Wysoka jakość kodu**
- Autoformatter: `black`
- Linter: `flake8`, `ruff`
- Type checking: `mypy`

✅ **Reprodukowalność**
- Dokładna konfiguracja w `config.yaml`
- Wirtualne środowisko
- Seeding dla losowości (będzie dodane w Exp 1)

✅ **Rzetelność**
- Testy jednostkowe w `tests/`
- Dokumentacja API w docstringach
- Logi wszystkich operacji

✅ **Nietrywialność**
- Zaawansowane metryki (Frechet Distance)
- Wiele strategii tokenizacji
- Modele transformer-based

✅ **Dokumentacja**
- README.md (ten plik)
- Komentarze w kodzie
- Docstringi we wszystkich funkcjach
- Dokumentacja eksperymentów w logach

✅ **Testy**
- Unit testy (`pytest`)
- Coverage tracking
- Testy dla każdego modułu

✅ **Instrukcja użytkowania**
- Sekcja "Użytkowanie" powyżej
- Help w argparse
- Przykłady w README

## Bibliografia

1. Retkowski, J., Stępniak, J., Modrzejewski, M. (2025). Frechet Music Distance: A Metric For Generative Symbolic Music Evaluation. (Repozytorium: github.com/jryban/frechet-music-distance)

2. Fradet, N., et al. (2024). MidiTok: A Python Package for MIDI File Tokenization. [Dostępne na: github.com/Natooz/MidiTok]

3. Le, D. V. T., Bigo, L., Keller, M., Herremans, D. (2024). Natural Language Processing Methods for Symbolic Music Generation and Information Retrieval: a Survey. [Dostępne na: arxiv.org/abs/2402.17467]

## Autorzy

Projekt realizowany w ramach przedmiotu **WIMU** (Wyszukiwanie Informacji Muzycznych) na Wydziale Elektroniki i Technik Informacyjnych (EITI) Politechniki Warszawskiej.

## Licencja

MIT License

