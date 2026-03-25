# 🚀 INSTRUKCJA URUCHAMIANIA I TESTOWANIA PROJEKTU

## Co zostało zaimplementowane

✅ **Projekt:** Wrażliwość Frechet Music Distance na wybór tokenizacji i modelu embeddingów
✅ **Status:** Pełny Tydzień 1 - Gotowy do testowania
✅ **Testy:** 14/14 przechodzących (100% pass rate)
✅ **GitHub:** https://github.com/Michal2390/FMD-sensitivity-for-tokenization-and-embeddings

---

## 📋 SZYBKI START - 5 MINUT

### 1. **Weryfikacja instalacji**
```bash
python --version           # Python 3.12.2 ✓
pip list | head            # Sprawdź czy pakiety są zainstalowane
```

### 2. **Uruchomienie testów**
```bash
pytest tests/ -v           # Uruchomi 14 testów
```

**Oczekiwany wynik:**
```
14 passed in 1.05s ======= 14/14 testów przechodzących ✅
```

### 3. **Uruchomienie głównego programu**
```bash
python run_experiment.py --help        # Wyświetli pomoc
python run_experiment.py --experiment exp1_tokenization_sensitivity
```

### 4. **Sprawdzenie kodu**
```bash
black src/ tests/ --check              # Sprawdzenie formatowania
flake8 src/ tests/                     # Linting
```

### 5. **Uruchomienie demo**
```bash
python demo.py                         # Demo modułów
```

---

## 📊 CO PROGRAM ROBI

### Główne komponenty:

```
┌─────────────────────────────────────────────────────────────┐
│          WIMU FMD Sensitivity Analysis Project             │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  1. DATA MANAGER                                             │
│     └─ Zarządza zbiorami MIDI (MAESTRO, MidiCaps, POP909)  │
│                                                              │
│  2. PREPROCESSING                                           │
│     └─ Standaryzacja, kwantyzacja, filtracja MIDI          │
│                                                              │
│  3. TOKENIZATION (4 strategie)                             │
│     ├─ REMI          (Event-based, relative timing)        │
│     ├─ TSD           (Time-Shift-Duration)                 │
│     ├─ Octuple       (8-track representation)              │
│     └─ MIDI-Like     (Event-based, direct MIDI)            │
│                                                              │
│  4. EMBEDDINGS (2 modele)                                  │
│     ├─ CLaMP 1       (ABC format based)                    │
│     └─ CLaMP 2       (Direct MIDI structure)               │
│                                                              │
│  5. METRYKA FMD                                             │
│     └─ Frechet Music Distance (porównywanie embeddingów)   │
│                                                              │
│  6. EKSPERYMENTY (5 scenariuszy)                           │
│     ├─ Exp 1: Wpływ tokenizacji                           │
│     ├─ Exp 2: Wpływ modelu                                │
│     ├─ Exp 3: Ablation study (expresja)                   │
│     ├─ Exp 4: Kwantyzacja czasu                           │
│     └─ Exp 5: Stabilność międzygatunkowa                  │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## 🧪 TESTY - SZCZEGÓŁY

### Co jest testowane:

```
✅ test_fmd.py (7 testów)
   - test_fmd_identical_distributions  - FMD(x,x) = 0
   - test_fmd_different_distributions  - FMD > 0 dla różnych zbiorów
   - test_fmd_symmetry                 - FMD(x,y) = FMD(y,x)
   - test_fmd_1d_embeddings            - Obsługa 1D wektorów
   - test_fmd_matrix                   - Macierzowe obliczenia
   - test_ranking_by_fmd               - Ranking zbiorów
   - test_ranking_stability            - Stabilność rankingów

✅ test_data.py (7 testów)
   - test_dataset_manager_initialization - Inicjalizacja
   - test_get_dataset_path              - Ścieżki zbiorów
   - test_get_dataset_info              - Info z konfigu
   - test_list_midi_files               - Listowanie plików
   - test_data_processor_initialization - Inicjalizacja preprocessoru
   - test_validate_midi_file            - Walidacja MIDI
   - test_get_file_statistics           - Statystyki plików
```

---

## 🔧 KOMENDY MAKEFILE

```bash
make help                  # Wyświetl dostępne komendy
make install               # Instalacja zależności
make test                  # Uruchomienie testów
make lint                  # Sprawdzenie lintingu
make format                # Formatowanie kodu
make type-check            # Type checking (mypy)

# Uruchomienie eksperymentów
make run-exp1              # Tokenization sensitivity
make run-exp2              # Model sensitivity
make run-exp3              # Expression ablation
make run-exp4              # Quantization sensitivity
make run-exp5              # Cross-genre stability
make run-all               # Wszystkie eksperymenty
```

---

## 📁 STRUKTURA PLIKÓW

```
.
├── README.md                          # Główna dokumentacja
├── DESIGN_PROPOSAL.md                 # Design proposal projektu
├── TYDZIEN_1_PODSUMOWANIE.md         # Podsumowanie tygodnia 1
├── run_experiment.py                  # Główny skrypt eksperymentów
├── demo.py                            # Demo programu
│
├── configs/
│   └── config.yaml                    # Konfiguracja eksperymentów
│
├── src/                               # Kod główny
│   ├── utils/config.py               # Ładowanie config + logging
│   ├── data/manager.py               # Zarządzanie danymi
│   ├── preprocessing/processor.py    # Preprocessing MIDI
│   ├── tokenization/tokenizer.py     # 4 tokenizatory
│   ├── embeddings/extractor.py       # Ekstrakcja embeddingów
│   └── metrics/fmd.py                # Metryka FMD
│
├── tests/                             # Testy jednostkowe
│   ├── test_fmd.py                   # 7 testów dla FMD
│   └── test_data.py                  # 7 testów dla data managera
│
├── data/
│   ├── raw/                          # Oryginalne dane MIDI
│   ├── processed/                    # Przetworzone dane
│   └── embeddings/                   # Wyekstraktowane embeddingi
│
├── results/
│   ├── plots/                        # Wizualizacje
│   └── reports/                      # Raporty eksperymentów
│
├── logs/
│   └── experiment.log                # Logi eksperymentów
│
└── requirements.txt                   # Zależności Python
```

---

## 🎯 HARMONOGRAM - CO BĘDZIE DALEJ

| Tydzień | Zakres | Status |
|---------|--------|--------|
| 1 (23.03-29.03) | Setup, moduły, testy | ✅ UKOŃCZONY |
| 2 (30.03-05.04) | Preprocessing, tokenizacja | ⏳ Następny |
| 3 (06.04-12.04) | Integracja CLaMP | ⏳ Następny |
| 4 (13.04-19.04) | Kalkulacja FMD | ⏳ Następny |
| 5 (20.04-26.04) | Eksperymenty | ⏳ Następny |
| 6 (27.04-03.05) | Analiza, wizualizacje | ⏳ Następny |
| 7 (04.05-10.05) | Finalizacja, raport | ⏳ Następny |

---

## 💻 WYMAGANIA DO URUCHOMIENIA

```
✅ Python 3.10+          (✓ masz 3.12.2)
✅ PyTorch               (✓ zainstalowany 2.11.0)
✅ HuggingFace           (✓ zainstalowany)
✅ MidiTok               (✓ zainstalowany)
✅ NumPy/SciPy           (✓ zainstalowane)
✅ Pytest                (✓ zainstalowany)
```

---

## 📈 REZULTATY TESTÓW

```
============================= test session starts =============================
platform win32 -- Python 3.12.2, pytest-9.0.2

tests/test_fmd.py::TestFrechetMusicDistance
  ✅ test_fmd_identical_distributions      PASSED [57%]
  ✅ test_fmd_different_distributions      PASSED [64%]
  ✅ test_fmd_symmetry                     PASSED [71%]
  ✅ test_fmd_1d_embeddings                PASSED [78%]
  ✅ test_fmd_matrix                       PASSED [85%]

tests/test_fmd.py::TestFMDRanking
  ✅ test_ranking_by_fmd                   PASSED [92%]
  ✅ test_ranking_stability                PASSED [100%]

tests/test_data.py::TestDatasetManager
  ✅ test_dataset_manager_initialization   PASSED [7%]
  ✅ test_get_dataset_path                 PASSED [14%]
  ✅ test_get_dataset_info                 PASSED [21%]
  ✅ test_list_midi_files                  PASSED [28%]

tests/test_data.py::TestDataProcessor
  ✅ test_data_processor_initialization    PASSED [35%]
  ✅ test_validate_midi_file               PASSED [42%]
  ✅ test_get_file_statistics              PASSED [50%]

========================= 14 passed in 1.05s =================== 100% ✅
Coverage: 23% (będzie rosło)
```

---

## 🔗 LINKI

- **GitHub Repo:** https://github.com/Michal2390/FMD-sensitivity-for-tokenization-and-embeddings
- **README:** README.md
- **Design Proposal:** DESIGN_PROPOSAL.md
- **Podsumowanie Tygodnia 1:** TYDZIEN_1_PODSUMOWANIE.md

---

## ✨ CECHY PROJEKTU

✅ **100% testów przechodzących** - 14/14 testów
✅ **Kod sformatowany** - Black formatting
✅ **Zlinowany** - Flake8, ruff
✅ **Type-checked** - Mypy
✅ **Dobra dokumentacja** - README, docstringi, komentarze
✅ **Reprodukowalny** - requirements.txt, config.yaml
✅ **CI/CD Ready** - Git + GitHub
✅ **Profesjonalny** - PEP8, best practices

---

## 📞 WSPARCIE

Problem? Sprawdź:
1. `README.md` - Pełna dokumentacja
2. `DESIGN_PROPOSAL.md` - Opis projektu
3. `pytest tests/ -v` - Czy testy przechodzą
4. `python run_experiment.py --help` - Dostępne opcje

---

**Projekt Tydzień 1: GOTOWY DO TESTOWANIA! 🎉**

