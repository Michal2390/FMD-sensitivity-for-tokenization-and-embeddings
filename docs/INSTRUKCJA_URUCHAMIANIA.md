# Instrukcja uruchamiania (aktualna, Windows/PowerShell)

Ten plik zawiera komendy dopasowane do aktualnej struktury repozytorium (`main.py`, `scripts\*.py`).

## 1. Wymagania

- Python **3.10+**
- PowerShell
- Połączenie z internetem (pierwsze pobranie modeli/danych)

## 2. Szybki start (zalecany)

Uruchom w katalogu projektu:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements-ci.txt
pip install -e .
```

Następnie szybki smoke test:

```powershell
python main.py --mode demo
pytest tests\test_fmd.py -q
```

## 3. Pełne uruchomienie testów

```powershell
pytest tests\ -v
```

albo przez entry point:

```powershell
python main.py --mode tests
```

## 4. Najważniejsze tryby uruchamiania

```powershell
python main.py --mode quick
python main.py --mode paper
python main.py --mode paper-full
python main.py --mode paper-plots
python main.py --mode fetch-data
python main.py --mode lakh
python main.py --mode lakh-plots
```

## 5. Uruchamianie skryptów eksperymentalnych

```powershell
python scripts\run_multi_genre_analysis.py
python scripts\run_nfmd_analysis.py
python scripts\run_sample_size_ablation.py
python scripts\run_cross_dataset_validation.py --source midicaps
python scripts\run_cross_dataset_validation.py --source cd1
```

## 6. Pełne zależności badawcze (opcjonalnie)

Jeśli potrzebujesz pełnego środowiska badawczego (np. audio/MERT), zamiast `requirements-ci.txt` użyj:

```powershell
pip install -r requirements.txt
pip install -e .
```

> Uwaga: część funkcji audio może wymagać dodatkowych narzędzi systemowych (np. FluidSynth).

## 7. Typowe problemy

1. Brak aktywacji venv w PowerShell:
   `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`
2. Brak komendy `python`:
   użyj `py` zamiast `python`.
3. Problemy z zależnościami ciężkimi:
   zacznij od `requirements-ci.txt`, potem przejdź na `requirements.txt`.

## 8. Nieaktualne komendy (nie używać)

W tym repo nie używaj:

- `python run_experiment.py ...` (plik jest w `scripts\`)
- `python demo.py` (użyj `python main.py --mode demo` lub `python scripts\demo.py`)
- `make run-exp1` itp. (brak takich targetów w aktualnym Makefile)

