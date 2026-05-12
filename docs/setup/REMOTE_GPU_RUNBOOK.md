# Remote GPU runbook — Lakh validation (96 variants)

## Cel
Uruchomić pełny przebieg `lakh` na mocniejszej maszynie GPU dla aktualnej gałęzi roboczej:

- branch: `feature/publication-improvements`
- entry point: `python -u main.py --mode lakh`
- outputs: `results/reports/lakh/` oraz `results/plots/paper/`

## Ważne: nie wznawiamy lokalnego procesu
Aktualny lokalny proces był uruchomiony bez mechanizmu checkpoint/resume.
Na zdalnej maszynie należy **zacząć od nowa**, a nie próbować kontynuować lokalnego PID.

## Co dokładnie ma zrobić kolejna sesja Copilota / operator
1. Przejść do repozytorium i pobrać branch `feature/publication-improvements`.
2. Upewnić się, że środowisko ma GPU i poprawne sterowniki dla PyTorch.
3. Zainstalować zależności.
4. Uruchomić **tylko jeden** proces `main.py --mode lakh`.
5. Monitorować postęp po markerach `[Lakh X/96]` w `logs/experiment.log`.
6. Po zakończeniu streścić wyniki z `results/reports/lakh/ANALYSIS_REPORT.md` oraz najważniejsze CSV.

## Zalecane komendy
```bash
cd /path/to/FMD-sensitivity-for-tokenization-and-embeddings

git fetch origin

git checkout feature/publication-improvements

git pull --ff-only origin feature/publication-improvements

python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
pip install -e .

python - <<'PY'
import torch
print('cuda_available=', torch.cuda.is_available())
print('device_count=', torch.cuda.device_count())
if torch.cuda.is_available():
    print('device_name=', torch.cuda.get_device_name(0))
PY
```

## Uruchomienie w `tmux`
```bash
tmux new -s fmd-lakh
cd /path/to/FMD-sensitivity-for-tokenization-and-embeddings
source .venv/bin/activate
python -u main.py --mode lakh | tee logs/lakh_gpu_run.log
```

## Monitoring
```bash
cd /path/to/FMD-sensitivity-for-tokenization-and-embeddings

grep -n '\[Lakh ' logs/experiment.log | tail -n 10

tail -n 40 logs/experiment.log

ps -o pid,etime,time,%cpu,%mem,stat,command -C python3 | cat
```

## Oczekiwany zakres pracy
Tryb `lakh` w tej gałęzi buduje **96 wariantów**:

- 4 tokenizery
- 6 modeli embeddingów
- 4 konfiguracje preprocessingu

Kod w `src/experiments/paper_pipeline.py` wykonuje dla każdego wariantu:
- preprocessing MIDI,
- tokenizację,
- ekstrakcję embeddingów,
- obliczenie FMD,
- bootstrap CI,
- na końcu analizę wrażliwości i wykresy.

## Gdzie patrzeć po zakończeniu
Najpierw sprawdzić:

- `results/reports/lakh/ANALYSIS_REPORT.md`
- `results/reports/lakh/lakh_pairwise_fmd.csv`
- `results/reports/lakh/anova_table.csv`
- `results/reports/lakh/variant_summary.csv`
- `results/reports/lakh/lakh_validation_summary.json`

## Dodatkowe uwagi operacyjne
- Nie uruchamiać dwóch równoległych procesów `main.py --mode lakh` na tym samym workspace.
- Jeśli log długo nie rośnie, sprawdzić CPU/GPU usage zanim uzna się proces za zawieszony.
- W razie potrzeby można uruchomić najpierw smoke check:

```bash
python scripts/run_same_song_bridge_validation.py --help
python scripts/run_embedding_architecture_audit.py --help
pytest tests/test_embedding_architecture_audit.py tests/test_fmd.py -q
```

## Kontekst aktualnego stanu (2026-05-12)
Lokalnie proces został uruchomiony dwukrotnie; późniejszy duplikat został zatrzymany.
W momencie przygotowania tego handoffu pojedynczy proces doszedł do około `Lakh 3/96`.
Ten stan nie jest checkpointem do wznowienia — na zdalnej maszynie należy wykonać świeży przebieg.

