"""Run the complete sensitivity study end to end.

Steps: self-similarity, cross-dataset ranking (+Spearman), perturbation
sensitivity on MAESTRO and POP909, bootstrap stability, and the per-file
paired analysis (with the retest noise floor) on both corpora. Every step
checkpoints its CSV as it goes, so a crash never loses completed work.

Usage (from the repository root; survives session restarts when started
detached, e.g. via PowerShell Start-Process):
    python scripts/run_full_study.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from utils.config import load_config, setup_logging  # noqa: E402


def main() -> None:
    cfg = load_config(str(ROOT / "configs/config.yaml"))
    setup_logging(
        cfg["logging"].get("level", "INFO"),
        cfg["logging"].get("log_file", "logs/experiment.log"),
    )
    from experiments.sensitivity_profiler import SensitivityProfiler

    profiler = SensitivityProfiler(cfg, str(ROOT / "configs/sensitivity_pivot.yaml"))

    steps = [
        ("self-similarity", lambda: profiler.run_self_similarity()),
        ("cross-dataset ranking", lambda: profiler.run_cross_dataset_ranking()),
        ("perturbation maestro", lambda: profiler.run_perturbation_sensitivity("maestro")),
        ("perturbation pop909", lambda: profiler.run_perturbation_sensitivity("pop909")),
        ("bootstrap stability", lambda: profiler.run_bootstrap_stability()),
        ("paired maestro", lambda: profiler.run_paired_file_analysis("maestro")),
        ("paired pop909", lambda: profiler.run_paired_file_analysis("pop909")),
    ]
    for name, fn in steps:
        print(f"=== STEP: {name} ===", flush=True)
        fn()

    print("=== ALL DONE ===", flush=True)


if __name__ == "__main__":
    main()
