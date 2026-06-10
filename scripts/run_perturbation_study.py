"""Run the perturbation study: MAESTRO (primary, incl. CLaMP2-ABC control)
plus the POP909 replication. Results are checkpointed after every
configuration, so a crash never loses completed cells.

Usage (from the repository root):
    python scripts/run_perturbation_study.py
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

    print("=== PERTURBATION: maestro (primary, incl. CLaMP2-ABC control) ===", flush=True)
    profiler.run_perturbation_sensitivity("maestro")

    print("=== PERTURBATION: pop909 (replication) ===", flush=True)
    profiler.run_perturbation_sensitivity("pop909")

    print("=== ALL DONE ===", flush=True)


if __name__ == "__main__":
    main()
