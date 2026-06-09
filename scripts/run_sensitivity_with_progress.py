"""Run sensitivity pipeline with early progress messages for overnight jobs."""

from __future__ import annotations

import sys
import time
from pathlib import Path


def log(message: str) -> None:
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {message}", flush=True)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    sys.path.insert(0, str(root / "src"))
    sys.path.insert(0, str(root / "scripts"))

    log("Starting sensitivity pipeline wrapper")
    start = time.perf_counter()
    log("Importing main.FMDSensitivityAnalysis")
    from main import FMDSensitivityAnalysis

    log(f"Imported main module in {time.perf_counter() - start:.1f}s")
    log("Constructing FMDSensitivityAnalysis")
    analysis = FMDSensitivityAnalysis()
    log(f"Constructed analysis in {time.perf_counter() - start:.1f}s total")
    log("Starting run_sensitivity_pivot()")
    analysis.run_sensitivity_pivot()
    log(f"Sensitivity pipeline finished in {time.perf_counter() - start:.1f}s")


if __name__ == "__main__":
    main()
