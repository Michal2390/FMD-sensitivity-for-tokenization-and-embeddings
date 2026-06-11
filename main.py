"""Main entry point for the FMD sensitivity study.

Modes:
  sensitivity  run the full sensitivity profiling pipeline (default)
  full         unit tests, then the sensitivity pipeline
  fetch-data   download MIDI datasets from configured external sources
  tests        run the unit-test suite
  demo         lightweight sanity check (no model downloads required)
  lint         run black + flake8 quality checks
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

from tqdm import tqdm

# Make `src/` importable (the project is run from the repository root).
sys.path.insert(0, str(Path(__file__).parent / "src"))

from utils.config import get_logger, load_config, setup_logging


class FMDSensitivityAnalysis:
    """Runner for the FMD sensitivity study."""

    def __init__(self):
        self.config = load_config("configs/config.yaml")
        setup_logging(
            self.config["logging"].get("level", "INFO"),
            self.config["logging"].get("log_file", "logs/experiment.log"),
        )
        self.logger = get_logger(__name__)

    # ── tests / quality ────────────────────────────────────────────────
    def run_all_tests(self) -> bool:
        """Run the unit-test suite."""
        import pytest

        self.logger.info("Running unit tests")
        return pytest.main(["tests/", "-v", "--tb=short"]) == 0

    def run_lint_check(self):
        """Run code-quality checks (non-fatal)."""
        self.logger.info("Running code quality checks")
        black = subprocess.run(
            ["black", "src/", "tests/", "scripts/", "main.py", "--check", "--quiet"],
            capture_output=True,
        )
        self.logger.info("Black: OK" if black.returncode == 0 else "Black: formatting needed")
        flake = subprocess.run(
            ["flake8", "src/", "tests/", "scripts/", "main.py", "--count"],
            capture_output=True,
            text=True,
        )
        self.logger.info("Flake8: OK" if flake.returncode == 0 else f"Flake8 issues:\n{flake.stdout}")

    def run_demo(self):
        """Lightweight sanity check: config loads and FMD computes."""
        import numpy as np
        from metrics.fmd import FrechetMusicDistance

        self.logger.info(
            f"Config OK — {len(self.config['embeddings']['models'])} models, "
            f"{len(self.config['tokenization']['tokenizers'])} tokenizers configured"
        )
        fmd = FrechetMusicDistance(self.config)
        rng = np.random.default_rng(0)
        value = fmd.compute_fmd(rng.standard_normal((100, 64)), rng.standard_normal((100, 64)))
        self.logger.info(f"Demo FMD between two random Gaussians: {value:.4f}")

    # ── data ───────────────────────────────────────────────────────────
    def run_fetch_data(self, dataset_names: list[str] | None = None) -> bool:
        """Download MIDI datasets from configured external sources."""
        from data.manager import DatasetManager

        manager = DatasetManager(self.config)
        configured = [d["name"] for d in self.config.get("data", {}).get("datasets", [])]
        targets = dataset_names or configured
        self.logger.info(f"Fetching datasets: {targets}")

        ok_all = True
        total = max(1, len(targets))
        for idx, dataset_name in enumerate(
            tqdm(targets, desc="Fetch datasets", unit="dataset", dynamic_ncols=True), start=1
        ):
            self.logger.info(f"[Progress] fetch-data {idx}/{total} -> {dataset_name}")
            info = manager.get_dataset_info(dataset_name)
            url = str(info.get("url", "")).strip().lower()
            download = url.startswith("http")
            if not download:
                self.logger.warning(f"{dataset_name}: no downloadable URL configured")
            ok_all = manager.ensure_dataset_exists(dataset_name, download=download) and ok_all
        return ok_all

    # ── main experiment ────────────────────────────────────────────────
    def run_sensitivity_pivot(self, step: str | None = None):
        """Run the sensitivity profiling pipeline (optionally a single step)."""
        from experiments.sensitivity_profiler import SensitivityProfiler

        self.logger.info("Running sensitivity pipeline")
        profiler = SensitivityProfiler(self.config, "configs/sensitivity_pivot.yaml")

        steps = {
            "self-similarity": profiler.run_self_similarity,
            "ranking": profiler.run_cross_dataset_ranking,
            "perturbation": profiler.run_perturbation_sensitivity,
            "paired": profiler.run_paired_file_analysis,
            "bootstrap": profiler.run_bootstrap_stability,
            "plots": profiler.generate_plots,
        }
        if step is None:
            return profiler.run_all()
        if step not in steps:
            raise ValueError(f"Unknown step: {step}. Use one of {sorted(steps)}")
        return steps[step]()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="FMD sensitivity study runner")
    parser.add_argument(
        "--mode",
        choices=["sensitivity", "full", "fetch-data", "tests", "demo", "lint"],
        default="sensitivity",
        help="sensitivity (default), full (tests+sensitivity), fetch-data, tests, demo, lint",
    )
    parser.add_argument(
        "--datasets", nargs="*", default=None,
        help="Optional dataset-name subset for --mode fetch-data",
    )
    parser.add_argument(
        "--sensitivity-step", type=str, default=None,
        choices=["self-similarity", "ranking", "perturbation", "paired", "bootstrap", "plots"],
        help="Run a single step of the sensitivity pipeline (default: all)",
    )
    return parser


def main():
    args = build_arg_parser().parse_args()
    analysis = FMDSensitivityAnalysis()
    start = time.perf_counter()

    def _done(success: bool = True):
        status = "SUCCESS" if success else "FAILED"
        print(f"\n=== Program finished: {status} | elapsed: {time.perf_counter() - start:.1f}s ===")

    try:
        if args.mode == "sensitivity":
            analysis.run_sensitivity_pivot(step=args.sensitivity_step)
        elif args.mode == "fetch-data":
            if not analysis.run_fetch_data(args.datasets):
                _done(False)
                raise SystemExit(1)
        elif args.mode == "tests":
            if not analysis.run_all_tests():
                _done(False)
                raise SystemExit(1)
        elif args.mode == "demo":
            analysis.run_demo()
        elif args.mode == "lint":
            analysis.run_lint_check()
        elif args.mode == "full":
            if not analysis.run_all_tests():
                _done(False)
                raise SystemExit(1)
            analysis.run_sensitivity_pivot()
        _done(True)
    except Exception:
        _done(False)
        raise


if __name__ == "__main__":
    main()
