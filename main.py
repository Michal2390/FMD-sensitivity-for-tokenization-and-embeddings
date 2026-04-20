"""Main entry point with one-click modes for research workflow."""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path
from tqdm import tqdm

# Add src and scripts to path
sys.path.insert(0, str(Path(__file__).parent / "src"))
sys.path.insert(0, str(Path(__file__).parent / "scripts"))

from experiments.paper_pipeline import PaperExperimentRunner
from experiments.publication_plots import generate_publication_plots
from data.manager import DatasetManager
from run_experiment import ExperimentRunner
from utils.config import get_logger, load_config, setup_logging


class FMDSensitivityAnalysis:
    """Main class for FMD sensitivity analysis."""

    def __init__(self):
        """Initialize the analysis."""
        self.config = load_config("configs/config.yaml")
        setup_logging(
            self.config["logging"].get("level", "INFO"),
            self.config["logging"].get("log_file", "logs/experiment.log"),
        )
        self.logger = get_logger(__name__)

    def run_all_tests(self) -> bool:
        """Run all unit tests."""
        import pytest

        self.logger.info("Running all unit tests")
        result = pytest.main(["tests/", "-v", "--tb=short"])
        return result == 0

    def run_demo(self):
        """Run lightweight sanity demo."""
        import numpy as np
        from data.manager import DatasetManager
        from metrics.fmd import FrechetMusicDistance

        self.logger.info("Running demo")
        self.logger.info(f"Experiments in config: {len(self.config['experiments'])}")
        self.logger.info(f"Tokenizers in config: {len(self.config['tokenization']['tokenizers'])}")
        self.logger.info(f"Embedding models in config: {len(self.config['embeddings']['models'])}")

        manager = DatasetManager(self.config)
        self.logger.info(f"Dataset manager ready at: {manager.raw_data_dir}")

        fmd_calc = FrechetMusicDistance(self.config)
        emb1 = np.random.randn(100, 64)
        emb2 = np.random.randn(100, 64)
        value = fmd_calc.compute_fmd(emb1, emb2)
        self.logger.info(f"Demo FMD value: {value:.4f}")

    def run_experiment(self, experiment_name: str):
        """Run a specific config experiment."""
        self.logger.info(f"Running experiment: {experiment_name}")
        runner = ExperimentRunner("configs/config.yaml")
        runner.run_experiment(experiment_name)

    def run_all_experiments(self):
        """Run all enabled config experiments."""
        self.logger.info("Running all enabled experiments")
        runner = ExperimentRunner("configs/config.yaml")
        runner.run_all_experiments()

    def run_lint_check(self):
        """Run code quality checks."""
        self.logger.info("Running code quality checks")

        black = subprocess.run(
            ["black", "src/", "tests/", "run_experiment.py", "--check", "--quiet"],
            capture_output=True,
        )
        self.logger.info("Black: OK" if black.returncode == 0 else "Black: formatting needed")

        flake = subprocess.run(
            ["flake8", "src/", "tests/", "run_experiment.py", "--count"],
            capture_output=True,
            text=True,
        )
        self.logger.info("Flake8: OK" if flake.returncode == 0 else f"Flake8 issues:\n{flake.stdout}")

    def run_paper_benchmark(self, full: bool = False):
        """Run paper-oriented comparison benchmark and report generation."""
        runner = PaperExperimentRunner(self.config)
        if full:
            self.logger.info("Running full paper benchmark")
            result = runner.run_full()
        else:
            self.logger.info("Running quick paper benchmark")
            result = runner.run_quick()

        self.logger.info(f"Paper benchmark done. Rows: {result.get('pairwise_rows')}")
        for label, path in result.get("outputs", {}).items():
            self.logger.info(f"Output {label}: {path}")

        plots_cfg = self.config.get("paper", {}).get("publication_plots", {})
        if plots_cfg.get("enabled", True):
            self.run_publication_plots()
        return result

    def run_publication_plots(self):
        """Generate publication-ready plots from latest paper outputs."""
        self.logger.info("Generating publication plots")
        outputs = generate_publication_plots(self.config)
        if not outputs:
            self.logger.warning("No publication plots generated")
            return outputs

        for label, path in outputs.items():
            self.logger.info(f"Plot {label}: {path}")
        return outputs

    def run_fetch_data(self, dataset_names: list[str] | None = None) -> bool:
        """Download MIDI datasets from configured external sources."""
        manager = DatasetManager(self.config)
        configured = [d["name"] for d in self.config.get("data", {}).get("datasets", [])]
        targets = dataset_names or configured
        self.logger.info(f"Fetching datasets: {targets}")

        ok_all = True
        total = max(1, len(targets))
        for idx, dataset_name in enumerate(
            tqdm(targets, desc="Fetch datasets", unit="dataset", dynamic_ncols=True),
            start=1,
        ):
            pct = 100.0 * idx / total
            self.logger.info(f"[Progress] fetch-data {idx}/{total} ({pct:.1f}%) -> {dataset_name}")
            info = manager.get_dataset_info(dataset_name)
            url = str(info.get("url", "")).strip().lower()
            if not url.startswith("http"):
                self.logger.warning(f"Skipping {dataset_name}: no external downloadable URL configured ({url})")
                ok = manager.ensure_dataset_exists(dataset_name, download=False)
            else:
                ok = manager.ensure_dataset_exists(dataset_name, download=True)
            ok_all = ok_all and ok
        return ok_all

    def run_lakh_validation(self):
        """Run Lakh MIDI 32-variant sensitivity validation (rock vs classical)."""
        self.logger.info("Running Lakh MIDI validation pipeline")
        runner = PaperExperimentRunner(self.config)
        result = runner.run_lakh_validation()

        self.logger.info(f"Lakh validation done. Valid rows: {result.get('valid_rows')}")
        for label, path in result.get("outputs", {}).items():
            self.logger.info(f"Output {label}: {path}")

        # Generate Lakh-specific plots
        self.run_lakh_plots()
        return result

    def run_lakh_plots(self):
        """Generate publication plots for Lakh validation outputs."""
        from experiments.lakh_plots import generate_lakh_plots

        self.logger.info("Generating Lakh validation plots")
        outputs = generate_lakh_plots(self.config)
        if not outputs:
            self.logger.warning("No Lakh plots generated")
            return outputs
        for label, path in outputs.items():
            self.logger.info(f"Lakh plot {label}: {path}")
        return outputs


def build_arg_parser() -> argparse.ArgumentParser:
    """Create CLI parser."""
    parser = argparse.ArgumentParser(description="FMD sensitivity analysis runner")
    parser.add_argument(
        "--mode",
        choices=[
            "quick",
            "full",
            "paper",
            "paper-full",
            "paper-plots",
            "lakh",
            "lakh-plots",
            "cross-validate",
            "fetch-data",
            "tests",
            "demo",
            "lint",
        ],
        default="quick",
        help=(
            "quick: one-click default (demo + quick paper benchmark), "
            "full: tests + all experiments + full paper benchmark, "
            "paper: quick paper benchmark, paper-full: full benchmark, "
            "paper-plots: only generate plots from existing paper outputs, "
            "lakh: Lakh MIDI validation (32 variants, rock vs classical), "
            "lakh-plots: generate Lakh validation plots from existing outputs, "
            "fetch-data: download datasets from configured external sources"
        ),
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default=None,
        help="Optional single experiment name to run (overrides mode logic for experiments)",
    )
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=None,
        help="Optional list of dataset names for --mode fetch-data",
    )
    parser.add_argument(
        "--cv-source",
        choices=["cd1", "midicaps", "all"],
        default="all",
        dest="cv_source",
        help="Source for cross-validation mode: cd1, midicaps, or all (default: all)",
    )
    return parser


def main():
    """One-click entry point for IDE Run button and CLI."""
    parser = build_arg_parser()
    args = parser.parse_args()

    analysis = FMDSensitivityAnalysis()
    start_time = time.perf_counter()

    def _done(success: bool = True):
        elapsed = time.perf_counter() - start_time
        status = "SUCCESS" if success else "FAILED"
        print(f"\n=== Program finished: {status} | elapsed: {elapsed:.1f}s ===")

    try:
        if args.experiment:
            analysis.run_experiment(args.experiment)
            _done(True)
            return

        if args.mode == "quick":
            steps = [
                ("demo", lambda: analysis.run_demo()),
                ("paper-quick", lambda: analysis.run_paper_benchmark(full=False)),
            ]
            for i, (label, fn) in enumerate(
                tqdm(steps, desc="Quick pipeline", unit="step", dynamic_ncols=True),
                start=1,
            ):
                pct = 100.0 * i / len(steps)
                analysis.logger.info(f"[Progress] {i}/{len(steps)} ({pct:.1f}%) -> {label}")
                fn()
            _done(True)
            return

        if args.mode == "paper":
            analysis.logger.info("[Progress] 100.0% -> paper")
            analysis.run_paper_benchmark(full=False)
            _done(True)
            return

        if args.mode == "paper-full":
            analysis.logger.info("[Progress] 100.0% -> paper-full")
            analysis.run_paper_benchmark(full=True)
            _done(True)
            return

        if args.mode == "paper-plots":
            analysis.logger.info("[Progress] 100.0% -> paper-plots")
            analysis.run_publication_plots()
            _done(True)
            return

        if args.mode == "fetch-data":
            ok = analysis.run_fetch_data(args.datasets)
            if not ok:
                _done(False)
                raise SystemExit(1)
            _done(True)
            return

        if args.mode == "lakh":
            analysis.logger.info("[Progress] 100.0% -> lakh")
            analysis.run_lakh_validation()
            _done(True)
            return

        if args.mode == "lakh-plots":
            analysis.logger.info("[Progress] 100.0% -> lakh-plots")
            analysis.run_lakh_plots()
            _done(True)
            return

        if args.mode == "cross-validate":
            analysis.logger.info("[Progress] 100.0% -> cross-validate")
            import subprocess as _sp
            source_arg = getattr(args, "cv_source", "all") or "all"
            cmd = [sys.executable, "run_cross_dataset_validation.py", "--source", source_arg]
            result = _sp.run(cmd)
            _done(result.returncode == 0)
            if result.returncode != 0:
                raise SystemExit(result.returncode)
            return

        if args.mode == "tests":
            ok = analysis.run_all_tests()
            if not ok:
                _done(False)
                raise SystemExit(1)
            _done(True)
            return

        if args.mode == "demo":
            analysis.logger.info("[Progress] 100.0% -> demo")
            analysis.run_demo()
            _done(True)
            return

        if args.mode == "lint":
            analysis.logger.info("[Progress] 100.0% -> lint")
            analysis.run_lint_check()
            _done(True)
            return

        # Full mode
        steps = [
            ("tests", lambda: analysis.run_all_tests()),
            ("experiments", lambda: analysis.run_all_experiments()),
            ("paper-full", lambda: analysis.run_paper_benchmark(full=True)),
        ]
        for i, (label, fn) in enumerate(
            tqdm(steps, desc="Full pipeline", unit="step", dynamic_ncols=True),
            start=1,
        ):
            pct = 100.0 * i / len(steps)
            analysis.logger.info(f"[Progress] {i}/{len(steps)} ({pct:.1f}%) -> {label}")
            result = fn()
            if label == "tests" and result is False:
                _done(False)
                raise SystemExit(1)
        _done(True)
    except Exception:
        _done(False)
        raise


if __name__ == "__main__":
    main()

