"""Main entry point with one-click modes for research workflow."""

import argparse
import subprocess
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from experiments.paper_pipeline import PaperExperimentRunner
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
        return result


def build_arg_parser() -> argparse.ArgumentParser:
    """Create CLI parser."""
    parser = argparse.ArgumentParser(description="FMD sensitivity analysis runner")
    parser.add_argument(
        "--mode",
        choices=["quick", "full", "paper", "paper-full", "tests", "demo", "lint"],
        default="quick",
        help=(
            "quick: one-click default (demo + quick paper benchmark), "
            "full: tests + all experiments + full paper benchmark, "
            "paper: quick paper benchmark, paper-full: full benchmark"
        ),
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default=None,
        help="Optional single experiment name to run (overrides mode logic for experiments)",
    )
    return parser


def main():
    """One-click entry point for IDE Run button and CLI."""
    parser = build_arg_parser()
    args = parser.parse_args()

    analysis = FMDSensitivityAnalysis()

    if args.experiment:
        analysis.run_experiment(args.experiment)
        return

    if args.mode == "quick":
        analysis.run_demo()
        analysis.run_paper_benchmark(full=False)
        return

    if args.mode == "paper":
        analysis.run_paper_benchmark(full=False)
        return

    if args.mode == "paper-full":
        analysis.run_paper_benchmark(full=True)
        return

    if args.mode == "tests":
        ok = analysis.run_all_tests()
        if not ok:
            raise SystemExit(1)
        return

    if args.mode == "demo":
        analysis.run_demo()
        return

    if args.mode == "lint":
        analysis.run_lint_check()
        return

    # Full mode
    ok = analysis.run_all_tests()
    if not ok:
        raise SystemExit(1)
    analysis.run_all_experiments()
    analysis.run_paper_benchmark(full=True)


if __name__ == "__main__":
    main()

