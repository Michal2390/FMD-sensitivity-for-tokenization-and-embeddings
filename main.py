"""Main entry point with easy IDE click support."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from utils.config import load_config, setup_logging, get_logger
from run_experiment import ExperimentRunner


class FMDSensitivityAnalysis:
    """Main class for FMD Sensitivity Analysis - Easy IDE click execution."""

    def __init__(self):
        """Initialize the analysis."""
        self.config = load_config("configs/config.yaml")
        setup_logging(
            self.config["logging"].get("level", "INFO"),
            self.config["logging"].get("log_file", "logs/experiment.log"),
        )
        self.logger = get_logger(__name__)

    def run_all_tests(self):
        """Run all unit tests."""
        import pytest

        self.logger.info("🧪 Running all unit tests...")
        result = pytest.main(["tests/", "-v", "--tb=short"])
        return result == 0

    def run_demo(self):
        """Run demo of all modules."""
        import numpy as np
        from data.manager import DatasetManager
        from metrics.fmd import FrechetMusicDistance

        self.logger.info("📊 Running demo...")

        self.logger.info("Demo 1: Loading configuration")
        self.logger.info(f"  ✓ Experiments: {len(self.config['experiments'])}")
        self.logger.info(
            f"  ✓ Tokenizers: {len(self.config['tokenization']['tokenizers'])}"
        )
        self.logger.info(
            f"  ✓ Models: {len(self.config['embeddings']['models'])}"
        )

        self.logger.info("Demo 2: DatasetManager")
        dm = DatasetManager(self.config)
        self.logger.info("  ✓ DatasetManager initialized")
        for ds in self.config["data"]["datasets"]:
            self.logger.info(f"    - {ds['name']} (v{ds['version']})")

        self.logger.info("Demo 3: FMD Metric")
        fmd_calc = FrechetMusicDistance(self.config)
        embeddings1 = np.random.randn(100, 64)
        embeddings2 = np.random.randn(100, 64)
        fmd_value = fmd_calc.compute_fmd(embeddings1, embeddings2)
        self.logger.info(f"  ✓ FMD value: {fmd_value:.4f}")

        self.logger.info("✅ Demo completed!")

    def run_experiment(self, experiment_name: str):
        """Run a specific experiment.

        Args:
            experiment_name: Name of the experiment (e.g., 'exp1_tokenization_sensitivity')
        """
        self.logger.info(f"🔬 Running experiment: {experiment_name}")
        runner = ExperimentRunner("configs/config.yaml")
        runner.run_experiment(experiment_name)

    def run_all_experiments(self):
        """Run all enabled experiments."""
        self.logger.info("🔬 Running all experiments...")
        runner = ExperimentRunner("configs/config.yaml")
        runner.run_all_experiments()

    def run_lint_check(self):
        """Run code quality checks."""
        import subprocess

        self.logger.info("🔍 Running code quality checks...")

        # Black check
        self.logger.info("Checking Black formatting...")
        result = subprocess.run(
            ["black", "src/", "tests/", "run_experiment.py", "--check", "--quiet"],
            capture_output=True,
        )
        if result.returncode == 0:
            self.logger.info("  ✓ Black: OK")
        else:
            self.logger.warning("  ⚠ Black: Some files need formatting")

        # Flake8 check
        self.logger.info("Checking Flake8 linting...")
        result = subprocess.run(
            ["flake8", "src/", "tests/", "run_experiment.py", "--count"],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            self.logger.info("  ✓ Flake8: OK")
        else:
            self.logger.warning(f"  ⚠ Flake8: {result.stdout}")


def main():
    """Main entry point - can be clicked in IDE!"""
    print("\n" + "=" * 80)
    print("🎓 FMD SENSITIVITY ANALYSIS - WEEK 1 PROJECT")
    print("=" * 80)
    print(
        """
    Choose what to run:
    1. Run all tests (pytest)
    2. Run demo
    3. Run specific experiment (exp1_tokenization_sensitivity)
    4. Run all experiments
    5. Run code quality checks
    
    Example usage in code:
        analysis = FMDSensitivityAnalysis()
        analysis.run_all_tests()
        analysis.run_demo()
        analysis.run_experiment('exp1_tokenization_sensitivity')
    """
    )
    print("=" * 80 + "\n")

    analysis = FMDSensitivityAnalysis()

    # Run all by default when clicking Run in IDE
    print("▶ Running: All tests + Demo\n")
    if analysis.run_all_tests():
        print("\n✅ Tests passed!\n")
    print("\n▶ Running: Demo\n")
    analysis.run_demo()
    print("\n✅ Demo completed!\n")
    print("=" * 80)
    print("✨ All systems operational! Ready for Week 2 development.")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()

