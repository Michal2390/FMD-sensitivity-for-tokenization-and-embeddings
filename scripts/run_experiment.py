#!/usr/bin/env python3
"""Main experiment runner for FMD sensitivity analysis."""

import argparse
import sys
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from utils.config import load_config, setup_logging, get_logger
from data.manager import DatasetManager
from preprocessing.processor import PreprocessingPipeline
from tokenization.tokenizer import TokenizationPipeline
from embeddings.extractor import EmbeddingExtractor
from metrics.fmd import FrechetMusicDistance, FMDComparator

logger = get_logger(__name__)


class ExperimentRunner:
    """Orchestrates the entire FMD sensitivity analysis experiment."""

    def __init__(self, config_path: str = "configs/config.yaml"):
        """
        Initialize experiment runner.

        Args:
            config_path: Path to configuration file
        """
        self.config = load_config(config_path)
        setup_logging(
            self.config["logging"].get("level", "INFO"),
            self.config["logging"].get("log_file", "logs/experiment.log"),
        )

        self.data_manager = DatasetManager(self.config)
        self.preprocessor = PreprocessingPipeline(self.config)
        self.tokenizer = TokenizationPipeline(self.config)
        self.embeddings = EmbeddingExtractor(self.config)
        self.fmd = FrechetMusicDistance(self.config)
        self.comparator = FMDComparator(self.config)

        logger.info("ExperimentRunner initialized")

    def run_experiment(self, experiment_name: str):
        """
        Run a specific experiment.

        Args:
            experiment_name: Name of experiment to run
        """
        logger.info(f"Starting experiment: {experiment_name}")

        experiment_config = self.config["experiments"].get(experiment_name)
        if not experiment_config:
            logger.error(f"Experiment '{experiment_name}' not found in configuration")
            return

        if not experiment_config.get("enabled", False):
            logger.warning(f"Experiment '{experiment_name}' is disabled")
            return

        logger.info(f"Description: {experiment_config.get('description', 'N/A')}")

        # Placeholder for actual experiment logic
        logger.info(f"Experiment '{experiment_name}' would run here")

    def run_all_experiments(self):
        """Run all enabled experiments."""
        experiments = self.config["experiments"]

        for exp_name, exp_config in experiments.items():
            if exp_config.get("enabled", False):
                try:
                    self.run_experiment(exp_name)
                except Exception as e:
                    logger.error(f"Error running experiment {exp_name}: {e}")

    def run_exp1_tokenization_sensitivity(self):
        """Experiment 1: Tokenization Sensitivity."""
        logger.info("=" * 80)
        logger.info("EXPERIMENT 1: Tokenization Sensitivity")
        logger.info("=" * 80)

        exp_config = self.config["experiments"]["exp1_tokenization_sensitivity"]
        results = {}

        for variant in exp_config["variants"]:
            logger.info(f"Running variant: {variant}")
            # Implementation would go here

        logger.info("Experiment 1 complete")
        return results

    def run_exp2_model_sensitivity(self):
        """Experiment 2: Model Sensitivity."""
        logger.info("=" * 80)
        logger.info("EXPERIMENT 2: Model Sensitivity")
        logger.info("=" * 80)

        exp_config = self.config["experiments"]["exp2_model_sensitivity"]
        results = {}

        for variant in exp_config["variants"]:
            logger.info(f"Running variant: {variant}")
            # Implementation would go here

        logger.info("Experiment 2 complete")
        return results

    def run_exp3_expression_ablation(self):
        """Experiment 3: Expression Ablation Study."""
        logger.info("=" * 80)
        logger.info("EXPERIMENT 3: Expression Ablation Study")
        logger.info("=" * 80)

        exp_config = self.config["experiments"]["exp3_expression_ablation"]
        results = {}

        for variant in exp_config["variants"]:
            logger.info(f"Running variant: {variant}")
            # Implementation would go here

        logger.info("Experiment 3 complete")
        return results

    def run_exp4_quantization_sensitivity(self):
        """Experiment 4: Quantization Sensitivity."""
        logger.info("=" * 80)
        logger.info("EXPERIMENT 4: Quantization Sensitivity")
        logger.info("=" * 80)

        exp_config = self.config["experiments"]["exp4_quantization_sensitivity"]
        results = {}

        for variant in exp_config["variants"]:
            logger.info(f"Running variant: {variant}")
            # Implementation would go here

        logger.info("Experiment 4 complete")
        return results

    def run_exp5_cross_genre(self):
        """Experiment 5: Cross-Genre Stability."""
        logger.info("=" * 80)
        logger.info("EXPERIMENT 5: Cross-Genre Stability")
        logger.info("=" * 80)

        exp_config = self.config["experiments"]["exp5_cross_genre"]
        results = {}

        for pair in exp_config["pairs"]:
            logger.info(f"Comparing datasets: {pair}")
            # Implementation would go here

        logger.info("Experiment 5 complete")
        return results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="FMD Sensitivity Analysis Experiment Runner")
    parser.add_argument(
        "--config", default="configs/config.yaml", help="Path to configuration file"
    )
    parser.add_argument(
        "--experiment", help="Specific experiment to run (e.g., exp1_tokenization_sensitivity)"
    )
    parser.add_argument("--all", action="store_true", help="Run all enabled experiments")

    args = parser.parse_args()

    try:
        runner = ExperimentRunner(args.config)

        if args.all:
            logger.info("Running all enabled experiments...")
            runner.run_all_experiments()
        elif args.experiment:
            logger.info(f"Running specific experiment: {args.experiment}")
            runner.run_experiment(args.experiment)
        else:
            logger.info("No experiment specified. Use --help for options.")
            parser.print_help()

    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
