"""Generate publication plots for FMD sensitivity analysis."""

import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path
from loguru import logger
from scipy.stats import ttest_ind, mannwhitneyu

from src.metrics.fmd import FrechetMusicDistance
from src.utils.config import load_config


def load_paper_results(results_dir: Path) -> Dict:
    """Load paper benchmark results."""
    json_file = results_dir / "paper" / "paper_results.json"
    if json_file.exists():
        with open(json_file, 'r') as f:
            return json.load(f)
    return {}


def generate_sensitivity_plots(results: Dict, output_dir: Path):
    """Generate plots showing FMD sensitivity to tokenization and models."""
    # Extract FMD values by variant
    variants = {}
    for key, value in results.items():
        if 'fmd' in value:
            # Parse variant from key
            parts = key.split('|')
            tokenizer = next((p.split('=')[1] for p in parts if p.startswith('tok=')), 'unknown')
            model = next((p.split('=')[1] for p in parts if p.startswith('model=')), 'unknown')
            vel = next((p.split('=')[1] for p in parts if p.startswith('vel=')), 'on')
            quant = next((p.split('=')[1] for p in parts if p.startswith('quant=')), 'off')

            variant_key = f"{tokenizer}_{model}_{vel}_{quant}"
            variants[variant_key] = value['fmd']

    # Plot tokenizer sensitivity
    tokenizers = ['REMI', 'TSD', 'Octuple', 'MIDI-Like']
    fmd_by_tok = [variants.get(f"{tok}_CLaMP-1_on_off", 0) for tok in tokenizers]

    plt.figure(figsize=(10, 6))
    plt.bar(tokenizers, fmd_by_tok)
    plt.title('FMD Sensitivity to Tokenization (CLaMP-1, no velocity, no quantization)')
    plt.ylabel('FMD Value')
    plt.savefig(output_dir / 'tokenizer_sensitivity.png')
    plt.close()

    # Plot model sensitivity
    models = ['CLaMP-1', 'CLaMP-2']
    fmd_by_model = [variants.get(f"REMI_{model}_on_off", 0) for model in models]

    plt.figure(figsize=(8, 6))
    plt.bar(models, fmd_by_model)
    plt.title('FMD Sensitivity to Embedding Model (REMI, no velocity, no quantization)')
    plt.ylabel('FMD Value')
    plt.savefig(output_dir / 'model_sensitivity.png')
    plt.close()

    logger.info("Generated sensitivity plots")


def generate_bootstrap_ci_plots(results: Dict, output_dir: Path, config: Dict):
    """Generate plots with bootstrap confidence intervals."""
    fmd_calc = FrechetMusicDistance(config)

    # Example: Compute CI for a pair
    # This would need actual embeddings, simplified here
    plt.figure(figsize=(10, 6))
    # Placeholder for CI plot
    plt.title('FMD with Bootstrap 95% CI (Placeholder)')
    plt.savefig(output_dir / 'bootstrap_ci_example.png')
    plt.close()

    logger.info("Generated bootstrap CI plots")


def generate_genre_comparison_plots(results: Dict, output_dir: Path):
    """Generate plots comparing FMD across genres."""
    # Extract pairwise FMD for special pairs
    pairs = ['classical_rock', 'jazz_rap']
    fmd_values = [results.get(pair, {}).get('fmd', 0) for pair in pairs]

    plt.figure(figsize=(8, 6))
    plt.bar(pairs, fmd_values)
    plt.title('FMD for Special Genre Pairs')
    plt.ylabel('FMD Value')
    plt.savefig(output_dir / 'genre_comparison.png')
    plt.close()

    logger.info("Generated genre comparison plots")


def main():
    """Main function to generate all publication plots."""
    config = load_config()
    results_dir = Path(config['results']['reports_dir'])
    plots_dir = Path(config['results']['plots_dir'])
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Load results
    results = load_paper_results(results_dir)
    if not results:
        logger.warning("No paper results found, skipping plot generation")
        return

    # Generate plots
    generate_sensitivity_plots(results, plots_dir)
    generate_bootstrap_ci_plots(results, plots_dir, config)
    generate_genre_comparison_plots(results, plots_dir)

    logger.info(f"All plots saved to {plots_dir}")


if __name__ == "__main__":
    main()
