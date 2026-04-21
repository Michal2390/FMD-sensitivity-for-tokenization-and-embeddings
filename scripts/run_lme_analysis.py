#!/usr/bin/env python3
"""Linear Mixed-Effects model analysis for FMD data.

Reads existing multi-genre FMD CSV and fits LME models that properly
account for the hierarchical structure (repeated measures within genre pairs).

Compares ANOVA vs LME conclusions.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict

import pandas as pd
import yaml
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from experiments.sensitivity_analysis import fit_lme_model, fit_lme_with_interactions

OUTPUT_DIR = Path("results/reports/lakh_multi")
INPUT_CSV = OUTPUT_DIR / "multi_genre_fmd.csv"
NFMD_CSV = OUTPUT_DIR / "nfmd_multi_genre.csv"


def main():
    logger.info("=" * 70)
    logger.info("LINEAR MIXED-EFFECTS MODEL ANALYSIS")
    logger.info("=" * 70)

    lme_dir = OUTPUT_DIR / "lme"
    lme_dir.mkdir(parents=True, exist_ok=True)

    # ── Load multi-genre FMD data ─────────────────────────────────────
    for csv_path, label in [(INPUT_CSV, "raw_fmd"), (NFMD_CSV, "nfmd")]:
        if not csv_path.exists():
            logger.warning(f"CSV not found: {csv_path}, skipping {label}")
            continue

        logger.info(f"\n=== Fitting LME on {label} ({csv_path.name}) ===")
        df = pd.read_csv(csv_path)
        logger.info(f"Loaded {len(df)} rows")

        # Ensure preprocess column
        if "preprocess" not in df.columns:
            if "remove_velocity" in df.columns and "hard_quantization" in df.columns:
                df["preprocess"] = df["remove_velocity"].astype(str) + "_" + df["hard_quantization"].astype(str)

        # Ensure pair column
        if "pair" not in df.columns:
            if "genre_a" in df.columns and "genre_b" in df.columns:
                df["pair"] = df["genre_a"] + "_vs_" + df["genre_b"]

        if "pair" not in df.columns:
            logger.warning("No 'pair' column found — cannot fit LME with random effects")
            continue

        # Determine response variables
        responses = ["fmd"]
        if "nfmd_trace" in df.columns:
            responses.append("nfmd_trace")
        if "nfmd_norm" in df.columns:
            responses.append("nfmd_norm")

        for response in responses:
            if response not in df.columns:
                continue

            sub_dir = lme_dir / f"{label}_{response}"

            # Model 1: Main effects only
            logger.info(f"\n  --- LME main effects: {response} ---")
            result1 = fit_lme_model(
                df,
                response=response,
                fixed_effects="C(tokenizer) + C(model) + C(preprocess)",
                random_effects="pair",
                output_dir=sub_dir / "main_effects",
            )
            if "error" not in result1:
                logger.info(f"  AIC={result1['aic']:.2f}, BIC={result1['bic']:.2f}")
                logger.info(f"  Random effects variance (pair): {result1['random_effects_variance']:.6f}")

            # Model 2: With interactions
            logger.info(f"\n  --- LME with interactions: {response} ---")
            result2 = fit_lme_with_interactions(
                df,
                response=response,
                random_effects="pair",
                output_dir=sub_dir / "interactions",
            )
            if "error" not in result2:
                logger.info(f"  AIC={result2['aic']:.2f}, BIC={result2['bic']:.2f}")

            # Compare AIC
            if "error" not in result1 and "error" not in result2:
                delta_aic = result2["aic"] - result1["aic"]
                better = "interactions" if delta_aic < 0 else "main effects"
                logger.info(f"  ΔAIC = {delta_aic:.2f} → {better} model preferred")

    logger.info("\n" + "=" * 70)
    logger.info("LME ANALYSIS COMPLETE")
    logger.info(f"  Results: {lme_dir}/")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()

