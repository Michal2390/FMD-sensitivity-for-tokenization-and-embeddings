#!/usr/bin/env python3
"""Sample-size sensitivity analysis (power analysis).

Investigates how ANOVA effect sizes (η²) stabilise as a function of
the number of MIDI files per genre.  This answers the practical question:
"How many files do I need for reliable FMD sensitivity conclusions?"

The script re-uses the existing multi-genre FMD observations stored in
``results/reports/lakh_multi/multi_genre_fmd.csv`` and simulates smaller
sample sizes by sub-sampling rows.  No re-extraction of embeddings is
needed — only statistical recomputation.

Outputs
-------
- ``results/reports/lakh_multi/sample_size_ablation.csv``
- ``results/plots/paper/sample_size_stability.{png,pdf}``
- ``results/plots/paper/sample_size_power.{png,pdf}``
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent / "src"))

from scipy import stats as sp_stats

# ──────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────
INPUT_CSV = Path("results/reports/lakh_multi/multi_genre_fmd.csv")
OUTPUT_DIR = Path("results/reports/lakh_multi")
PLOTS_DIR = Path("results/plots/paper")

# Sample sizes to test (rows per variant×pair cell)
SAMPLE_SIZES = [2, 3, 5, 7, 10]  # out of max 10 repeats available
# Number of random re-samplings per sample size
N_RESAMPLES = 30
SEED = 42
# ──────────────────────────────────────────────────────────────────────


def compute_oneway_eta_sq(df: pd.DataFrame, factor: str) -> float:
    """One-way η² = SS_between / SS_total."""
    grand_mean = df["fmd"].mean()
    ss_total = ((df["fmd"] - grand_mean) ** 2).sum()
    if ss_total < 1e-15:
        return 0.0
    groups = [g["fmd"].values for _, g in df.groupby(factor)]
    ss_between = sum(
        len(g) * (np.mean(g) - grand_mean) ** 2 for g in groups
    )
    return float(ss_between / ss_total)


def compute_oneway_p(df: pd.DataFrame, factor: str) -> float:
    """One-way ANOVA p-value."""
    groups = [g["fmd"].values for _, g in df.groupby(factor)]
    if len(groups) < 2:
        return 1.0
    _, p = sp_stats.f_oneway(*groups)
    return float(p)


def subsample_df(
    df: pd.DataFrame, n_per_cell: int, rng: np.random.Generator
) -> pd.DataFrame:
    """Subsample *n_per_cell* repeats from each (variant, pair) cell."""
    parts = []
    for (variant, pair), grp in df.groupby(["variant", "pair"]):
        if len(grp) < n_per_cell:
            parts.append(grp)
        else:
            idx = rng.choice(len(grp), size=n_per_cell, replace=False)
            parts.append(grp.iloc[idx])
    return pd.concat(parts, ignore_index=True)


def main():
    logger.info("=" * 70)
    logger.info("SAMPLE-SIZE SENSITIVITY ANALYSIS (POWER ANALYSIS)")
    logger.info("=" * 70)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    if not INPUT_CSV.exists():
        logger.error(f"Input CSV not found: {INPUT_CSV}")
        logger.error("Run run_multi_genre_analysis.py first.")
        sys.exit(1)

    df_full = pd.read_csv(INPUT_CSV)
    df_full["preprocess"] = (
        df_full["remove_velocity"].astype(str) + "_" + df_full["hard_quantization"].astype(str)
    )
    logger.info(f"Loaded {len(df_full)} rows from {INPUT_CSV}")

    factors = ["tokenizer", "model", "preprocess"]
    rng = np.random.default_rng(SEED)

    # ── Run ablation ──────────────────────────────────────────────────
    rows: List[Dict] = []

    for n_per_cell in SAMPLE_SIZES:
        for resample_idx in range(N_RESAMPLES):
            sub_df = subsample_df(df_full, n_per_cell, rng)
            total_rows = len(sub_df)

            for factor in factors:
                eta_sq = compute_oneway_eta_sq(sub_df, factor)
                p_val = compute_oneway_p(sub_df, factor)
                rows.append({
                    "n_per_cell": n_per_cell,
                    "total_rows": total_rows,
                    "resample": resample_idx,
                    "factor": factor,
                    "eta_sq": eta_sq,
                    "p_value": p_val,
                    "significant": p_val < 0.05,
                })

        logger.info(
            f"  n_per_cell={n_per_cell}: {N_RESAMPLES} resamples done "
            f"(~{rows[-1]['total_rows']} rows each)"
        )

    result_df = pd.DataFrame(rows)
    csv_out = OUTPUT_DIR / "sample_size_ablation.csv"
    result_df.to_csv(csv_out, index=False)
    logger.info(f"Saved: {csv_out}")

    # ── Summary statistics ────────────────────────────────────────────
    summary = (
        result_df.groupby(["n_per_cell", "factor"])
        .agg(
            eta_sq_mean=("eta_sq", "mean"),
            eta_sq_std=("eta_sq", "std"),
            eta_sq_ci_lo=("eta_sq", lambda x: np.percentile(x, 2.5)),
            eta_sq_ci_hi=("eta_sq", lambda x: np.percentile(x, 97.5)),
            power=("significant", "mean"),  # fraction of resamples where p < 0.05
            n_resamples=("eta_sq", "count"),
        )
        .reset_index()
    )
    summary.to_csv(OUTPUT_DIR / "sample_size_summary.csv", index=False)
    logger.info(f"\n{summary.to_string(index=False)}")

    # ── Generate plots ────────────────────────────────────────────────
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_theme(style="whitegrid", font_scale=1.1)
    DPI = 300

    def save_fig(fig, name):
        fig.tight_layout()
        for fmt in ("png",):
            fig.savefig(PLOTS_DIR / f"{name}.{fmt}", dpi=DPI, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"  ✅ {name}")

    # Plot 1: η² stability curves with 95% CI ribbon
    fig, ax = plt.subplots(figsize=(10, 6))
    palette = sns.color_palette("Set1", len(factors))

    for i, factor in enumerate(factors):
        sub = summary[summary["factor"] == factor]
        x = sub["n_per_cell"].values
        y = sub["eta_sq_mean"].values
        lo = sub["eta_sq_ci_lo"].values
        hi = sub["eta_sq_ci_hi"].values

        ax.plot(x, y, "o-", color=palette[i], label=factor, linewidth=2, markersize=8)
        ax.fill_between(x, lo, hi, color=palette[i], alpha=0.15)

    ax.set_xlabel("Repeats per variant×pair cell")
    ax.set_ylabel("η²")
    ax.set_title("η² Stability vs Sample Size (95% CI from 30 resamples)")
    ax.legend()
    ax.axhline(0.14, color="red", linestyle="--", alpha=0.3, label="large threshold")
    ax.axhline(0.06, color="orange", linestyle="--", alpha=0.3, label="medium threshold")
    ax.set_xticks(SAMPLE_SIZES)
    save_fig(fig, "sample_size_stability")

    # Plot 2: Power curves (% of resamples where p < 0.05)
    fig, ax = plt.subplots(figsize=(10, 6))

    for i, factor in enumerate(factors):
        sub = summary[summary["factor"] == factor]
        ax.plot(
            sub["n_per_cell"], sub["power"],
            "o-", color=palette[i], label=factor, linewidth=2, markersize=8,
        )

    ax.set_xlabel("Repeats per variant×pair cell")
    ax.set_ylabel("Power (fraction p < 0.05)")
    ax.set_title("Statistical Power vs Sample Size")
    ax.axhline(0.80, color="gray", linestyle="--", alpha=0.5, label="0.80 threshold")
    ax.legend()
    ax.set_ylim(-0.05, 1.05)
    ax.set_xticks(SAMPLE_SIZES)
    save_fig(fig, "sample_size_power")

    # Plot 3: Combined 2-panel figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    for i, factor in enumerate(factors):
        sub = summary[summary["factor"] == factor]
        x = sub["n_per_cell"].values
        y = sub["eta_sq_mean"].values
        lo = sub["eta_sq_ci_lo"].values
        hi = sub["eta_sq_ci_hi"].values

        ax1.plot(x, y, "o-", color=palette[i], label=factor, linewidth=2, markersize=7)
        ax1.fill_between(x, lo, hi, color=palette[i], alpha=0.15)

        ax2.plot(
            sub["n_per_cell"], sub["power"],
            "o-", color=palette[i], label=factor, linewidth=2, markersize=7,
        )

    ax1.set_xlabel("Repeats per cell")
    ax1.set_ylabel("η²")
    ax1.set_title("A) η² Stability")
    ax1.legend(fontsize=9)
    ax1.axhline(0.14, color="red", linestyle="--", alpha=0.3)
    ax1.axhline(0.06, color="orange", linestyle="--", alpha=0.3)
    ax1.set_xticks(SAMPLE_SIZES)

    ax2.set_xlabel("Repeats per cell")
    ax2.set_ylabel("Power (p < 0.05)")
    ax2.set_title("B) Statistical Power")
    ax2.axhline(0.80, color="gray", linestyle="--", alpha=0.5)
    ax2.legend(fontsize=9)
    ax2.set_ylim(-0.05, 1.05)
    ax2.set_xticks(SAMPLE_SIZES)

    save_fig(fig, "sample_size_combined")

    # ── Save JSON summary ─────────────────────────────────────────────
    json_out = {
        "config": {
            "sample_sizes": SAMPLE_SIZES,
            "n_resamples": N_RESAMPLES,
            "seed": SEED,
            "input_csv": str(INPUT_CSV),
            "total_input_rows": len(df_full),
        },
        "summary": summary.to_dict(orient="records"),
    }
    with open(OUTPUT_DIR / "sample_size_ablation_results.json", "w") as f:
        json.dump(json_out, f, indent=2, default=str)

    logger.info("\n" + "=" * 70)
    logger.info("SAMPLE-SIZE ABLATION COMPLETE")
    logger.info(f"  Results: {csv_out}")
    logger.info(f"  Plots:   {PLOTS_DIR}/sample_size_*.{{png,pdf}}")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()


