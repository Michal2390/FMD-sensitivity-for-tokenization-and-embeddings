"""Publication-ready plots for paper benchmark outputs."""

from __future__ import annotations

from pathlib import Path
from typing import Dict
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from loguru import logger


def _safe_save(fig: plt.Figure, output_path: Path) -> str:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return str(output_path)


def _build_symmetric_matrix(df: pd.DataFrame) -> pd.DataFrame:
    reversed_df = df.rename(columns={"dataset_a": "dataset_b", "dataset_b": "dataset_a"})
    merged = pd.concat([df, reversed_df], ignore_index=True)
    grouped = (
        merged.groupby(["dataset_a", "dataset_b"], as_index=False)["fmd"]
        .mean()
        .rename(columns={"dataset_a": "row", "dataset_b": "col"})
    )
    matrix = grouped.pivot(index="row", columns="col", values="fmd").sort_index().sort_index(axis=1)
    return matrix


def generate_publication_plots(config: Dict, report_dir: str | None = None, output_dir: str | None = None) -> Dict:
    """Generate publication-oriented figures from paper benchmark artifacts."""
    sns.set_theme(style="whitegrid")

    report_root = Path(report_dir) if report_dir else Path(config["results"]["reports_dir"]) / "paper"
    plots_root = Path(output_dir) if output_dir else Path(config["results"]["plots_dir"]) / "paper"
    plots_root.mkdir(parents=True, exist_ok=True)

    pairwise_csv = report_root / "pairwise_fmd.csv"
    special_csv = report_root / "special_pair_fmd.csv"
    special_summary_csv = report_root / "special_pair_summary.csv"
    special_top_csv = report_root / "special_pair_top_variants.csv"
    results_json = report_root / "paper_results.json"

    outputs: Dict[str, str] = {}

    if pairwise_csv.exists():
        pairwise_df = pd.read_csv(pairwise_csv)
        if not pairwise_df.empty:
            matrix = _build_symmetric_matrix(pairwise_df[["dataset_a", "dataset_b", "fmd"]])
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(matrix, cmap="viridis", annot=True, fmt=".2f", ax=ax)
            ax.set_title("Mean Pairwise FMD (All Variants)")
            outputs["pairwise_heatmap"] = _safe_save(fig, plots_root / "fig_pairwise_heatmap.png")

    if special_csv.exists():
        special_df = pd.read_csv(special_csv)
        if not special_df.empty:
            fig, ax = plt.subplots(figsize=(9, 5))
            sns.boxplot(data=special_df, x="pair", y="fmd", ax=ax)
            ax.set_title("Special Genre-Pair FMD Distribution")
            ax.set_xlabel("Genre pair")
            ax.set_ylabel("FMD")
            ax.tick_params(axis="x", rotation=25)
            outputs["special_pairs_boxplot"] = _safe_save(fig, plots_root / "fig_special_pairs_boxplot.png")

    if special_summary_csv.exists():
        summary_df = pd.read_csv(special_summary_csv)
        if not summary_df.empty:
            summary_df = summary_df.sort_values("mean_fmd", ascending=False)
            fig, ax = plt.subplots(figsize=(8, 4.5))
            sns.barplot(data=summary_df, x="pair", y="distinguishability_ratio", ax=ax, color="#4c72b0")
            ax.axhline(1.0, linestyle="--", color="black", linewidth=1)
            ax.set_title("Distinguishability Ratio by Special Pair")
            ax.set_xlabel("Genre pair")
            ax.set_ylabel("ratio vs global special-pair mean")
            ax.tick_params(axis="x", rotation=25)
            outputs["special_pairs_ratio"] = _safe_save(
                fig, plots_root / "fig_special_pairs_distinguishability.png"
            )

    if special_top_csv.exists():
        top_df = pd.read_csv(special_top_csv)
        if not top_df.empty:
            top_df = top_df.sort_values(["pair", "rank"])
            top_df["rank"] = top_df["rank"].astype(str)
            fig, ax = plt.subplots(figsize=(10, 5.5))
            sns.barplot(data=top_df, x="pair", y="fmd", hue="rank", ax=ax)
            ax.set_title("Top Separating Variants per Special Pair")
            ax.set_xlabel("Genre pair")
            ax.set_ylabel("FMD")
            ax.tick_params(axis="x", rotation=25)
            ax.legend(title="rank")
            outputs["top_variants_per_pair"] = _safe_save(
                fig, plots_root / "fig_top_variants_per_pair.png"
            )

    if results_json.exists():
        with open(results_json, "r", encoding="utf-8") as handle:
            payload = json.load(handle)

        stability = payload.get("ranking", {}).get("stability", {})
        if stability:
            stability_df = pd.DataFrame(
                [{"dataset": name, "stability": score} for name, score in stability.items()]
            ).sort_values("stability", ascending=False)
            fig, ax = plt.subplots(figsize=(8, 4.5))
            sns.barplot(data=stability_df, x="dataset", y="stability", ax=ax, color="#55a868")
            ax.set_title("Ranking Stability Across Variants")
            ax.set_ylim(0.0, 1.05)
            outputs["ranking_stability"] = _safe_save(fig, plots_root / "fig_ranking_stability.png")

        expected_details = payload.get("expected_eval", {}).get("details", [])
        if expected_details:
            detail_df = pd.DataFrame(expected_details)
            metric_df = detail_df.melt(
                id_vars=["variant", "reference"],
                value_vars=["spearman", "kendall"],
                var_name="metric",
                value_name="value",
            ).dropna()
            if not metric_df.empty:
                fig, ax = plt.subplots(figsize=(7, 4.5))
                sns.boxplot(data=metric_df, x="metric", y="value", ax=ax)
                ax.set_title("Rank Consistency vs Expected Orders")
                ax.set_ylim(-1.05, 1.05)
                outputs["rank_consistency"] = _safe_save(fig, plots_root / "fig_rank_consistency.png")

    if not outputs:
        logger.warning("No publication plots were generated - missing or empty paper report files")
    else:
        logger.info(f"Generated publication plots: {outputs}")

    return outputs



