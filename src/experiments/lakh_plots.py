"""Publication-quality plots for Lakh MIDI validation results.

Generates:
- Violin plot FMD per model (shows full distribution)
- Boxplot FMD per tokenizer (panels per model)
- Heatmap η² variance decomposition
- Interaction plot: tokenizer × model
- t-SNE scatter (colour = genre, shape = tokenizer)
- CDF/ECDF plot: FMD distributions per model
- Bootstrap CI error bars per variant
- Grouped bar chart intra- vs inter-genre cosine similarity
- Scatter FMD vs token entropy with regression line
- ANOVA summary table as figure
- Permutation test results
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import json
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from loguru import logger


_DPI = 300
_FORMATS = ("png", "pdf")


def _save(fig: plt.Figure, base_path: Path) -> str:
    """Save figure in PNG and PDF at publication DPI."""
    base_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    for fmt in _FORMATS:
        out = base_path.with_suffix(f".{fmt}")
        fig.savefig(out, dpi=_DPI, bbox_inches="tight")
    plt.close(fig)
    return str(base_path.with_suffix(".png"))


def _parse_variant(variant_str: str) -> Dict[str, str]:
    """Parse 'tok=REMI|model=MusicBERT|vel=on|quant=off' into dict."""
    parts = {}
    for segment in variant_str.split("|"):
        if "=" in segment:
            k, v = segment.split("=", 1)
            parts[k] = v
    return parts


def generate_lakh_plots(config: Dict) -> Dict[str, str]:
    """Generate all Lakh validation plots from saved artefacts."""
    sns.set_theme(style="whitegrid", font_scale=1.1)

    lakh_cfg = config.get("lakh", {})
    report_dir = Path(lakh_cfg.get("output_dir", "results/reports/lakh"))
    plots_dir = Path(lakh_cfg.get("plots_dir", "results/plots/paper"))
    plots_dir.mkdir(parents=True, exist_ok=True)

    outputs: Dict[str, str] = {}

    # Load pairwise data once
    pairwise_csv = report_dir / "lakh_pairwise_fmd.csv"
    df = None
    if pairwise_csv.exists():
        df = pd.read_csv(pairwise_csv)
        df = df[df["valid"] == True]  # noqa: E712
        if not df.empty:
            # Parse variant columns
            parsed = df["variant"].apply(_parse_variant).apply(pd.Series)
            for col in parsed.columns:
                if col not in df.columns:
                    df[col] = parsed[col]

    # ------------------------------------------------------------------
    # 1. Violin plot FMD per model (NEW — shows full distribution)
    # ------------------------------------------------------------------
    if df is not None and not df.empty:
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.violinplot(data=df, x="model", y="fmd", ax=ax, palette="Set1", inner="box", cut=0)
        sns.stripplot(data=df, x="model", y="fmd", ax=ax, color="black", alpha=0.3, size=4, jitter=True)
        ax.set_title("FMD Distribution by Embedding Model")
        ax.set_xlabel("Model")
        ax.set_ylabel("FMD")
        outputs["lakh_fmd_violin"] = _save(fig, plots_dir / "lakh_fmd_violin_by_model")

    # ------------------------------------------------------------------
    # 2. Boxplot FMD per tokenizer (faceted by model)
    # ------------------------------------------------------------------
    if df is not None and not df.empty:
        fig, axes = plt.subplots(1, df["model"].nunique(), figsize=(6 * df["model"].nunique(), 5), sharey=True)
        if df["model"].nunique() == 1:
            axes = [axes]
        for ax, (model, grp) in zip(axes, df.groupby("model")):
            sns.boxplot(data=grp, x="tokenizer", y="fmd", ax=ax, palette="Set2")
            ax.set_title(f"Model: {model}")
            ax.set_xlabel("Tokenizer")
            ax.set_ylabel("FMD")
        outputs["lakh_fmd_boxplot"] = _save(fig, plots_dir / "lakh_fmd_boxplot_by_tokenizer")

        # Combined boxplot with hue=model
        fig2, ax2 = plt.subplots(figsize=(9, 5))
        sns.boxplot(data=df, x="tokenizer", y="fmd", hue="model", ax=ax2, palette="Set1")
        ax2.set_title("FMD Sensitivity: Tokenizer × Model")
        ax2.set_xlabel("Tokenizer")
        ax2.set_ylabel("FMD")
        outputs["lakh_fmd_boxplot_combined"] = _save(fig2, plots_dir / "lakh_fmd_boxplot_combined")

    # ------------------------------------------------------------------
    # 3. Interaction plot: tokenizer × model (NEW)
    # ------------------------------------------------------------------
    if df is not None and not df.empty:
        fig, ax = plt.subplots(figsize=(8, 5))
        interaction_df = df.groupby(["tokenizer", "model"])["fmd"].agg(["mean", "std"]).reset_index()
        for model_name, grp in interaction_df.groupby("model"):
            ax.errorbar(
                grp["tokenizer"], grp["mean"], yerr=grp["std"],
                marker="o", capsize=5, label=model_name, linewidth=2
            )
        ax.set_xlabel("Tokenizer")
        ax.set_ylabel("Mean FMD ± SD")
        ax.set_title("Interaction Plot: Tokenizer × Model")
        ax.legend(title="Model")
        outputs["lakh_interaction_tok_model"] = _save(fig, plots_dir / "lakh_interaction_tok_model")

    # ------------------------------------------------------------------
    # 4. Bootstrap CI error bars per variant (NEW)
    # ------------------------------------------------------------------
    if df is not None and not df.empty and "bootstrap_ci_lower" in df.columns:
        fig, ax = plt.subplots(figsize=(14, 6))
        sorted_df = df.sort_values("fmd").reset_index(drop=True)
        colors = {"MusicBERT": "#1f77b4", "MusicBERT-large": "#ff7f0e", "MERT": "#2ca02c", "NLP-Baseline": "#d62728"}
        for i, row in sorted_df.iterrows():
            color = colors.get(row["model"], "gray")
            ci_lower = row.get("bootstrap_ci_lower", row["fmd"])
            ci_upper = row.get("bootstrap_ci_upper", row["fmd"])
            ax.errorbar(
                i, row["fmd"],
                yerr=[[row["fmd"] - ci_lower], [ci_upper - row["fmd"]]],
                fmt="o", color=color, capsize=3, markersize=5
            )
        # Custom legend
        for model_name, color in colors.items():
            ax.plot([], [], "o", color=color, label=model_name)
        ax.legend(title="Model")
        ax.set_xlabel("Variant (sorted by FMD)")
        ax.set_ylabel("FMD with 95% Bootstrap CI")
        ax.set_title("FMD per Variant with Bootstrap Confidence Intervals")
        outputs["lakh_bootstrap_ci"] = _save(fig, plots_dir / "lakh_bootstrap_ci")

    # ------------------------------------------------------------------
    # 5. CDF/ECDF plot: FMD by model (NEW)
    # ------------------------------------------------------------------
    if df is not None and not df.empty:
        fig, ax = plt.subplots(figsize=(8, 5))
        for model_name, grp in df.groupby("model"):
            sorted_fmd = np.sort(grp["fmd"].values)
            ecdf = np.arange(1, len(sorted_fmd) + 1) / len(sorted_fmd)
            ax.step(sorted_fmd, ecdf, label=model_name, linewidth=2)
        ax.set_xlabel("FMD")
        ax.set_ylabel("ECDF")
        ax.set_title("Empirical CDF of FMD by Model")
        ax.legend(title="Model")
        ax.grid(True, alpha=0.3)
        outputs["lakh_ecdf"] = _save(fig, plots_dir / "lakh_ecdf_by_model")

    # ------------------------------------------------------------------
    # 6. η² and partial η² variance decomposition
    # ------------------------------------------------------------------
    sens_json = report_dir / "sensitivity_results.json"
    if sens_json.exists():
        with open(sens_json, "r") as fh:
            sens = json.load(fh)

        eta_sq = sens.get("eta_squared")
        partial_eta_sq = sens.get("partial_eta_squared")

        if eta_sq:
            factors = list(eta_sq.keys())
            values = [eta_sq[f] for f in factors]

            # If partial η² available, show side-by-side
            if partial_eta_sq:
                p_values = [partial_eta_sq.get(f, 0) for f in factors]
                fig, ax = plt.subplots(figsize=(max(8, len(factors) * 2), 5))
                x = np.arange(len(factors))
                w = 0.35
                bars1 = ax.bar(x - w/2, values, w, label="η²", color=sns.color_palette("viridis", 1)[0])
                bars2 = ax.bar(x + w/2, p_values, w, label="Partial η²", color=sns.color_palette("viridis", 3)[2])
                ax.set_xticks(x)
                ax.set_xticklabels(factors, rotation=30, ha="right")
                ax.set_ylabel("Proportion of Variance")
                ax.set_title("Variance Decomposition: η² and Partial η²")
                ax.legend()
                for bar, v in zip(bars1, values):
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                            f"{v:.3f}", ha="center", va="bottom", fontsize=8)
                for bar, v in zip(bars2, p_values):
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                            f"{v:.3f}", ha="center", va="bottom", fontsize=8)
            else:
                fig, ax = plt.subplots(figsize=(max(6, len(factors) * 1.5), 4))
                colours = sns.color_palette("viridis", len(factors))
                bars = ax.bar(factors, values, color=colours)
                ax.set_ylabel("η² (proportion of variance)")
                ax.set_title("Variance Decomposition (η²)")
                for bar, v in zip(bars, values):
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                            f"{v:.3f}", ha="center", va="bottom", fontsize=9)
                ax.tick_params(axis="x", rotation=30)

            ax.set_ylim(0, max(values + (p_values if partial_eta_sq else [])) * 1.3 if values else 1.0)
            outputs["lakh_eta_squared"] = _save(fig, plots_dir / "lakh_eta_squared")

        # Cohen's d
        cohens = sens.get("cohens_d", {})
        if cohens:
            cd_df = pd.DataFrame([{"comparison": k, "cohens_d": v} for k, v in cohens.items()])
            cd_df = cd_df.sort_values("cohens_d", key=abs, ascending=False).head(15)
            fig, ax = plt.subplots(figsize=(8, max(4, len(cd_df) * 0.35)))
            sns.barplot(data=cd_df, y="comparison", x="cohens_d", ax=ax, palette="coolwarm")
            ax.set_title("Effect Sizes (Cohen's d)")
            ax.set_xlabel("Cohen's d")
            outputs["lakh_cohens_d"] = _save(fig, plots_dir / "lakh_cohens_d")

        # Permutation test results (NEW)
        perm = sens.get("permutation_tests", {})
        if perm:
            perm_rows = []
            for factor, res in perm.items():
                if res.get("observed_F") is not None:
                    perm_rows.append({
                        "Factor": factor,
                        "F-statistic": res["observed_F"],
                        "Permutation p": res["permutation_p"],
                        "N perms": res["n_permutations"],
                    })
            if perm_rows:
                perm_df = pd.DataFrame(perm_rows)
                fig, ax = plt.subplots(figsize=(8, max(2, len(perm_df) * 0.6 + 1)))
                ax.axis("off")
                table = ax.table(
                    cellText=perm_df.round(4).values,
                    colLabels=perm_df.columns,
                    cellLoc="center", loc="center",
                )
                table.auto_set_font_size(False)
                table.set_fontsize(9)
                table.scale(1.2, 1.4)
                ax.set_title("Permutation Test Results", fontsize=12, pad=20)
                outputs["lakh_permutation_tests"] = _save(fig, plots_dir / "lakh_permutation_tests")

    # ------------------------------------------------------------------
    # 7. Cosine similarity: intra vs inter
    # ------------------------------------------------------------------
    cosine_csv = report_dir / "cosine_similarity.csv"
    if cosine_csv.exists():
        cos_df = pd.read_csv(cosine_csv)
        if not cos_df.empty:
            parsed = cos_df["variant"].apply(_parse_variant).apply(pd.Series)
            cos_df = pd.concat([cos_df, parsed], axis=1)
            melted = cos_df.melt(
                id_vars=["variant", "tok", "model"],
                value_vars=["intra_a", "intra_b", "inter"],
                var_name="similarity_type", value_name="cosine_sim",
            )
            fig, ax = plt.subplots(figsize=(10, 5))
            sns.barplot(data=melted, x="tok", y="cosine_sim", hue="similarity_type", ax=ax, palette="Set2")
            ax.set_title("Intra- vs Inter-Genre Cosine Similarity")
            ax.set_xlabel("Tokenizer")
            ax.set_ylabel("Mean Cosine Similarity")
            ax.legend(title="Type")
            outputs["lakh_cosine_similarity"] = _save(fig, plots_dir / "lakh_cosine_similarity")

    # ------------------------------------------------------------------
    # 8. t-SNE scatter
    # ------------------------------------------------------------------
    tsne_csv = report_dir / "tsne_projections.csv"
    if tsne_csv.exists():
        tsne_df = pd.read_csv(tsne_csv)
        if not tsne_df.empty:
            parsed = tsne_df["variant"].apply(_parse_variant).apply(pd.Series)
            tsne_df = pd.concat([tsne_df, parsed], axis=1)
            unique_variants = tsne_df["variant"].unique()
            sample_variants = []
            seen_tok = set()
            for v in unique_variants:
                p = _parse_variant(v)
                if p.get("tok") not in seen_tok and p.get("vel") == "on" and p.get("quant") == "off":
                    sample_variants.append(v)
                    seen_tok.add(p.get("tok"))
                if len(sample_variants) >= 4:
                    break

            if sample_variants:
                sub = tsne_df[tsne_df["variant"].isin(sample_variants)]
                g = sns.FacetGrid(sub, col="tok", col_wrap=2, height=4, sharex=False, sharey=False)
                g.map_dataframe(sns.scatterplot, x="tsne1", y="tsne2", hue="genre",
                                style="genre", palette="Set1", alpha=0.7, s=30)
                g.add_legend()
                g.figure.suptitle("t-SNE: Genre Embeddings by Tokenizer", y=1.02)
                outputs["lakh_tsne"] = _save(g.figure, plots_dir / "lakh_tsne_by_tokenizer")

    # ------------------------------------------------------------------
    # 9. Token statistics
    # ------------------------------------------------------------------
    tok_csv = report_dir / "token_statistics.csv"
    if tok_csv.exists():
        tok_df = pd.read_csv(tok_csv)
        if not tok_df.empty:
            parsed = tok_df["variant"].apply(_parse_variant).apply(pd.Series)
            tok_df = pd.concat([tok_df, parsed], axis=1)
            for metric in ("length_mean", "entropy_mean"):
                if metric in tok_df.columns:
                    fig, ax = plt.subplots(figsize=(8, 5))
                    sns.barplot(data=tok_df, x="tok", y=metric, hue="genre", ax=ax, palette="Set1")
                    nice = metric.replace("_mean", "").title()
                    ax.set_title(f"Token {nice} by Tokenizer and Genre")
                    ax.set_xlabel("Tokenizer")
                    ax.set_ylabel(f"Mean {nice}")
                    outputs[f"lakh_token_{metric}"] = _save(fig, plots_dir / f"lakh_token_{metric}")

    # ------------------------------------------------------------------
    # 10. FMD vs token entropy scatter
    # ------------------------------------------------------------------
    corr_json = report_dir / "fmd_token_correlations.json"
    if corr_json.exists() and pairwise_csv.exists() and tok_csv.exists():
        try:
            fmd_df = pd.read_csv(pairwise_csv)
            fmd_df = fmd_df[fmd_df["valid"] == True]  # noqa
            tok_df2 = pd.read_csv(tok_csv)
            if "entropy_mean" in tok_df2.columns:
                tok_agg = tok_df2.groupby("variant")["entropy_mean"].mean().reset_index()
                merged = fmd_df.merge(tok_agg, on="variant", how="inner")
                if not merged.empty:
                    fig, ax = plt.subplots(figsize=(7, 5))
                    sns.regplot(data=merged, x="entropy_mean", y="fmd", ax=ax,
                                scatter_kws={"alpha": 0.6}, line_kws={"color": "red"})
                    ax.set_title("FMD vs Token Entropy")
                    ax.set_xlabel("Mean Token Entropy")
                    ax.set_ylabel("FMD")
                    outputs["lakh_fmd_vs_entropy"] = _save(fig, plots_dir / "lakh_fmd_vs_entropy")
        except Exception as exc:
            logger.warning(f"FMD-entropy scatter failed: {exc}")

    # ------------------------------------------------------------------
    # 11. ANOVA table as figure
    # ------------------------------------------------------------------
    anova_csv = report_dir / "anova_table.csv"
    if anova_csv.exists():
        try:
            anova_df = pd.read_csv(anova_csv, index_col=0)
            if not anova_df.empty:
                fig, ax = plt.subplots(figsize=(12, max(2, len(anova_df) * 0.5 + 1)))
                ax.axis("off")
                display_df = anova_df.round(4).reset_index()
                table = ax.table(
                    cellText=display_df.values,
                    colLabels=display_df.columns,
                    cellLoc="center", loc="center",
                )
                table.auto_set_font_size(False)
                table.set_fontsize(8)
                table.scale(1.2, 1.4)
                ax.set_title("Three-Way ANOVA Table (with Interactions)", fontsize=12, pad=20)
                outputs["lakh_anova_table"] = _save(fig, plots_dir / "lakh_anova_table")
        except Exception as exc:
            logger.warning(f"ANOVA table figure failed: {exc}")

    # ------------------------------------------------------------------
    if not outputs:
        logger.warning("No Lakh plots generated – missing report files in " + str(report_dir))
    else:
        logger.info(f"Generated {len(outputs)} Lakh validation plots in {plots_dir}")

    return outputs

