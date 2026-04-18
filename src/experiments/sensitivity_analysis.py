"""Statistical sensitivity analysis for FMD across pipeline variants.

Publication-quality analysis: three-way ANOVA, variance decomposition (η²),
post-hoc Tukey HSD, Kruskal-Wallis, Cohen's d effect sizes, and summary
tables exported to CSV / JSON.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger
from scipy import stats as sp_stats

# Optional: statsmodels for ANOVA / Tukey
try:
    import statsmodels.api as sm
    from statsmodels.formula.api import ols
    from statsmodels.stats.multicomp import pairwise_tukeyhsd

    _HAS_STATSMODELS = True
except ImportError:  # pragma: no cover
    _HAS_STATSMODELS = False
    logger.warning(
        "statsmodels not installed – ANOVA/Tukey will use scipy fallback. "
        "Install with: pip install statsmodels"
    )


# ======================================================================
# Data container
# ======================================================================

@dataclass
class SensitivityResult:
    """Container for all sensitivity analysis outputs."""

    # Raw data
    fmd_df: pd.DataFrame = field(default_factory=pd.DataFrame)

    # ANOVA
    anova_table: Optional[pd.DataFrame] = None
    eta_squared: Optional[Dict[str, float]] = None

    # Post-hoc
    tukey_tokenizer: Optional[pd.DataFrame] = None
    tukey_model: Optional[pd.DataFrame] = None

    # Nonparametric
    kruskal_tokenizer: Optional[Dict[str, float]] = None
    kruskal_model: Optional[Dict[str, float]] = None

    # Effect sizes
    cohens_d: Optional[Dict[str, float]] = None

    # Summary per variant
    variant_summary: Optional[pd.DataFrame] = None


# ======================================================================
# Core analysis functions
# ======================================================================

def _cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """Compute Cohen's d for two independent samples."""
    n1, n2 = len(a), len(b)
    if n1 < 2 or n2 < 2:
        return 0.0
    var1 = np.var(a, ddof=1)
    var2 = np.var(b, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    if pooled_std < 1e-12:
        return 0.0
    return float((np.mean(a) - np.mean(b)) / pooled_std)


def build_fmd_dataframe(bootstrap_rows: List[Dict]) -> pd.DataFrame:
    """Convert bootstrap FMD rows into a tidy DataFrame.

    Expected row keys: ``variant``, ``tokenizer``, ``model``,
    ``remove_velocity``, ``hard_quantization``, ``fmd`` (or
    ``bootstrap_mean``).
    """
    records = []
    for row in bootstrap_rows:
        fmd_val = row.get("fmd") or row.get("bootstrap_mean")
        if fmd_val is None:
            continue
        records.append(
            {
                "variant": row["variant"],
                "tokenizer": str(row["tokenizer"]),
                "model": str(row["model"]),
                "remove_velocity": bool(row.get("remove_velocity", False)),
                "hard_quantization": bool(row.get("hard_quantization", False)),
                "fmd": float(fmd_val),
                "bootstrap_std": float(row.get("bootstrap_std") or 0.0),
                "ci_lower": float(row.get("bootstrap_ci_lower") or fmd_val),
                "ci_upper": float(row.get("bootstrap_ci_upper") or fmd_val),
            }
        )
    return pd.DataFrame(records)


# ------------------------------------------------------------------
# ANOVA
# ------------------------------------------------------------------

def run_three_way_anova(df: pd.DataFrame) -> Tuple[Optional[pd.DataFrame], Optional[Dict[str, float]]]:
    """Run three-way ANOVA (tokenizer × model × preprocessing).

    Preprocessing is encoded as a single factor with 4 levels
    (vel_on/quant_off, vel_off/quant_off, vel_on/quant_on, vel_off/quant_on).

    Returns (anova_table, eta_squared_dict).
    """
    if df.empty:
        return None, None

    work = df.copy()
    work["preprocess"] = (
        work["remove_velocity"].astype(str) + "_" + work["hard_quantization"].astype(str)
    )

    if _HAS_STATSMODELS:
        try:
            formula = "fmd ~ C(tokenizer) * C(model) * C(preprocess)"
            model = ols(formula, data=work).fit()
            anova_table = sm.stats.anova_lm(model, typ=2)

            # η² = SS_factor / SS_total
            ss_total = anova_table["sum_sq"].sum()
            eta_sq = {}
            for idx in anova_table.index:
                if idx == "Residual":
                    continue
                eta_sq[idx] = float(anova_table.loc[idx, "sum_sq"] / ss_total) if ss_total > 0 else 0.0

            logger.info("Three-way ANOVA completed (statsmodels)")
            return anova_table, eta_sq
        except Exception as exc:
            logger.warning(f"statsmodels ANOVA failed: {exc}; falling back to scipy")

    # Scipy fallback: one-way per factor
    eta_sq: Dict[str, float] = {}
    rows: List[Dict[str, Any]] = []

    for factor in ("tokenizer", "model", "preprocess"):
        groups = [g["fmd"].values for _, g in work.groupby(factor)]
        if len(groups) < 2:
            continue
        stat, p = sp_stats.f_oneway(*groups)
        ss_between = sum(len(g) * (np.mean(g) - work["fmd"].mean()) ** 2 for g in groups)
        ss_total = ((work["fmd"] - work["fmd"].mean()) ** 2).sum()
        eta = float(ss_between / ss_total) if ss_total > 0 else 0.0
        eta_sq[factor] = eta
        rows.append({"source": factor, "F": float(stat), "p": float(p), "eta_sq": eta})

    anova_table = pd.DataFrame(rows)
    logger.info("ANOVA completed (scipy one-way fallback)")
    return anova_table, eta_sq


# ------------------------------------------------------------------
# Post-hoc tests
# ------------------------------------------------------------------

def run_tukey_hsd(df: pd.DataFrame, factor: str) -> Optional[pd.DataFrame]:
    """Run Tukey HSD on *factor* (e.g. ``'tokenizer'``)."""
    if df.empty or factor not in df.columns:
        return None
    if not _HAS_STATSMODELS:
        logger.warning("Tukey HSD requires statsmodels")
        return None

    try:
        result = pairwise_tukeyhsd(df["fmd"].values, df[factor].values, alpha=0.05)
        tukey_df = pd.DataFrame(
            data=result._results_table.data[1:],
            columns=result._results_table.data[0],
        )
        logger.info(f"Tukey HSD for '{factor}' completed ({len(tukey_df)} comparisons)")
        return tukey_df
    except Exception as exc:
        logger.warning(f"Tukey HSD failed for '{factor}': {exc}")
        return None


# ------------------------------------------------------------------
# Non-parametric
# ------------------------------------------------------------------

def run_kruskal_wallis(df: pd.DataFrame, factor: str) -> Optional[Dict[str, float]]:
    """Kruskal-Wallis test for *factor*."""
    if df.empty or factor not in df.columns:
        return None
    groups = [g["fmd"].values for _, g in df.groupby(factor)]
    if len(groups) < 2:
        return None
    stat, p = sp_stats.kruskal(*groups)
    return {"H": float(stat), "p": float(p)}


# ------------------------------------------------------------------
# Effect sizes between factor levels
# ------------------------------------------------------------------

def compute_pairwise_cohens_d(df: pd.DataFrame, factor: str) -> Dict[str, float]:
    """Cohen's d for every pair of levels in *factor*."""
    results: Dict[str, float] = {}
    levels = sorted(df[factor].unique())
    for a, b in combinations(levels, 2):
        vals_a = df.loc[df[factor] == a, "fmd"].values
        vals_b = df.loc[df[factor] == b, "fmd"].values
        results[f"{a} vs {b}"] = _cohens_d(vals_a, vals_b)
    return results


# ------------------------------------------------------------------
# Variant-level summary
# ------------------------------------------------------------------

def variant_summary_table(df: pd.DataFrame) -> pd.DataFrame:
    """Per-variant descriptive statistics."""
    if df.empty:
        return pd.DataFrame()
    summary = (
        df.groupby("variant")
        .agg(
            mean_fmd=("fmd", "mean"),
            std_fmd=("fmd", "std"),
            ci_lower=("ci_lower", "mean"),
            ci_upper=("ci_upper", "mean"),
            n=("fmd", "count"),
        )
        .reset_index()
        .sort_values("mean_fmd")
    )
    return summary


# ======================================================================
# High-level runner
# ======================================================================

def run_sensitivity_analysis(
    bootstrap_rows: List[Dict],
    output_dir: Path | str = "results/reports/lakh",
) -> SensitivityResult:
    """Run the full statistical sensitivity analysis and save artefacts.

    Parameters
    ----------
    bootstrap_rows:
        List of dicts, one per (variant, bootstrap_resample) or per variant.
        Must contain keys: variant, tokenizer, model, remove_velocity,
        hard_quantization, fmd (or bootstrap_mean), bootstrap_std,
        bootstrap_ci_lower, bootstrap_ci_upper.
    output_dir:
        Directory to write CSV/JSON outputs.

    Returns
    -------
    SensitivityResult with all computed tables.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = build_fmd_dataframe(bootstrap_rows)
    if df.empty:
        logger.warning("No usable FMD data for sensitivity analysis")
        return SensitivityResult()

    logger.info(f"Sensitivity analysis: {len(df)} rows, {df['variant'].nunique()} variants")

    # 1. ANOVA + η²
    anova_table, eta_sq = run_three_way_anova(df)

    # 2. Tukey HSD
    tukey_tok = run_tukey_hsd(df, "tokenizer")
    tukey_mod = run_tukey_hsd(df, "model")

    # 3. Kruskal-Wallis
    kw_tok = run_kruskal_wallis(df, "tokenizer")
    kw_mod = run_kruskal_wallis(df, "model")

    # 4. Effect sizes
    cohens: Dict[str, float] = {}
    cohens.update({f"tok_{k}": v for k, v in compute_pairwise_cohens_d(df, "tokenizer").items()})
    cohens.update({f"mod_{k}": v for k, v in compute_pairwise_cohens_d(df, "model").items()})

    # 5. Summary
    summary = variant_summary_table(df)

    # ------------------------------------------------------------------
    # Save outputs
    # ------------------------------------------------------------------

    # CSV – variant summary
    summary.to_csv(output_dir / "variant_summary.csv", index=False)

    # CSV – ANOVA table
    if anova_table is not None:
        anova_table.to_csv(output_dir / "anova_table.csv")

    # CSV – Tukey
    if tukey_tok is not None:
        tukey_tok.to_csv(output_dir / "tukey_tokenizer.csv", index=False)
    if tukey_mod is not None:
        tukey_mod.to_csv(output_dir / "tukey_model.csv", index=False)

    # JSON – aggregated results
    json_payload = {
        "eta_squared": eta_sq,
        "kruskal_wallis_tokenizer": kw_tok,
        "kruskal_wallis_model": kw_mod,
        "cohens_d": cohens,
        "n_variants": int(df["variant"].nunique()),
        "n_rows": int(len(df)),
    }
    with open(output_dir / "sensitivity_results.json", "w", encoding="utf-8") as fh:
        json.dump(json_payload, fh, indent=2, default=str)

    logger.info(f"Sensitivity analysis outputs saved to {output_dir}")

    return SensitivityResult(
        fmd_df=df,
        anova_table=anova_table,
        eta_squared=eta_sq,
        tukey_tokenizer=tukey_tok,
        tukey_model=tukey_mod,
        kruskal_tokenizer=kw_tok,
        kruskal_model=kw_mod,
        cohens_d=cohens,
        variant_summary=summary,
    )


