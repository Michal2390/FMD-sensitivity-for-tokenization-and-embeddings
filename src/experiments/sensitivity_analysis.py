"""Statistical sensitivity analysis for FMD across pipeline variants.

Publication-quality analysis: three-way ANOVA with interactions,
variance decomposition (η² and partial η²), post-hoc Tukey HSD,
Kruskal-Wallis, Cohen's d effect sizes, permutation tests,
and summary tables exported to CSV / JSON.
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
    from statsmodels.stats.multitest import multipletests

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
    partial_eta_squared: Optional[Dict[str, float]] = None

    # Bootstrap CI for η²
    eta_sq_ci: Optional[Dict[str, Tuple[float, float, float]]] = None

    # Post-hoc
    tukey_tokenizer: Optional[pd.DataFrame] = None
    tukey_model: Optional[pd.DataFrame] = None

    # Corrected p-values
    corrected_pvalues: Optional[pd.DataFrame] = None

    # Nonparametric
    kruskal_tokenizer: Optional[Dict[str, float]] = None
    kruskal_model: Optional[Dict[str, float]] = None

    # Permutation tests
    permutation_results: Optional[Dict[str, Dict]] = None

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


# ------------------------------------------------------------------
# Bootstrap CI for η²
# ------------------------------------------------------------------

def _compute_eta_squared_oneway(values: np.ndarray, labels: np.ndarray) -> float:
    """Compute one-way η² = SS_between / SS_total."""
    grand_mean = values.mean()
    ss_total = ((values - grand_mean) ** 2).sum()
    if ss_total < 1e-15:
        return 0.0
    unique_labels = np.unique(labels)
    ss_between = sum(
        np.sum(labels == lbl) * (values[labels == lbl].mean() - grand_mean) ** 2
        for lbl in unique_labels
    )
    return float(ss_between / ss_total)


def bootstrap_eta_squared(
    df: pd.DataFrame,
    factor: str,
    n_bootstrap: int = 5000,
    ci: float = 0.95,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """Bootstrap confidence interval for η² (one-way, percentile method).

    Parameters
    ----------
    df : DataFrame with 'fmd' column and *factor* column.
    factor : Column name for grouping.
    n_bootstrap : Number of bootstrap resamples.
    ci : Confidence level (default 0.95).
    seed : Random seed.

    Returns
    -------
    (point_estimate, ci_lower, ci_upper)
    """
    if df.empty or factor not in df.columns:
        return (0.0, 0.0, 0.0)

    values = df["fmd"].values
    labels = df[factor].values
    n = len(values)

    point_est = _compute_eta_squared_oneway(values, labels)

    rng = np.random.default_rng(seed)
    boot_etas = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        boot_etas[i] = _compute_eta_squared_oneway(values[idx], labels[idx])

    alpha = 1 - ci
    lower = float(np.percentile(boot_etas, 100 * alpha / 2))
    upper = float(np.percentile(boot_etas, 100 * (1 - alpha / 2)))

    logger.info(
        f"Bootstrap η² for '{factor}': {point_est:.4f} "
        f"[{lower:.4f}, {upper:.4f}] ({n_bootstrap} resamples)"
    )
    return (point_est, lower, upper)


# ------------------------------------------------------------------
# Multiple comparison correction
# ------------------------------------------------------------------

def apply_multiple_comparison_correction(
    p_values: List[float],
    method: str = "holm",
) -> np.ndarray:
    """Apply multiple comparison correction to a list of p-values.

    Parameters
    ----------
    p_values : Raw p-values.
    method : 'holm' or 'bonferroni'.

    Returns
    -------
    Array of adjusted p-values.
    """
    if not _HAS_STATSMODELS:
        logger.warning("statsmodels required for multiple comparison correction")
        return np.array(p_values)

    if not p_values:
        return np.array([])

    _, p_adj, _, _ = multipletests(p_values, method=method)
    return p_adj


def correct_tukey_pvalues(tukey_df: pd.DataFrame) -> pd.DataFrame:
    """Add Holm and Bonferroni corrected p-values to a Tukey HSD result DataFrame.

    Parameters
    ----------
    tukey_df : DataFrame from ``run_tukey_hsd()`` with a 'p-adj' column.

    Returns
    -------
    DataFrame with added 'p_adj_holm' and 'p_adj_bonf' columns.
    """
    if tukey_df is None or tukey_df.empty:
        return tukey_df

    raw_p = tukey_df["p-adj"].astype(float).tolist()
    tukey_df = tukey_df.copy()
    tukey_df["p_adj_holm"] = apply_multiple_comparison_correction(raw_p, method="holm")
    tukey_df["p_adj_bonf"] = apply_multiple_comparison_correction(raw_p, method="bonferroni")
    tukey_df["still_sig_holm"] = tukey_df["p_adj_holm"] < 0.05
    tukey_df["still_sig_bonf"] = tukey_df["p_adj_bonf"] < 0.05
    return tukey_df


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
# ANOVA (with interaction effects)
# ------------------------------------------------------------------

def run_three_way_anova(df: pd.DataFrame) -> Tuple[Optional[pd.DataFrame], Optional[Dict[str, float]], Optional[Dict[str, float]]]:
    """Run three-way ANOVA (tokenizer × model × preprocessing) with interactions.

    Returns (anova_table, eta_squared_dict, partial_eta_squared_dict).
    """
    if df.empty:
        return None, None, None

    work = df.copy()
    work["preprocess"] = (
        work["remove_velocity"].astype(str) + "_" + work["hard_quantization"].astype(str)
    )

    if _HAS_STATSMODELS:
        try:
            # Full 3-way ANOVA with all interaction effects
            formula = "fmd ~ C(tokenizer) * C(model) * C(preprocess)"
            model = ols(formula, data=work).fit()
            anova_table = sm.stats.anova_lm(model, typ=2)

            # η² = SS_factor / SS_total
            ss_total = anova_table["sum_sq"].sum()
            eta_sq = {}
            partial_eta_sq = {}
            ss_residual = anova_table.loc["Residual", "sum_sq"] if "Residual" in anova_table.index else 0.0

            for idx in anova_table.index:
                if idx == "Residual":
                    continue
                ss_effect = anova_table.loc[idx, "sum_sq"]
                # η² (eta-squared)
                eta_sq[idx] = float(ss_effect / ss_total) if ss_total > 0 else 0.0
                # Partial η² = SS_effect / (SS_effect + SS_residual)
                denom = ss_effect + ss_residual
                partial_eta_sq[idx] = float(ss_effect / denom) if denom > 0 else 0.0

            logger.info("Three-way ANOVA with interactions completed (statsmodels)")
            return anova_table, eta_sq, partial_eta_sq
        except Exception as exc:
            logger.warning(f"statsmodels ANOVA failed: {exc}; falling back to scipy")

    # Scipy fallback: one-way per factor
    eta_sq: Dict[str, float] = {}
    partial_eta_sq: Dict[str, float] = {}
    rows: List[Dict[str, Any]] = []

    for factor in ("tokenizer", "model", "preprocess"):
        groups = [g["fmd"].values for _, g in work.groupby(factor)]
        if len(groups) < 2:
            continue
        stat, p = sp_stats.f_oneway(*groups)
        ss_between = sum(len(g) * (np.mean(g) - work["fmd"].mean()) ** 2 for g in groups)
        ss_total = ((work["fmd"] - work["fmd"].mean()) ** 2).sum()
        ss_within = ss_total - ss_between
        eta = float(ss_between / ss_total) if ss_total > 0 else 0.0
        p_eta = float(ss_between / (ss_between + ss_within)) if (ss_between + ss_within) > 0 else 0.0
        eta_sq[factor] = eta
        partial_eta_sq[factor] = p_eta
        rows.append({"source": factor, "F": float(stat), "p": float(p), "eta_sq": eta, "partial_eta_sq": p_eta})

    anova_table = pd.DataFrame(rows)
    logger.info("ANOVA completed (scipy one-way fallback)")
    return anova_table, eta_sq, partial_eta_sq


# ------------------------------------------------------------------
# Permutation tests
# ------------------------------------------------------------------

def run_permutation_test(
    df: pd.DataFrame,
    factor: str,
    n_permutations: int = 5000,
    seed: int = 42,
) -> Dict[str, float]:
    """Permutation test for a single factor.

    Shuffles factor labels and computes F-statistic distribution
    to get a non-parametric p-value.

    Args:
        df: DataFrame with 'fmd' column and factor column.
        factor: Column name to test.
        n_permutations: Number of permutations.
        seed: Random seed.

    Returns:
        Dict with observed_F, permutation_p, n_permutations.
    """
    if df.empty or factor not in df.columns:
        return {"observed_F": None, "permutation_p": None, "n_permutations": 0}

    groups = [g["fmd"].values for _, g in df.groupby(factor)]
    if len(groups) < 2:
        return {"observed_F": None, "permutation_p": None, "n_permutations": 0}

    observed_F, _ = sp_stats.f_oneway(*groups)

    rng = np.random.default_rng(seed)
    fmd_values = df["fmd"].values.copy()
    n_extreme = 0

    for _ in range(n_permutations):
        # Shuffle FMD values (break association with factor)
        shuffled = rng.permutation(fmd_values)
        shuffled_groups = []
        start = 0
        for g in groups:
            shuffled_groups.append(shuffled[start:start + len(g)])
            start += len(g)
        perm_F, _ = sp_stats.f_oneway(*shuffled_groups)
        if perm_F >= observed_F:
            n_extreme += 1

    p_value = (n_extreme + 1) / (n_permutations + 1)

    logger.info(f"Permutation test for '{factor}': F={observed_F:.4f}, p={p_value:.4f} ({n_permutations} perms)")

    return {
        "observed_F": float(observed_F),
        "permutation_p": float(p_value),
        "n_permutations": n_permutations,
    }


# ------------------------------------------------------------------
# Post-hoc tests
# ------------------------------------------------------------------

def run_tukey_hsd(df: pd.DataFrame, factor: str) -> Optional[pd.DataFrame]:
    """Run Tukey HSD on *factor*."""
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
    n_permutations: int = 5000,
    seed: int = 42,
) -> SensitivityResult:
    """Run the full statistical sensitivity analysis and save artefacts.

    Parameters
    ----------
    bootstrap_rows:
        List of dicts, one per variant.
    output_dir:
        Directory to write CSV/JSON outputs.
    n_permutations:
        Number of permutations for permutation tests.
    seed:
        Random seed for permutation tests.

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

    # 1. ANOVA + η² + partial η²
    anova_table, eta_sq, partial_eta_sq = run_three_way_anova(df)

    # 1b. Bootstrap CI for η²
    work_bs = df.copy()
    work_bs["preprocess"] = (
        work_bs["remove_velocity"].astype(str) + "_" + work_bs["hard_quantization"].astype(str)
    )
    eta_sq_ci: Dict[str, Tuple[float, float, float]] = {}
    for factor in ("tokenizer", "model", "preprocess"):
        eta_sq_ci[factor] = bootstrap_eta_squared(work_bs, factor, n_bootstrap=5000, ci=0.95, seed=seed)
    # Also bootstrap tok×model interaction approximation
    work_bs["tok_model"] = work_bs["tokenizer"] + "_" + work_bs["model"]
    eta_sq_ci["tokenizer:model"] = bootstrap_eta_squared(work_bs, "tok_model", n_bootstrap=5000, ci=0.95, seed=seed)

    # 2. Tukey HSD + multiple comparison correction
    tukey_tok = run_tukey_hsd(df, "tokenizer")
    tukey_mod = run_tukey_hsd(df, "model")
    tukey_tok = correct_tukey_pvalues(tukey_tok)
    tukey_mod = correct_tukey_pvalues(tukey_mod)

    # 3. Kruskal-Wallis
    kw_tok = run_kruskal_wallis(df, "tokenizer")
    kw_mod = run_kruskal_wallis(df, "model")

    # 4. Effect sizes
    cohens: Dict[str, float] = {}
    cohens.update({f"tok_{k}": v for k, v in compute_pairwise_cohens_d(df, "tokenizer").items()})
    cohens.update({f"mod_{k}": v for k, v in compute_pairwise_cohens_d(df, "model").items()})

    # 5. Permutation tests
    work = df.copy()
    work["preprocess"] = (
        work["remove_velocity"].astype(str) + "_" + work["hard_quantization"].astype(str)
    )
    perm_results = {}
    for factor in ("tokenizer", "model", "preprocess"):
        perm_results[factor] = run_permutation_test(work, factor, n_permutations=n_permutations, seed=seed)

    # 6. Summary
    summary = variant_summary_table(df)

    # ------------------------------------------------------------------
    # Save outputs
    # ------------------------------------------------------------------
    summary.to_csv(output_dir / "variant_summary.csv", index=False)

    if anova_table is not None:
        anova_table.to_csv(output_dir / "anova_table.csv")

    if tukey_tok is not None:
        tukey_tok.to_csv(output_dir / "tukey_tokenizer.csv", index=False)
    if tukey_mod is not None:
        tukey_mod.to_csv(output_dir / "tukey_model.csv", index=False)

    json_payload = {
        "eta_squared": eta_sq,
        "partial_eta_squared": partial_eta_sq,
        "eta_sq_bootstrap_ci": {k: {"point": v[0], "ci_lower": v[1], "ci_upper": v[2]} for k, v in eta_sq_ci.items()},
        "kruskal_wallis_tokenizer": kw_tok,
        "kruskal_wallis_model": kw_mod,
        "cohens_d": cohens,
        "permutation_tests": perm_results,
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
        partial_eta_squared=partial_eta_sq,
        eta_sq_ci=eta_sq_ci,
        tukey_tokenizer=tukey_tok,
        tukey_model=tukey_mod,
        corrected_pvalues=None,  # stored in tukey DataFrames
        kruskal_tokenizer=kw_tok,
        kruskal_model=kw_mod,
        permutation_results=perm_results,
        cohens_d=cohens,
        variant_summary=summary,
    )


# ======================================================================
# Linear Mixed-Effects Model (LME)
# ======================================================================

def fit_lme_model(
    df: pd.DataFrame,
    response: str = "fmd",
    fixed_effects: str = "C(tokenizer) + C(model) + C(preprocess)",
    random_effects: str = "pair",
    output_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Fit a Linear Mixed-Effects model to FMD data.

    Supplements ANOVA by properly modelling the hierarchical structure:
    - Fixed effects: tokenizer, model, preprocessing
    - Random effects: genre pair (and optionally MIDI file)

    Uses statsmodels.formula.api.mixedlm.

    Args:
        df: DataFrame with columns: fmd, tokenizer, model, preprocess, pair.
        response: Response variable column name.
        fixed_effects: Patsy formula string for fixed effects.
        random_effects: Column name for random intercept grouping.
        output_dir: If provided, save LME summary to file.

    Returns:
        Dict with keys: summary_text, aic, bic, log_likelihood,
        fixed_effects_table (DataFrame), random_effects_variance.
    """
    if not _HAS_STATSMODELS:
        logger.warning("statsmodels not available — cannot fit LME model")
        return {"error": "statsmodels not installed"}

    from statsmodels.formula.api import mixedlm

    # Ensure preprocess column exists
    if "preprocess" not in df.columns:
        if "remove_velocity" in df.columns and "hard_quantization" in df.columns:
            df = df.copy()
            df["preprocess"] = df["remove_velocity"].astype(str) + "_" + df["hard_quantization"].astype(str)

    # Drop NaN in response column before fitting
    df = df.dropna(subset=[response]).reset_index(drop=True)

    formula = f"{response} ~ {fixed_effects}"
    logger.info(f"Fitting LME: {formula} | random=~1|{random_effects}")

    try:
        model = mixedlm(formula, df, groups=df[random_effects])
        result = model.fit(reml=True)

        summary_text = str(result.summary())
        logger.info(f"LME fit complete. AIC={result.aic:.2f}, BIC={result.bic:.2f}")

        # Extract fixed effects table
        fe_table = pd.DataFrame({
            "coefficient": result.fe_params,
            "std_err": result.bse_fe,
            "z": result.tvalues,
            "p": result.pvalues,
        })

        # Random effects variance
        re_var = float(result.cov_re.iloc[0, 0]) if hasattr(result.cov_re, 'iloc') else float(result.cov_re)

        output = {
            "summary_text": summary_text,
            "aic": float(result.aic),
            "bic": float(result.bic),
            "log_likelihood": float(result.llf),
            "fixed_effects_table": fe_table,
            "random_effects_variance": re_var,
            "n_groups": int(result.nobs),
            "converged": result.converged,
        }

        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            with open(output_dir / "lme_summary.txt", "w") as f:
                f.write(summary_text)
            fe_table.to_csv(output_dir / "lme_fixed_effects.csv")
            logger.info(f"LME results saved to {output_dir}")

        return output

    except Exception as e:
        logger.error(f"LME fitting failed: {e}")
        return {"error": str(e)}


def fit_lme_with_interactions(
    df: pd.DataFrame,
    response: str = "fmd",
    random_effects: str = "pair",
    output_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Fit LME with all two-way interactions between fixed effects.

    Formula: fmd ~ C(tokenizer) * C(model) + C(tokenizer) * C(preprocess) +
                    C(model) * C(preprocess) | random=~1|pair

    Args:
        df: DataFrame with FMD data.
        response: Response variable.
        random_effects: Grouping variable for random intercept.
        output_dir: Output directory.

    Returns:
        Dict with LME results.
    """
    fixed = ("C(tokenizer) * C(model) + C(tokenizer) * C(preprocess) + "
             "C(model) * C(preprocess)")
    return fit_lme_model(df, response=response, fixed_effects=fixed,
                         random_effects=random_effects, output_dir=output_dir)

