r"""Generate LaTeX (booktabs) tables for the paper directly from result CSVs.

Every table in the draft is produced from
``results/reports/sensitivity_pivot/*.csv`` so that no number is ever typed
by hand (directly addresses the "hard-coded values" critique). Output goes to
``results/reports/sensitivity_pivot/tables/*.tex`` and the draft \input{}s them.

Run after the sensitivity pipeline finishes:
    python scripts/generate_draft_tables.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REPORTS = Path("results/reports/sensitivity_pivot")
OUT = REPORTS / "tables"
OUT.mkdir(parents=True, exist_ok=True)

CONFIG_ORDER = ["MusicBERT-REMI", "MusicBERT-TSD", "CLaMP2-MTF", "CLaMP1-ABC"]
PERT_ORDER = ["no_velocity", "quantized_time", "constant_tempo", "all_combined"]
PERT_LABEL = {
    "no_velocity": "Velocity removed",
    "quantized_time": "Timing quantized",
    "constant_tempo": "Tempo flattened",
    "all_combined": "All combined",
}


def _tex_escape(s: str) -> str:
    return str(s).replace("_", r"\_")


def _write(name: str, body: str) -> None:
    (OUT / name).write_text(body, encoding="utf-8")
    print(f"  wrote {OUT / name}")


def table_self_similarity() -> None:
    df = pd.read_csv(REPORTS / "self_similarity.csv")
    piv = df.pivot_table(index="configuration", columns="dataset",
                         values="split_half_fmd").reindex(CONFIG_ORDER)
    datasets = list(piv.columns)
    header = " & ".join(["Configuration"] + [_tex_escape(d) for d in datasets])
    lines = [
        r"\begin{tabular}{@{}l" + "c" * len(datasets) + r"@{}}",
        r"\toprule", header + r" \\", r"\midrule",
    ]
    for cfg in CONFIG_ORDER:
        if cfg not in piv.index:
            continue
        vals = " & ".join(f"{piv.loc[cfg, d]:.3f}" for d in datasets)
        lines.append(rf"\texttt{{{_tex_escape(cfg)}}} & {vals} \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    _write("tab_self_similarity.tex", "\n".join(lines))


def table_perturbation() -> None:
    df = pd.read_csv(REPORTS / "perturbation_sensitivity.csv")
    pcol = "perm_p_value" if "perm_p_value" in df.columns else "boot_p_value"
    has_stats = "snr" in df.columns and pcol in df.columns
    lines = [
        r"\begin{tabular}{@{}llcccc@{}}",
        r"\toprule",
        r"Configuration & Perturbation & FMD & SNR & 95\% CI & perm.\ $p$ \\",
        r"\midrule",
    ]
    for cfg in CONFIG_ORDER:
        sub = df[df["configuration"] == cfg]
        if sub.empty:
            continue
        for p in PERT_ORDER:
            r = sub[sub["perturbation"] == p]
            if r.empty:
                continue
            r = r.iloc[0]
            fmd = f"{r['fmd_vs_original']:.4f}"
            if has_stats:
                snr = f"{r['snr']:.2f}"
                ci = f"[{r['fmd_ci_lower']:.3f},\\,{r['fmd_ci_upper']:.3f}]"
                pv = "$<$0.01" if r[pcol] < 0.01 else f"{r[pcol]:.2f}"
                sig = r"\textbf{" if bool(r.get("significant", False)) else "{"
                snr = sig + snr + "}"
            else:
                snr = ci = pv = "--"
            lines.append(
                rf"\texttt{{{_tex_escape(cfg)}}} & {PERT_LABEL[p]} & {fmd} & {snr} & {ci} & {pv} \\"
            )
        lines.append(r"\addlinespace")
    lines += [r"\bottomrule", r"\end{tabular}"]
    _write("tab_perturbation.tex", "\n".join(lines))


def table_spearman() -> None:
    path = REPORTS / "spearman_ranking_agreement.csv"
    if not path.exists() or path.stat().st_size < 5:
        _write("tab_spearman.tex", "% spearman file empty\n")
        return
    df = pd.read_csv(path)
    lines = [
        r"\begin{tabular}{@{}llccc@{}}",
        r"\toprule",
        r"Configuration A & Configuration B & $\rho$ & $p$ & $n$ \\",
        r"\midrule",
    ]
    for _, r in df.iterrows():
        pv = r.get("p_value", float("nan"))
        if pd.isna(pv):
            pv_s = "n/a"
        elif pv < 0.001:
            pv_s = r"$<$0.001"
        else:
            pv_s = f"{pv:.3f}"
        rho = f"{r['spearman_tau']:.3f}"
        if not pd.isna(pv) and pv < 0.05:   # bold = statistically significant
            rho = r"\textbf{" + rho + "}"
        lines.append(
            rf"\texttt{{{_tex_escape(r['config_a'])}}} & "
            rf"\texttt{{{_tex_escape(r['config_b'])}}} & {rho} & {pv_s} & {int(r['n_pairs'])} \\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    _write("tab_spearman.tex", "\n".join(lines))


def table_cross_dataset() -> None:
    """Cross-dataset FMD normalized by each config's noise floor (separation
    ratio): comparable across configs despite different embedding scales."""
    cross = pd.read_csv(REPORTS / "cross_dataset_fmd.csv")
    ss = pd.read_csv(REPORTS / "self_similarity.csv")
    floor = (ss.groupby("configuration")["split_half_fmd"].mean()).to_dict()
    cross["sep_ratio"] = cross.apply(
        lambda r: r["fmd"] / floor.get(r["configuration"], np.nan), axis=1
    )
    piv = cross.pivot_table(index="pair", columns="configuration",
                            values="sep_ratio")
    cols = [c for c in CONFIG_ORDER if c in piv.columns]
    piv = piv[cols].sort_values(cols[0], ascending=False)
    header = " & ".join(["Dataset pair"] + [_tex_escape(c) for c in cols])
    lines = [
        r"\begin{tabular}{@{}l" + "c" * len(cols) + r"@{}}",
        r"\toprule", header + r" \\", r"\midrule",
    ]
    for pair, row in piv.iterrows():
        vals = " & ".join(f"{row[c]:.2f}" for c in cols)
        lines.append(rf"{_tex_escape(pair)} & {vals} \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    _write("tab_cross_dataset.tex", "\n".join(lines))


def table_bootstrap() -> None:
    df = pd.read_csv(REPORTS / "bootstrap_stability.csv").set_index("configuration")
    df = df.reindex([c for c in CONFIG_ORDER if c in df.index])
    lines = [
        r"\begin{tabular}{@{}lcccc@{}}",
        r"\toprule",
        r"Configuration & mean & std & 95\% CI & CV \\",
        r"\midrule",
    ]
    for cfg, r in df.iterrows():
        lines.append(
            rf"\texttt{{{_tex_escape(cfg)}}} & {r['fmd_mean']:.4f} & {r['fmd_std']:.4f} & "
            rf"[{r['fmd_ci_lower']:.3f},\,{r['fmd_ci_upper']:.3f}] & {r['cv']*100:.1f}\% \\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    _write("tab_bootstrap.tex", "\n".join(lines))


if __name__ == "__main__":
    print("Generating LaTeX tables from result CSVs:")
    for fn in (table_self_similarity, table_perturbation, table_spearman,
               table_cross_dataset, table_bootstrap):
        try:
            fn()
        except Exception as e:  # noqa: BLE001
            print(f"  [skip] {fn.__name__}: {e}")
    print(f"Done -> {OUT}/")
