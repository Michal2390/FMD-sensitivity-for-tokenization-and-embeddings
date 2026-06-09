"""Generate publication-grade figures for the paper from result CSVs.

Key methodological choice: raw FMD is NOT comparable across embedding spaces
(CLaMP-1/2 are L2-normalised -> unit sphere -> small FMD; MusicBERT is
unnormalised -> large FMD). Every cross-configuration view here is therefore
either (a) a per-configuration signal-to-noise ratio (SNR = FMD / split-half
noise floor) or (b) a rank-based statistic (Spearman) -- both scale-invariant.

Outputs -> results/plots/sensitivity_pivot/paper/
Run after the sensitivity pipeline finishes:
    python scripts/generate_draft_figures.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPORTS = Path("results/reports/sensitivity_pivot")
OUT = Path("results/plots/sensitivity_pivot/paper")
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.titlesize": 12,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

CONFIG_ORDER = ["MusicBERT-REMI", "MusicBERT-TSD", "CLaMP2-MTF", "CLaMP1-ABC"]
PERT_ORDER = ["no_velocity", "quantized_time", "constant_tempo", "all_combined"]
PERT_LABEL = {
    "no_velocity": "Velocity\nremoved",
    "quantized_time": "Timing\nquantized",
    "constant_tempo": "Tempo\nflattened",
    "all_combined": "All\ncombined",
}

ss = pd.read_csv(REPORTS / "self_similarity.csv")
pert = pd.read_csv(REPORTS / "perturbation_sensitivity.csv")
cross = pd.read_csv(REPORTS / "cross_dataset_fmd.csv")
boot = pd.read_csv(REPORTS / "bootstrap_stability.csv")
spear_path = REPORTS / "spearman_ranking_agreement.csv"

# noise floor for perturbations = MAESTRO split-half (the perturbation dataset)
floor_maestro = (ss[ss["dataset"] == "maestro"]
                 .set_index("configuration")["split_half_fmd"].to_dict())
present = [c for c in CONFIG_ORDER if c in pert["configuration"].unique()]


def fig1_noise_floor():
    piv = ss.pivot_table(index="configuration", columns="dataset",
                         values="split_half_fmd").reindex(present)
    datasets = list(piv.columns)
    x = np.arange(len(present))
    w = 0.8 / max(len(datasets), 1)
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    for i, d in enumerate(datasets):
        ax.bar(x + (i - len(datasets) / 2) * w + w / 2, piv[d], w, label=d)
    ax.set_yscale("log")
    ax.set_ylabel("Split-half FMD  (log scale)")
    ax.set_title("Per-configuration noise floor (self-similarity)")
    ax.set_xticks(x); ax.set_xticklabels(present, rotation=20, ha="right")
    ax.legend(frameon=False, fontsize=8, ncol=2)
    ax.grid(axis="y", ls=":", alpha=0.5)
    fig.savefig(OUT / "fig1_noise_floor.png"); plt.close(fig)


def _snr_matrix():
    snr = np.full((len(PERT_ORDER), len(present)), np.nan)
    sig = np.zeros_like(snr, dtype=bool)
    for j, cfg in enumerate(present):
        for i, p in enumerate(PERT_ORDER):
            r = pert[(pert["configuration"] == cfg) & (pert["perturbation"] == p)]
            if r.empty:
                continue
            r = r.iloc[0]
            if "snr" in pert.columns and not pd.isna(r["snr"]):
                snr[i, j] = r["snr"]
            else:
                snr[i, j] = r["fmd_vs_original"] / floor_maestro.get(cfg, np.nan)
            if "significant" in pert.columns:
                sig[i, j] = bool(r["significant"])
            else:
                sig[i, j] = snr[i, j] >= 1.0
    return snr, sig


def fig2_snr_heatmap():
    snr, sig = _snr_matrix()
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    im = ax.imshow(snr, cmap="RdYlGn_r", aspect="auto",
                   norm=mcolors.LogNorm(vmin=0.05, vmax=4.0))
    ax.set_xticks(range(len(present))); ax.set_xticklabels(present, rotation=20, ha="right")
    ax.set_yticks(range(len(PERT_ORDER))); ax.set_yticklabels([PERT_LABEL[p] for p in PERT_ORDER])
    for i in range(len(PERT_ORDER)):
        for j in range(len(present)):
            v = snr[i, j]
            if np.isnan(v):
                continue
            txt = f"{v:.2f}" + ("\n(sig.)" if sig[i, j] else "")
            ax.text(j, i, txt, ha="center", va="center", fontsize=9,
                    fontweight="bold" if sig[i, j] else "normal")
    ax.set_title("Perturbation detectability: SNR = FMD / noise floor\n"
                 "(sig. = bootstrap-significant above the noise floor)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04).set_label("SNR (log scale)")
    fig.savefig(OUT / "fig2_snr_heatmap.png"); plt.close(fig)


def fig3_bootstrap_cv():
    bo = boot.set_index("configuration").reindex(present)
    colors = ["#C44E52" if "MusicBERT" in c else "#55A868" for c in present]
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    ax.bar(range(len(present)), bo["cv"] * 100, color=colors)
    for i, c in enumerate(present):
        ax.text(i, bo.loc[c, "cv"] * 100 + 0.4, f"{bo.loc[c,'cv']*100:.1f}%",
                ha="center", fontsize=9)
    ax.set_ylabel("Coefficient of variation (%)")
    ax.set_title("Bootstrap stability of cross-dataset FMD\n(MAESTRO vs POP909, 50 resamples)")
    ax.set_xticks(range(len(present))); ax.set_xticklabels(present, rotation=20, ha="right")
    ax.grid(axis="y", ls=":", alpha=0.5)
    fig.savefig(OUT / "fig3_bootstrap_cv.png"); plt.close(fig)


def fig4_perturbation_significance():
    """Forest plot: SNR with bootstrap CI per (config, perturbation)."""
    if "snr" not in pert.columns or "noise_floor_mean" not in pert.columns:
        print("  [skip] fig4: perturbation CSV lacks bootstrap columns")
        return
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    ylabels, y = [], 0
    yticks = []
    for cfg in present:
        for p in PERT_ORDER:
            r = pert[(pert["configuration"] == cfg) & (pert["perturbation"] == p)]
            if r.empty:
                continue
            r = r.iloc[0]
            nf = r["noise_floor_mean"]
            snr = r["snr"]
            lo = r["fmd_ci_lower"] / nf if nf > 0 else snr
            hi = r["fmd_ci_upper"] / nf if nf > 0 else snr
            sig = bool(r.get("significant", snr >= 1))
            color = "#C44E52" if sig else "#888888"
            ax.plot([lo, hi], [y, y], color=color, lw=2, zorder=2)
            ax.scatter([snr], [y], color=color, s=30, zorder=3)
            ylabels.append(f"{cfg} · {p.replace('_',' ')}")
            yticks.append(y)
            y += 1
        y += 0.6
    ax.axvline(1.0, color="k", ls="--", lw=1, alpha=0.7)
    ax.text(1.02, ax.get_ylim()[1], "detection\nthreshold", fontsize=8, va="top")
    ax.set_xscale("log")
    ax.set_yticks(yticks); ax.set_yticklabels(ylabels, fontsize=8)
    ax.set_xlabel("SNR (FMD / noise floor), bootstrap 95% CI — log scale")
    ax.set_title("Perturbation significance per configuration")
    ax.grid(axis="x", ls=":", alpha=0.5)
    fig.savefig(OUT / "fig4_perturbation_significance.png"); plt.close(fig)


def fig5_spearman_heatmap():
    if not spear_path.exists() or spear_path.stat().st_size < 5:
        print("  [skip] fig5: spearman file empty"); return
    sp = pd.read_csv(spear_path)
    if sp.empty:
        print("  [skip] fig5: spearman empty"); return
    cfgs = present
    M = np.full((len(cfgs), len(cfgs)), np.nan)
    for i, a in enumerate(cfgs):
        M[i, i] = 1.0
        for j, b in enumerate(cfgs):
            row = sp[((sp.config_a == a) & (sp.config_b == b)) |
                     ((sp.config_a == b) & (sp.config_b == a))]
            if not row.empty:
                M[i, j] = row.iloc[0]["spearman_tau"]
    fig, ax = plt.subplots(figsize=(5.4, 4.6))
    im = ax.imshow(M, cmap="RdBu", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(cfgs))); ax.set_xticklabels(cfgs, rotation=25, ha="right", fontsize=8)
    ax.set_yticks(range(len(cfgs))); ax.set_yticklabels(cfgs, fontsize=8)
    interp = bool(sp.get("interpretable", pd.Series([False])).any())
    npairs = int(sp["n_pairs"].iloc[0]) if "n_pairs" in sp.columns else 0
    for i in range(len(cfgs)):
        for j in range(len(cfgs)):
            if not np.isnan(M[i, j]):
                ax.text(j, i, f"{M[i,j]:.2f}", ha="center", va="center", fontsize=9)
    ax.set_title(f"Cross-configuration rank agreement\nSpearman $\\rho$ over {npairs} dataset pairs"
                 + ("" if interp else "  (n<10: indicative)"))
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04).set_label(r"Spearman $\rho$")
    fig.savefig(OUT / "fig5_spearman_heatmap.png"); plt.close(fig)


def fig6_cross_dataset_sep():
    floor_all = ss.groupby("configuration")["split_half_fmd"].mean().to_dict()
    c = cross.copy()
    c["sep"] = c.apply(lambda r: r["fmd"] / floor_all.get(r["configuration"], np.nan), axis=1)
    piv = c.pivot_table(index="pair", columns="configuration", values="sep")
    cols = [x for x in CONFIG_ORDER if x in piv.columns]
    if not cols:
        print("  [skip] fig6: no configs"); return
    piv = piv[cols].sort_values(cols[0], ascending=False)
    fig, ax = plt.subplots(figsize=(7.0, max(4.0, 0.32 * len(piv))))
    im = ax.imshow(piv.values, cmap="viridis", aspect="auto",
                   norm=mcolors.LogNorm(vmin=max(piv.values[piv.values > 0].min(), 0.1),
                                        vmax=piv.values.max()))
    ax.set_xticks(range(len(cols))); ax.set_xticklabels(cols, rotation=20, ha="right", fontsize=9)
    ax.set_yticks(range(len(piv))); ax.set_yticklabels(list(piv.index), fontsize=8)
    for i in range(len(piv)):
        for j in range(len(cols)):
            v = piv.values[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=7,
                        color="white" if v < piv.values.max() * 0.5 else "black")
    ax.set_title("Cross-dataset separation ratio\n(FMD / noise floor; >1 = above sampling noise)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04).set_label("separation ratio (log)")
    fig.savefig(OUT / "fig6_cross_dataset_sep.png"); plt.close(fig)


def print_summary():
    snr, sig = _snr_matrix()
    print("\nPerturbation SNR (rows=perturbation, cols=config):")
    print("            " + " ".join(f"{c[:12]:>13s}" for c in present))
    for i, p in enumerate(PERT_ORDER):
        print(f"{p:14s} " + " ".join(
            f"{snr[i,j]:6.2f}{'*' if sig[i,j] else ' '}     " for j in range(len(present))))
    print("  (* = bootstrap-significant above noise floor)")


if __name__ == "__main__":
    print("Generating figures from result CSVs:")
    for fn in (fig1_noise_floor, fig2_snr_heatmap, fig3_bootstrap_cv,
               fig4_perturbation_significance, fig5_spearman_heatmap,
               fig6_cross_dataset_sep):
        try:
            fn(); print(f"  ok  {fn.__name__}")
        except Exception as e:  # noqa: BLE001
            print(f"  [skip] {fn.__name__}: {e}")
    print_summary()
    print(f"\nFigures -> {OUT}/")
