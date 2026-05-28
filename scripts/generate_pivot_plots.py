"""Generate publication-quality plots for the sensitivity pivot (professor's Krok 7).

Generates:
1. Perturbation sensitivity heatmap (Krok 5 - "najmocniejsza czesc")
2. Cross-dataset FMD grouped bar chart (Krok 4 - ranking)
3. Bootstrap stability with CI error bars (Krok 6)
4. Self-similarity sanity check (Krok 3)
5. Spearman agreement matrix (Krok 4)
6. Combined summary figure (all key results in one)
"""

import pathlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns

# Setup
plt.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 11,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'font.family': 'sans-serif',
})

DATA_DIR = pathlib.Path("results/reports/sensitivity_pivot")
PLOT_DIR = pathlib.Path("results/plots/sensitivity_pivot")
PLOT_DIR.mkdir(parents=True, exist_ok=True)

# Colors for configs
COLORS = {
    'CLaMP2-ABC': '#E63946',  # Red
    'CLaMP2-MTF': '#457B9D',  # Blue
    'CLaMP1-ABC': '#2A9D8F',  # Teal
}

CONFIG_ORDER = ['CLaMP2-ABC', 'CLaMP2-MTF', 'CLaMP1-ABC']


def plot_perturbation_heatmap():
    """Krok 5: Perturbation sensitivity heatmap - the KEY result."""
    df = pd.read_csv(DATA_DIR / "perturbation_sensitivity.csv")

    pivot = df.pivot_table(index="perturbation", columns="configuration", values="fmd_vs_original")
    pivot = pivot[CONFIG_ORDER]

    # Order perturbations logically
    pert_order = ['no_velocity', 'quantized_time', 'constant_tempo', 'all_combined']
    pert_labels = ['Remove Velocity\n(dynamics)', 'Quantize Time\n(16th grid)',
                   'Constant Tempo\n(120 BPM)', 'All Combined\n(de-expression)']
    pivot = pivot.reindex(pert_order)

    fig, ax = plt.subplots(figsize=(8, 5))

    # Custom colormap: white -> yellow -> orange -> red
    colors_cmap = ['#FFFFFF', '#FFF3CD', '#FFAD60', '#E63946', '#800020']
    cmap = LinearSegmentedColormap.from_list('sensitivity', colors_cmap, N=256)

    sns.heatmap(pivot, annot=True, fmt=".3f", cmap=cmap,
                ax=ax, linewidths=2, linecolor='white',
                vmin=0, vmax=0.55,
                annot_kws={'size': 14, 'weight': 'bold'},
                cbar_kws={'label': 'FMD (original vs perturbed)', 'shrink': 0.8})

    ax.set_yticklabels(pert_labels, rotation=0)
    ax.set_xticklabels(['CLaMP2 + REMI\n(ABC)', 'CLaMP2 + MIDI-Like\n(MTF)', 'CLaMP1 + REMI\n(ABC)'])
    ax.set_title('Perturbation Sensitivity Profile\n"What does each FMD configuration actually see?"',
                 fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('')
    ax.set_ylabel('')

    # Add annotation box
    textstr = ('Key insight: Only velocity removal is detected (FMD > 0.4),\n'
               'and only by CLaMP-2. CLaMP-1 is blind to all perturbations.')
    props = dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.8, edgecolor='orange')
    ax.text(0.5, -0.22, textstr, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', horizontalalignment='center', bbox=props)

    plt.tight_layout()
    fig.savefig(PLOT_DIR / "fig1_perturbation_heatmap.png")
    fig.savefig(PLOT_DIR / "fig1_perturbation_heatmap.pdf")
    plt.close(fig)
    print("  [1/6] fig1_perturbation_heatmap.png")


def plot_cross_dataset_ranking():
    """Krok 4: Cross-dataset FMD bar chart."""
    df = pd.read_csv(DATA_DIR / "cross_dataset_fmd.csv")

    # Rename pairs for clarity
    pair_labels = {
        'maestro_vs_pop909': 'MAESTRO\nvs POP909',
        'maestro_vs_folk': 'MAESTRO\nvs Folk',
        'pop909_vs_folk': 'POP909\nvs Folk',
    }
    df['pair_label'] = df['pair'].map(pair_labels)

    fig, ax = plt.subplots(figsize=(9, 5))

    x = np.arange(len(pair_labels))
    width = 0.25

    for i, cfg in enumerate(CONFIG_ORDER):
        subset = df[df['configuration'] == cfg].sort_values('pair')
        vals = subset['fmd'].values
        bars = ax.bar(x + i * width - width, vals, width,
                      label=cfg, color=COLORS[cfg], edgecolor='white', linewidth=0.5)
        # Add value labels on bars
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=8, fontweight='bold')

    ax.set_xlabel('')
    ax.set_ylabel('FMD (Frechet Music Distance)', fontweight='bold')
    ax.set_title('Cross-Dataset Stylistic Distance by Configuration\n(Krok 4: Do configurations agree on which datasets are most different?)',
                 fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([pair_labels[k] for k in sorted(pair_labels.keys())])
    ax.legend(title='Configuration', loc='upper left', framealpha=0.9)
    ax.set_ylim(0, 0.95)
    ax.axhline(y=0.07, color='gray', linestyle='--', alpha=0.5, label='Noise floor')
    ax.text(2.5, 0.075, 'noise floor (0.07)', fontsize=8, color='gray', ha='right')

    # Add ranking annotation
    ax.annotate('CLaMP2-ABC ranks\nPOP909-Folk as MOST different\n(others rank MAESTRO-Folk)',
                xy=(2.1, 0.814), xytext=(1.8, 0.85),
                fontsize=8, ha='center', color=COLORS['CLaMP2-ABC'],
                arrowprops=dict(arrowstyle='->', color=COLORS['CLaMP2-ABC']))

    plt.tight_layout()
    fig.savefig(PLOT_DIR / "fig2_cross_dataset_ranking.png")
    fig.savefig(PLOT_DIR / "fig2_cross_dataset_ranking.pdf")
    plt.close(fig)
    print("  [2/6] fig2_cross_dataset_ranking.png")


def plot_bootstrap_stability():
    """Krok 6: Bootstrap CI error bars."""
    df = pd.read_csv(DATA_DIR / "bootstrap_stability.csv")

    fig, ax = plt.subplots(figsize=(7, 4.5))

    x_pos = np.arange(len(CONFIG_ORDER))

    for i, cfg in enumerate(CONFIG_ORDER):
        row = df[df['configuration'] == cfg].iloc[0]
        mean = row['fmd_mean']
        ci_lo = row['fmd_ci_lower']
        ci_hi = row['fmd_ci_upper']
        std = row['fmd_std']

        ax.bar(i, mean, color=COLORS[cfg], edgecolor='black', linewidth=0.5, width=0.6, alpha=0.85)
        ax.errorbar(i, mean, yerr=[[mean - ci_lo], [ci_hi - mean]],
                    fmt='none', color='black', capsize=8, capthick=2, linewidth=2)

        # Add text annotation
        ax.text(i, ci_hi + 0.005, f'{mean:.3f} +/- {std:.3f}\nCV={row["cv"]*100:.1f}%',
                ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax.set_xticks(x_pos)
    ax.set_xticklabels(['CLaMP2 + REMI\n(ABC)', 'CLaMP2 + MIDI-Like\n(MTF)', 'CLaMP1 + REMI\n(ABC)'])
    ax.set_ylabel('FMD (MAESTRO vs POP909)', fontweight='bold')
    ax.set_title('Bootstrap Stability (10x resampling, N=200)\n(Krok 6: Which configuration is most stable?)',
                 fontsize=12, fontweight='bold')
    ax.set_ylim(0, 0.40)

    # Add note
    ax.text(0.5, -0.18, 'All configurations show comparable stability (CV ~ 8-9%). No significant differences.',
            transform=ax.transAxes, fontsize=9, ha='center', style='italic', color='gray')

    plt.tight_layout()
    fig.savefig(PLOT_DIR / "fig3_bootstrap_stability.png")
    fig.savefig(PLOT_DIR / "fig3_bootstrap_stability.pdf")
    plt.close(fig)
    print("  [3/6] fig3_bootstrap_stability.png")


def plot_self_similarity():
    """Krok 3: Self-similarity sanity check."""
    df = pd.read_csv(DATA_DIR / "self_similarity.csv")

    fig, ax = plt.subplots(figsize=(8, 4.5))

    datasets = ['maestro', 'pop909', 'folk']
    dataset_labels = ['MAESTRO', 'POP909', 'Folk']
    x = np.arange(len(datasets))
    width = 0.25

    for i, cfg in enumerate(CONFIG_ORDER):
        vals = [df[(df['dataset'] == ds) & (df['configuration'] == cfg)]['split_half_fmd'].values[0]
                for ds in datasets]
        bars = ax.bar(x + i * width - width, vals, width,
                      label=cfg, color=COLORS[cfg], edgecolor='white', linewidth=0.5)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.001,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=8)

    ax.axhline(y=0, color='black', linewidth=0.5)
    ax.axhline(y=0.07, color='red', linestyle='--', alpha=0.5, linewidth=1.5)
    ax.text(2.5, 0.072, 'max noise = 0.07', fontsize=8, color='red', ha='right')

    ax.set_xlabel('')
    ax.set_ylabel('Split-Half FMD', fontweight='bold')
    ax.set_title('Self-Similarity Sanity Check (should be ~ 0)\n(Krok 3: Are configurations internally stable?)',
                 fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(dataset_labels)
    ax.legend(title='Configuration')
    ax.set_ylim(0, 0.10)

    # Green checkmark annotation
    ax.text(0.5, -0.15, '✅ All values near zero (0.019-0.070) - all configurations are stable. Noise floor established.',
            transform=ax.transAxes, fontsize=9, ha='center', color='green')

    plt.tight_layout()
    fig.savefig(PLOT_DIR / "fig4_self_similarity.png")
    fig.savefig(PLOT_DIR / "fig4_self_similarity.pdf")
    plt.close(fig)
    print("  [4/6] fig4_self_similarity.png")


def plot_spearman_matrix():
    """Krok 4b: Spearman tau agreement matrix."""
    df = pd.read_csv(DATA_DIR / "spearman_ranking_agreement.csv")

    # Build 3x3 matrix
    matrix = np.eye(3)
    for _, row in df.iterrows():
        i = CONFIG_ORDER.index(row['config_a'])
        j = CONFIG_ORDER.index(row['config_b'])
        matrix[i, j] = row['spearman_tau']
        matrix[j, i] = row['spearman_tau']

    fig, ax = plt.subplots(figsize=(5.5, 4.5))

    # Custom diverging colormap
    cmap = sns.diverging_palette(10, 150, s=80, l=55, as_cmap=True)

    mask = np.zeros_like(matrix, dtype=bool)
    # Don't mask anything for a full matrix view

    sns.heatmap(matrix, annot=True, fmt=".2f", cmap=cmap,
                vmin=0, vmax=1, center=0.75,
                xticklabels=['CLaMP2\nABC', 'CLaMP2\nMTF', 'CLaMP1\nABC'],
                yticklabels=['CLaMP2\nABC', 'CLaMP2\nMTF', 'CLaMP1\nABC'],
                ax=ax, linewidths=2, linecolor='white',
                annot_kws={'size': 16, 'weight': 'bold'},
                cbar_kws={'label': 'Spearman tau', 'shrink': 0.8},
                square=True)

    ax.set_title('Ranking Agreement Between Configurations\n(Spearman tau - do they rank datasets the same way?)',
                 fontsize=11, fontweight='bold', pad=10)

    # Highlight the key finding
    ax.text(0.5, -0.15,
            'CLaMP2-MTF and CLaMP1-ABC agree perfectly (tau=1.0)\n'
            'CLaMP2-ABC disagrees with both (tau=0.5) - tokenization inverts ranking!',
            transform=ax.transAxes, fontsize=9, ha='center',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8, edgecolor='orange'))

    plt.tight_layout()
    fig.savefig(PLOT_DIR / "fig5_spearman_agreement.png")
    fig.savefig(PLOT_DIR / "fig5_spearman_agreement.pdf")
    plt.close(fig)
    print("  [5/6] fig5_spearman_agreement.png")


def plot_combined_summary():
    """Krok 7: Combined 4-panel summary figure for the paper."""
    fig = plt.figure(figsize=(14, 10))

    # Panel A: Perturbation heatmap (top-left, main result)
    ax1 = fig.add_subplot(2, 2, 1)
    df_pert = pd.read_csv(DATA_DIR / "perturbation_sensitivity.csv")
    pivot = df_pert.pivot_table(index="perturbation", columns="configuration", values="fmd_vs_original")
    pivot = pivot[CONFIG_ORDER]
    pert_order = ['no_velocity', 'quantized_time', 'constant_tempo', 'all_combined']
    pert_labels_short = ['No Velocity', 'Quantized Time', 'Constant Tempo', 'All Combined']
    pivot = pivot.reindex(pert_order)

    colors_cmap = ['#FFFFFF', '#FFF3CD', '#FFAD60', '#E63946', '#800020']
    cmap = LinearSegmentedColormap.from_list('s', colors_cmap, N=256)
    sns.heatmap(pivot, annot=True, fmt=".3f", cmap=cmap, ax=ax1,
                linewidths=1, linecolor='white', vmin=0, vmax=0.55,
                annot_kws={'size': 10, 'weight': 'bold'},
                cbar_kws={'shrink': 0.6})
    ax1.set_yticklabels(pert_labels_short, rotation=0, fontsize=9)
    ax1.set_xticklabels(['CLaMP2-ABC', 'CLaMP2-MTF', 'CLaMP1-ABC'], fontsize=9)
    ax1.set_title('(A) Perturbation Sensitivity', fontweight='bold', fontsize=11)
    ax1.set_xlabel('')
    ax1.set_ylabel('')

    # Panel B: Cross-dataset ranking (top-right)
    ax2 = fig.add_subplot(2, 2, 2)
    df_cross = pd.read_csv(DATA_DIR / "cross_dataset_fmd.csv")
    pair_order = ['maestro_vs_folk', 'maestro_vs_pop909', 'pop909_vs_folk']
    pair_short = ['MAESTRO\nvs Folk', 'MAESTRO\nvs POP909', 'POP909\nvs Folk']

    x = np.arange(3)
    width = 0.25
    for i, cfg in enumerate(CONFIG_ORDER):
        vals = [df_cross[(df_cross['pair'] == p) & (df_cross['configuration'] == cfg)]['fmd'].values[0]
                for p in pair_order]
        ax2.bar(x + i * width - width, vals, width, label=cfg, color=COLORS[cfg], edgecolor='white')

    ax2.set_xticks(x)
    ax2.set_xticklabels(pair_short, fontsize=9)
    ax2.set_ylabel('FMD')
    ax2.set_title('(B) Cross-Dataset Distance', fontweight='bold', fontsize=11)
    ax2.legend(fontsize=8, loc='upper right')
    ax2.axhline(0.07, color='gray', ls='--', alpha=0.4)

    # Panel C: Spearman matrix (bottom-left)
    ax3 = fig.add_subplot(2, 2, 3)
    df_sp = pd.read_csv(DATA_DIR / "spearman_ranking_agreement.csv")
    matrix = np.eye(3)
    for _, row in df_sp.iterrows():
        i = CONFIG_ORDER.index(row['config_a'])
        j = CONFIG_ORDER.index(row['config_b'])
        matrix[i, j] = row['spearman_tau']
        matrix[j, i] = row['spearman_tau']

    cmap_sp = sns.diverging_palette(10, 150, s=80, l=55, as_cmap=True)
    sns.heatmap(matrix, annot=True, fmt=".2f", cmap=cmap_sp, vmin=0, vmax=1, center=0.75,
                xticklabels=['CLaMP2-ABC', 'CLaMP2-MTF', 'CLaMP1-ABC'],
                yticklabels=['CLaMP2-ABC', 'CLaMP2-MTF', 'CLaMP1-ABC'],
                ax=ax3, linewidths=1, linecolor='white', square=True,
                annot_kws={'size': 12, 'weight': 'bold'}, cbar_kws={'shrink': 0.6})
    ax3.set_title('(C) Ranking Agreement (Spearman tau)', fontweight='bold', fontsize=11)

    # Panel D: Bootstrap stability (bottom-right)
    ax4 = fig.add_subplot(2, 2, 4)
    df_bs = pd.read_csv(DATA_DIR / "bootstrap_stability.csv")

    for i, cfg in enumerate(CONFIG_ORDER):
        row = df_bs[df_bs['configuration'] == cfg].iloc[0]
        ax4.bar(i, row['fmd_mean'], color=COLORS[cfg], edgecolor='black', linewidth=0.5, width=0.6, alpha=0.85)
        ax4.errorbar(i, row['fmd_mean'],
                     yerr=[[row['fmd_mean'] - row['fmd_ci_lower']], [row['fmd_ci_upper'] - row['fmd_mean']]],
                     fmt='none', color='black', capsize=6, capthick=1.5, linewidth=1.5)
        ax4.text(i, row['fmd_ci_upper'] + 0.005, f"CV={row['cv']*100:.1f}%",
                 ha='center', fontsize=9, fontweight='bold')

    ax4.set_xticks(range(3))
    ax4.set_xticklabels(['CLaMP2-ABC', 'CLaMP2-MTF', 'CLaMP1-ABC'], fontsize=9)
    ax4.set_ylabel('FMD (MAESTRO vs POP909)')
    ax4.set_title('(D) Bootstrap Stability (95% CI)', fontweight='bold', fontsize=11)
    ax4.set_ylim(0, 0.38)

    plt.suptitle('FMD Sensitivity Profiling - Summary of Key Results',
                 fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(PLOT_DIR / "fig6_combined_summary_4panel.png")
    fig.savefig(PLOT_DIR / "fig6_combined_summary_4panel.pdf")
    plt.close(fig)
    print("  [6/6] fig6_combined_summary_4panel.png")


def main():
    print("Generating publication-quality plots for sensitivity pivot...")
    print(f"  Input:  {DATA_DIR}")
    print(f"  Output: {PLOT_DIR}")
    print()

    plot_perturbation_heatmap()
    plot_cross_dataset_ranking()
    plot_bootstrap_stability()
    plot_self_similarity()
    plot_spearman_matrix()
    plot_combined_summary()

    print()
    print("All 6 figures generated (PNG + PDF).")
    print("Key figures for the paper:")
    print("  - fig1_perturbation_heatmap (Krok 5 - main result)")
    print("  - fig2_cross_dataset_ranking (Krok 4)")
    print("  - fig5_spearman_agreement (Krok 4 - ranking agreement)")
    print("  - fig6_combined_summary_4panel (Krok 7 - all in one)")


if __name__ == "__main__":
    main()

