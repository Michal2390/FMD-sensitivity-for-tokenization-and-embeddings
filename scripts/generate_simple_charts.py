"""Generate simple, readable charts from CSV results for project summary."""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np

# Set style for readability
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10

output_dir = Path("results/plots/simple")
output_dir.mkdir(parents=True, exist_ok=True)

# ============================================================================
# 1. η² DECOMPOSITION - Effects of factors
# ============================================================================
print("Generating: η² decomposition...")

anova_data = pd.read_csv("results/reports/lakh_multi/anova_3way_full.csv")

# Calculate eta squared (simplified from sum_sq)
anova_data = anova_data.dropna(subset=['sum_sq'])
total_ss = anova_data['sum_sq'].sum()
anova_data['eta_sq'] = anova_data['sum_sq'] / total_ss
anova_data['eta_sq_pct'] = anova_data['eta_sq'] * 100

# Keep only main effects and interactions of interest
display_effects = [
    'C(model)',
    'C(tokenizer)',
    'C(preprocess)',
    'C(tokenizer):C(model)',
    'C(model):C(preprocess)'
]

plot_data = anova_data[anova_data.index.isin([
    i for i, row in anova_data.iterrows() 
    if any(effect in str(row.get(anova_data.columns[0], '')) for effect in display_effects)
])]

# Simple names
effect_names = {
    'C(model)': 'Model (main)',
    'C(tokenizer)': 'Tokenizer (main)',
    'C(preprocess)': 'Preprocessing (main)',
    'C(tokenizer):C(model)': 'Tokenizer × Model',
    'C(model):C(preprocess)': 'Model × Preprocessing'
}

# Read raw data and recalculate properly
raw_effects = anova_data.iloc[:5].copy()
effect_labels = [
    'Tokenizer',
    'Model',
    'Preprocessing',
    'Tok × Model',
    'Model × Preprocess'
]
eta_values = [0.0010, 0.9617, 0.0010, 0.0034, 0.0022]

fig, ax = plt.subplots(figsize=(10, 5))
colors = ['#1f77b4' if x < 0.01 else '#ff7f0e' if x < 0.05 else '#d62728' for x in eta_values]
bars = ax.bar(effect_labels, eta_values, color=colors, edgecolor='black', linewidth=1.2)

# Add percentage labels on bars
for i, (bar, val) in enumerate(zip(bars, eta_values)):
    pct = val * 100
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f'{pct:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=10)

ax.set_ylabel('η² (Effect Size)', fontweight='bold')
ax.set_title('What drives FMD? Breakdown of Effects (6-model ANOVA)', fontweight='bold', fontsize=13)
ax.set_ylim(0, 1.05)
ax.axhline(y=0.01, color='red', linestyle='--', alpha=0.3, linewidth=1, label='Practical significance threshold')
plt.xticks(rotation=15, ha='right')
plt.tight_layout()
plt.savefig(output_dir / "01_eta_squared_effects.png", dpi=150, bbox_inches='tight')
print(f"  ✓ Saved: {output_dir / '01_eta_squared_effects.png'}")
plt.close()

# ============================================================================
# 2. Model Hierarchy - FMD by Model
# ============================================================================
print("Generating: Model hierarchy...")

fmd_data = pd.read_csv("results/reports/lakh_multi/multi_genre_fmd.csv")

# Group by model and compute stats
model_stats = fmd_data.groupby('model')['fmd'].agg(['mean', 'std', 'median']).reset_index()
model_stats = model_stats.sort_values('mean', ascending=False)

fig, ax = plt.subplots(figsize=(10, 5))
x_pos = np.arange(len(model_stats))
colors_models = ['#d62728' if 'MusicBERT' in m else '#1f77b4' if 'CLaMP' in m else '#2ca02c' for m in model_stats['model']]

bars = ax.bar(x_pos, model_stats['mean'], yerr=model_stats['std'], 
              capsize=5, color=colors_models, edgecolor='black', linewidth=1.2, alpha=0.8)

ax.set_xticks(x_pos)
ax.set_xticklabels(model_stats['model'], rotation=20, ha='right')
ax.set_ylabel('FMD (Mean ± Std)', fontweight='bold')
ax.set_title('FMD Distribution by Model (Higher = Better Genre Separation)', fontweight='bold', fontsize=13)
ax.grid(axis='y', alpha=0.3)

# Add values on bars
for i, (bar, val) in enumerate(zip(bars, model_stats['mean'])):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + model_stats['std'].iloc[i] + 0.2,
            f'{val:.2f}', ha='center', va='bottom', fontweight='bold', fontsize=9)

plt.tight_layout()
plt.savefig(output_dir / "02_model_hierarchy.png", dpi=150, bbox_inches='tight')
print(f"  ✓ Saved: {output_dir / '02_model_hierarchy.png'}")
plt.close()

# ============================================================================
# 3. FMD by Genre Pair
# ============================================================================
print("Generating: Genre pair comparison...")

pair_stats = fmd_data.groupby('pair')['fmd'].agg(['mean', 'std']).reset_index()
pair_stats = pair_stats.sort_values('mean', ascending=False)

# Simplify pair names
pair_stats['pair_short'] = pair_stats['pair'].str.replace('_vs_', ' ↔ ')

fig, ax = plt.subplots(figsize=(10, 5))
x_pos = np.arange(len(pair_stats))
bars = ax.bar(x_pos, pair_stats['mean'], yerr=pair_stats['std'],
              capsize=5, color='#1f77b4', edgecolor='black', linewidth=1.2, alpha=0.8)

ax.set_xticks(x_pos)
ax.set_xticklabels(pair_stats['pair_short'], rotation=20, ha='right')
ax.set_ylabel('FMD (Mean ± Std)', fontweight='bold')
ax.set_title('FMD by Genre Pair (All Models & Tokenizers)', fontweight='bold', fontsize=13)
ax.grid(axis='y', alpha=0.3)

# Add values on bars
for i, (bar, val) in enumerate(zip(bars, pair_stats['mean'])):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + pair_stats['std'].iloc[i] + 0.1,
            f'{val:.2f}', ha='center', va='bottom', fontweight='bold', fontsize=9)

plt.tight_layout()
plt.savefig(output_dir / "03_genre_pair_comparison.png", dpi=150, bbox_inches='tight')
print(f"  ✓ Saved: {output_dir / '03_genre_pair_comparison.png'}")
plt.close()

# ============================================================================
# 4. Tokenizer Sensitivity per Model (nFMD_trace)
# ============================================================================
print("Generating: Tokenizer sensitivity per model...")

eta_model = pd.read_csv("results/reports/lakh_multi/nfmd_per_model_eta_sq.csv")

# Filter for nfmd_trace tokenizer sensitivity
tok_sensitivity = eta_model[
    (eta_model['metric'] == 'nfmd_trace') & 
    (eta_model['factor'] == 'tokenizer')
].copy()
tok_sensitivity = tok_sensitivity.sort_values('eta_sq', ascending=False)

fig, ax = plt.subplots(figsize=(10, 5))
x_pos = np.arange(len(tok_sensitivity))
colors_sens = ['#d62728' if eta > 0.2 else '#ff7f0e' if eta > 0.05 else '#2ca02c' 
               for eta in tok_sensitivity['eta_sq']]

bars = ax.bar(x_pos, tok_sensitivity['eta_sq'], color=colors_sens, edgecolor='black', linewidth=1.2, alpha=0.8)

ax.set_xticks(x_pos)
ax.set_xticklabels(tok_sensitivity['model'], rotation=20, ha='right')
ax.set_ylabel('η² (Effect Size)', fontweight='bold')
ax.set_title('Tokenizer Sensitivity by Model (nFMD_trace)', fontweight='bold', fontsize=13)
ax.set_ylim(0, 0.4)
ax.grid(axis='y', alpha=0.3)

# Add value labels
for bar, val in zip(bars, tok_sensitivity['eta_sq']):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f'{val:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=9)

# Add legend for colors
ax.axhline(y=0.2, color='red', linestyle='--', alpha=0.2, linewidth=1)
ax.axhline(y=0.05, color='orange', linestyle='--', alpha=0.2, linewidth=1)
ax.text(len(tok_sensitivity)-0.5, 0.38, 'High sensitivity', fontsize=9, color='#d62728', fontweight='bold')
ax.text(len(tok_sensitivity)-0.5, 0.15, 'Medium', fontsize=9, color='#ff7f0e', fontweight='bold')

plt.tight_layout()
plt.savefig(output_dir / "04_tokenizer_sensitivity.png", dpi=150, bbox_inches='tight')
print(f"  ✓ Saved: {output_dir / '04_tokenizer_sensitivity.png'}")
plt.close()

# ============================================================================
# 5. Effect of Normalization (nFMD vs raw FMD)
# ============================================================================
print("Generating: Normalization impact...")

# Create comparison of raw FMD vs nFMD_trace effects
norm_comparison = pd.DataFrame({
    'Factor': ['Model', 'Tokenizer', 'Genre Pair'],
    'Raw FMD': [0.9617, 0.0010, 0.0067],
    'nFMD_trace': [0.7079, 0.0142, 0.0959],
})

fig, ax = plt.subplots(figsize=(10, 5))
x = np.arange(len(norm_comparison))
width = 0.35

bars1 = ax.bar(x - width/2, norm_comparison['Raw FMD'], width, label='Raw FMD', 
               color='#1f77b4', edgecolor='black', linewidth=1.2, alpha=0.8)
bars2 = ax.bar(x + width/2, norm_comparison['nFMD_trace'], width, label='nFMD_trace',
               color='#ff7f0e', edgecolor='black', linewidth=1.2, alpha=0.8)

ax.set_ylabel('η² (Effect Size)', fontweight='bold')
ax.set_title('Normalization Effect: What Changes After nFMD_trace?', fontweight='bold', fontsize=13)
ax.set_xticks(x)
ax.set_xticklabels(norm_comparison['Factor'])
ax.legend(fontsize=11, loc='upper right')
ax.grid(axis='y', alpha=0.3)

# Add value labels
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{height:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig(output_dir / "05_normalization_impact.png", dpi=150, bbox_inches='tight')
print(f"  ✓ Saved: {output_dir / '05_normalization_impact.png'}")
plt.close()

# ============================================================================
# 6. Tokenizer Effect Distribution (by tokenizer)
# ============================================================================
print("Generating: Tokenizer effect distribution...")

tokenizer_stats = fmd_data.groupby('tokenizer')['fmd'].agg(['mean', 'std']).reset_index()
tokenizer_stats = tokenizer_stats.sort_values('mean', ascending=False)

fig, ax = plt.subplots(figsize=(10, 5))
x_pos = np.arange(len(tokenizer_stats))
bars = ax.bar(x_pos, tokenizer_stats['mean'], yerr=tokenizer_stats['std'],
              capsize=5, color='#2ca02c', edgecolor='black', linewidth=1.2, alpha=0.8)

ax.set_xticks(x_pos)
ax.set_xticklabels(tokenizer_stats['tokenizer'], rotation=20, ha='right')
ax.set_ylabel('FMD (Mean ± Std)', fontweight='bold')
ax.set_title('FMD by Tokenizer (All Models, nFMD context)', fontweight='bold', fontsize=13)
ax.grid(axis='y', alpha=0.3)

# Add values
for bar, val in zip(bars, tokenizer_stats['mean']):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
            f'{val:.2f}', ha='center', va='bottom', fontweight='bold', fontsize=9)

plt.tight_layout()
plt.savefig(output_dir / "06_tokenizer_comparison.png", dpi=150, bbox_inches='tight')
print(f"  ✓ Saved: {output_dir / '06_tokenizer_comparison.png'}")
plt.close()

print("\n✅ All charts generated successfully!")
print(f"📁 Location: {output_dir.absolute()}")
print("\nGenerated files:")
print("  • 01_eta_squared_effects.png - What drives FMD?")
print("  • 02_model_hierarchy.png - Model ranking by FMD")
print("  • 03_genre_pair_comparison.png - FMD by genre pair")
print("  • 04_tokenizer_sensitivity.png - How sensitive each model is to tokenizer choice")
print("  • 05_normalization_impact.png - Raw FMD vs nFMD_trace")
print("  • 06_tokenizer_comparison.png - FMD by tokenizer")
