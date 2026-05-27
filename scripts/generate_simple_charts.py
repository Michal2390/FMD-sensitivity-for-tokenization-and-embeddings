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
# 1. η² DECOMPOSITION - Effects of factors (from raw FMD ANOVA)
# ============================================================================
print("Generating: η² decomposition (raw FMD)...")

# Raw effects from the raw FMD ANOVA
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
ax.set_title('What drives FMD? Breakdown of Effects (Raw FMD, 6-model ANOVA)', fontweight='bold', fontsize=13)
ax.set_ylim(0, 1.05)
ax.axhline(y=0.01, color='red', linestyle='--', alpha=0.3, linewidth=1)
plt.xticks(rotation=15, ha='right')
plt.tight_layout()
plt.savefig(output_dir / "01_eta_squared_effects.png", dpi=150, bbox_inches='tight')
print(f"  ✓ Saved: {output_dir / '01_eta_squared_effects.png'}")
plt.close()

# ============================================================================
# 2. Model Hierarchy - FMD by Model (raw FMD)
# ============================================================================
print("Generating: Model hierarchy (raw FMD)...")

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
ax.set_title('FMD Distribution by Model (Raw FMD, Higher = Better Genre Separation)', fontweight='bold', fontsize=13)
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
# 3. FMD by Genre Pair (raw FMD)
# ============================================================================
print("Generating: Genre pair comparison (raw FMD)...")

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
ax.set_title('FMD by Genre Pair (Raw FMD, All Models & Tokenizers)', fontweight='bold', fontsize=13)
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
# 4. Tokenizer Sensitivity per Model (raw FMD, calculating eta_sq from data)
# ============================================================================
print("Generating: Tokenizer sensitivity per model (raw FMD)...")

# Calculate eta_sq for tokenizer effect per model using raw FMD data
fmd_data_copy = fmd_data.copy()

tokenizer_sensitivities = []
for model in sorted(fmd_data_copy['model'].unique()):
    model_data = fmd_data_copy[fmd_data_copy['model'] == model]
    
    # Calculate within-group and between-group variance for tokenizer
    grand_mean = model_data['fmd'].mean()
    ss_between = model_data.groupby('tokenizer').apply(
        lambda x: len(x) * (x['fmd'].mean() - grand_mean) ** 2
    ).sum()
    ss_total = ((model_data['fmd'] - grand_mean) ** 2).sum()
    
    if ss_total > 0:
        eta_sq = ss_between / ss_total
    else:
        eta_sq = 0
    
    tokenizer_sensitivities.append({
        'model': model,
        'eta_sq': eta_sq
    })

tok_sens_df = pd.DataFrame(tokenizer_sensitivities)
tok_sens_df = tok_sens_df.sort_values('eta_sq', ascending=False)

fig, ax = plt.subplots(figsize=(10, 5))
x_pos = np.arange(len(tok_sens_df))
colors_sens = ['#d62728' if eta > 0.1 else '#ff7f0e' if eta > 0.02 else '#2ca02c' 
               for eta in tok_sens_df['eta_sq']]

bars = ax.bar(x_pos, tok_sens_df['eta_sq'], color=colors_sens, edgecolor='black', linewidth=1.2, alpha=0.8)

ax.set_xticks(x_pos)
ax.set_xticklabels(tok_sens_df['model'], rotation=20, ha='right')
ax.set_ylabel('η² (Effect Size)', fontweight='bold')
ax.set_title('Tokenizer Sensitivity by Model (Raw FMD)', fontweight='bold', fontsize=13)
ax.set_ylim(0, max(tok_sens_df['eta_sq']) * 1.2)
ax.grid(axis='y', alpha=0.3)

# Add value labels
for bar, val in zip(bars, tok_sens_df['eta_sq']):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
            f'{val:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=9)

plt.tight_layout()
plt.savefig(output_dir / "04_tokenizer_sensitivity.png", dpi=150, bbox_inches='tight')
print(f"  ✓ Saved: {output_dir / '04_tokenizer_sensitivity.png'}")
plt.close()

# ============================================================================
# 5. Effect of Normalization (raw FMD vs nFMD_trace) — THIS USES BOTH
# ============================================================================
print("Generating: Normalization impact (raw FMD vs nFMD_trace comparison)...")

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
ax.set_title('Normalization Effect: How nFMD_trace Changes the Picture', fontweight='bold', fontsize=13)
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
# 6. Tokenizer Effect Distribution (raw FMD)
# ============================================================================
print("Generating: Tokenizer effect distribution (raw FMD)...")

tokenizer_stats = fmd_data.groupby('tokenizer')['fmd'].agg(['mean', 'std']).reset_index()
tokenizer_stats = tokenizer_stats.sort_values('mean', ascending=False)

fig, ax = plt.subplots(figsize=(10, 5))
x_pos = np.arange(len(tokenizer_stats))
bars = ax.bar(x_pos, tokenizer_stats['mean'], yerr=tokenizer_stats['std'],
              capsize=5, color='#2ca02c', edgecolor='black', linewidth=1.2, alpha=0.8)

ax.set_xticks(x_pos)
ax.set_xticklabels(tokenizer_stats['tokenizer'], rotation=20, ha='right')
ax.set_ylabel('FMD (Mean ± Std)', fontweight='bold')
ax.set_title('FMD by Tokenizer (Raw FMD, All Models)', fontweight='bold', fontsize=13)
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
print("  • 01_eta_squared_effects.png - What drives FMD? (raw FMD)")
print("  • 02_model_hierarchy.png - Model ranking by FMD (raw FMD)")
print("  • 03_genre_pair_comparison.png - FMD by genre pair (raw FMD)")
print("  • 04_tokenizer_sensitivity.png - How sensitive each model is to tokenizer choice (raw FMD)")
print("  • 05_normalization_impact.png - Raw FMD vs nFMD_trace COMPARISON")
print("  • 06_tokenizer_comparison.png - FMD by tokenizer (raw FMD)")
print("\nNote: Charts 1-4, 6 use RAW FMD. Chart 5 compares raw FMD vs nFMD_trace.")
