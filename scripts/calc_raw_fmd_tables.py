"""Calculate raw FMD eta_sq values for tables."""
import pandas as pd
import numpy as np

# Load raw FMD data
fmd_data = pd.read_csv('results/reports/lakh_multi/multi_genre_fmd.csv')

print("RAW FMD TOKENIZER SENSITIVITY PER MODEL")
print("=" * 70)
print()

# Calculate eta_sq for tokenizer effect per model
for model in sorted(fmd_data['model'].unique()):
    model_data = fmd_data[fmd_data['model'] == model]
    
    # Tokenizer effect
    grand_mean = model_data['fmd'].mean()
    ss_between_tok = model_data.groupby('tokenizer').apply(
        lambda x: len(x) * (x['fmd'].mean() - grand_mean) ** 2
    ).sum()
    ss_total = ((model_data['fmd'] - grand_mean) ** 2).sum()
    eta_sq_tok = ss_between_tok / ss_total if ss_total > 0 else 0
    
    # Preprocessing effect
    ss_between_pre = model_data.groupby(['remove_velocity', 'hard_quantization']).apply(
        lambda x: len(x) * (x['fmd'].mean() - grand_mean) ** 2
    ).sum()
    eta_sq_pre = ss_between_pre / ss_total if ss_total > 0 else 0
    
    print(f"{model:20} | η²(tok)={eta_sq_tok:.4f} | η²(pre)={eta_sq_pre:.4f}")

print()
print("TABLE FORMAT:")
print()
print("| Model | η²(tokenizer) | η²(preprocess) | Interpretacja |")
print("|-------|--------:|--------:|---------------|")

# Generate table rows
for model in sorted(fmd_data['model'].unique()):
    model_data = fmd_data[fmd_data['model'] == model]
    
    # Tokenizer effect
    grand_mean = model_data['fmd'].mean()
    ss_between_tok = model_data.groupby('tokenizer').apply(
        lambda x: len(x) * (x['fmd'].mean() - grand_mean) ** 2
    ).sum()
    ss_total = ((model_data['fmd'] - grand_mean) ** 2).sum()
    eta_sq_tok = ss_between_tok / ss_total if ss_total > 0 else 0
    
    # Preprocessing effect
    ss_between_pre = model_data.groupby(['remove_velocity', 'hard_quantization']).apply(
        lambda x: len(x) * (x['fmd'].mean() - grand_mean) ** 2
    ).sum()
    eta_sq_pre = ss_between_pre / ss_total if ss_total > 0 else 0
    
    # Interpretation
    if eta_sq_tok > 0.15:
        interp = "🔴 Bardzo wrażliwy na tokenizer"
    elif eta_sq_tok > 0.08:
        interp = "🟡 Umiarkowanie wrażliwy"
    elif eta_sq_tok > 0.05:
        interp = "🟡 Mało wrażliwy"
    else:
        interp = "🟢 Niska wrażliwość"
    
    print(f"| {model:20} | {eta_sq_tok:.4f} | {eta_sq_pre:.4f} | {interp} |")
