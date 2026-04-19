# Cross-Dataset Validation Report

**Generalizability analysis of FMD sensitivity findings.**

## Design
- **Sources:** lakh_cd2, midicaps
- **Total FMD observations:** 3840
  - lakh_cd2: 1920 rows, 6 pairs
  - midicaps: 1920 rows, 6 pairs

## η² Variance Decomposition by Source

| Factor | lakh_cd2 | midicaps |
|--------| --- | --- |
| tokenizer | 0.1009 | 0.1245 |
| model | 0.0865 | 0.1363 |
| preprocess | 0.0203 | 0.0003 |

## Tokenizer × Model Cell Means by Source

### lakh_cd2

| Tokenizer | Model | Mean FMD | Std | N |
|-----------|-------|----------|-----|---|
| REMI | CLaMP-2 | 0.0749 | 0.0165 | 240 |
| TSD | CLaMP-2 | 0.1339 | 0.0228 | 240 |
| Octuple | CLaMP-1 | 0.1824 | 0.0548 | 240 |
| MIDI-Like | CLaMP-1 | 0.1995 | 0.0583 | 240 |
| MIDI-Like | CLaMP-2 | 0.2232 | 0.0607 | 240 |
| REMI | CLaMP-1 | 0.2378 | 0.0668 | 240 |
| Octuple | CLaMP-2 | 0.2739 | 0.0220 | 240 |
| TSD | CLaMP-1 | 0.2862 | 0.0830 | 240 |

### midicaps

| Tokenizer | Model | Mean FMD | Std | N |
|-----------|-------|----------|-----|---|
| REMI | CLaMP-2 | 0.0896 | 0.0211 | 240 |
| TSD | CLaMP-2 | 0.2239 | 0.0917 | 240 |
| Octuple | CLaMP-1 | 0.2602 | 0.0668 | 240 |
| MIDI-Like | CLaMP-2 | 0.2895 | 0.0609 | 240 |
| MIDI-Like | CLaMP-1 | 0.3232 | 0.1132 | 240 |
| REMI | CLaMP-1 | 0.3385 | 0.0858 | 240 |
| Octuple | CLaMP-2 | 0.3720 | 0.0724 | 240 |
| TSD | CLaMP-1 | 0.4261 | 0.1061 | 240 |

## Pipeline Ranking Agreement

| Comparison | Spearman ρ | p-value | N common |
|------------|-----------|---------|----------|
| lakh_cd2 vs midicaps | 0.9208 | 8.4067e-14 | 32 |

## Interpretation

### Generalizability Assessment

✅ **Strong generalizability** (avg ρ = 0.921): Pipeline rankings are highly consistent across data sources. The sensitivity findings from Lakh CD2 replicate well.

### η² Consistency

- **tokenizer**: range [0.1009, 0.1245], CV = 0.10
- **model**: range [0.0865, 0.1363], CV = 0.22
- **preprocess**: range [0.0003, 0.0203], CV = 0.97

## Conclusions

1. Cross-dataset validation tested 2 independent data sources.
2. Total observations: 3840.
3. See plots in `results/plots/paper/cross_*.png` for visual comparison.
