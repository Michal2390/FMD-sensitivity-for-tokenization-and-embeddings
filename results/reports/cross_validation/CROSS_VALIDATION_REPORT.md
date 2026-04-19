# Cross-Dataset Validation Report

**Generalizability analysis of FMD sensitivity findings.**

## Design
- **Sources:** lakh_cd2, midicaps
- **Total FMD observations:** 5760
  - lakh_cd2: 2880 rows, 6 pairs
  - midicaps: 2880 rows, 6 pairs

## η² Variance Decomposition by Source

| Factor | lakh_cd2 | midicaps |
|--------| --- | --- |
| tokenizer | 0.0196 | 0.0393 |
| model | 0.7782 | 0.8046 |
| preprocess | 0.0022 | 0.0040 |

## Bootstrap 95% CI for η² by Source

### lakh_cd2

| Factor | η² | 95% CI lower | 95% CI upper |
|--------|----|-------------|-------------|
| tokenizer | 0.0196 | 0.0133 | 0.0290 |
| model | 0.7782 | 0.7637 | 0.7927 |
| preprocess | 0.0022 | 0.0005 | 0.0078 |

### midicaps

| Factor | η² | 95% CI lower | 95% CI upper |
|--------|----|-------------|-------------|
| tokenizer | 0.0393 | 0.0277 | 0.0540 |
| model | 0.8046 | 0.7933 | 0.8165 |
| preprocess | 0.0040 | 0.0012 | 0.0106 |

## Tokenizer × Model Cell Means by Source

### lakh_cd2

| Tokenizer | Model | Mean FMD | Std | N |
|-----------|-------|----------|-----|---|
| REMI | CLaMP-2 | 0.0749 | 0.0165 | 240 |
| TSD | CLaMP-2 | 0.1329 | 0.0222 | 240 |
| Octuple | CLaMP-1 | 0.1822 | 0.0574 | 240 |
| MIDI-Like | CLaMP-1 | 0.2011 | 0.0586 | 240 |
| MIDI-Like | CLaMP-2 | 0.2189 | 0.0571 | 240 |
| REMI | CLaMP-1 | 0.2378 | 0.0668 | 240 |
| Octuple | CLaMP-2 | 0.2747 | 0.0241 | 240 |
| TSD | CLaMP-1 | 0.2869 | 0.0858 | 240 |
| Octuple | MusicBERT | 1.4811 | 0.3647 | 240 |
| MIDI-Like | MusicBERT | 2.2404 | 1.1950 | 240 |
| TSD | MusicBERT | 2.7099 | 0.6150 | 240 |
| REMI | MusicBERT | 2.7191 | 0.5561 | 240 |

### midicaps

| Tokenizer | Model | Mean FMD | Std | N |
|-----------|-------|----------|-----|---|
| REMI | CLaMP-2 | 0.0960 | 0.0201 | 240 |
| TSD | CLaMP-2 | 0.2270 | 0.0812 | 240 |
| Octuple | CLaMP-1 | 0.3009 | 0.0808 | 240 |
| MIDI-Like | CLaMP-2 | 0.3067 | 0.0703 | 240 |
| MIDI-Like | CLaMP-1 | 0.3277 | 0.1078 | 240 |
| REMI | CLaMP-1 | 0.3694 | 0.0899 | 240 |
| Octuple | CLaMP-2 | 0.3963 | 0.0983 | 240 |
| TSD | CLaMP-1 | 0.4449 | 0.0779 | 240 |
| Octuple | MusicBERT | 1.8928 | 0.3702 | 240 |
| MIDI-Like | MusicBERT | 2.1723 | 0.5753 | 240 |
| REMI | MusicBERT | 3.4532 | 0.6268 | 240 |
| TSD | MusicBERT | 3.8445 | 0.6782 | 240 |

## Pipeline Ranking Agreement

| Comparison | Spearman ρ | p-value | N common |
|------------|-----------|---------|----------|
| lakh_cd2 vs midicaps | 0.9752 | 8.5170e-32 | 48 |

## Interpretation

### Generalizability Assessment

✅ **Strong generalizability** (avg ρ = 0.975): Pipeline rankings are highly consistent across data sources. The sensitivity findings from Lakh CD2 replicate well.

### η² Consistency

- **tokenizer**: range [0.0196, 0.0393], CV = 0.34
- **model**: range [0.7782, 0.8046], CV = 0.02
- **preprocess**: range [0.0022, 0.0040], CV = 0.29

## Conclusions

1. Cross-dataset validation tested 2 independent data sources.
2. Total observations: 5760.
3. See plots in `results/plots/paper/cross_*.png` for visual comparison.
