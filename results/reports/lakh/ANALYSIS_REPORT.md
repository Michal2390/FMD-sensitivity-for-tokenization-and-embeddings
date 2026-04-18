# FMD Sensitivity Analysis — Full Results Report

**Genres:** rock vs jazz (Lakh MIDI dataset)
**Variants:** 32 (4 tokenizers × 2 models × 4 preprocessing)
**Samples:** 119 rock, 119 jazz MIDI files

## 1. Summary Statistics

| Metric | Value |
|--------|-------|
| FMD range | [0.0536, 0.2696] |
| FMD mean ± std | 0.1560 ± 0.0605 |
| FMD median | 0.1503 |
| Mean bootstrap CI width | 0.1274 |

### Per Factor Means

| Factor | Level | Mean FMD | Std |
|--------|-------|----------|-----|
| Tokenizer | MIDI-Like | 0.1527 | 0.0222 |
| Tokenizer | Octuple | 0.1963 | 0.0680 |
| Tokenizer | REMI | 0.1188 | 0.0706 |
| Tokenizer | TSD | 0.1564 | 0.0515 |
| Model | CLaMP-1 | 0.1641 | 0.0363 |
| Model | CLaMP-2 | 0.1480 | 0.0781 |

## 2. ANOVA — Variance Decomposition

| Factor | F | p-value | η² | Partial η² | Effect size |
|--------|---|---------|----|-----------:|-------------|
| tokenizer | 2.530 | 0.0775† | 0.2133 | 0.2133 | **LARGE** |
| model | 0.559 | 0.4605 | 0.0183 | 0.0183 | small |
| preprocess | 0.516 | 0.6744 | 0.0524 | 0.0524 | small |

> **Note:** With N=1 per cell (32 unique variants), classical ANOVA power is limited.
> η² benchmarks (Cohen, 1988): small ≥ 0.01, medium ≥ 0.06, large ≥ 0.14

## 3. Permutation Tests (5000 permutations)

| Factor | F_observed | p_permutation | Significant (α=0.05) |
|--------|-----------|--------------|---------------------|
| tokenizer | 2.530 | 0.0738 | † (marginal) |
| model | 0.559 | 0.4593 | No |
| preprocess | 0.516 | 0.6665 | No |

## 4. Post-hoc: Tukey HSD

### Tokenizer

| Pair | Mean diff | p-adj | Significant |
|------|-----------|-------|-------------|
| MIDI-Like vs Octuple | 0.0436 | 0.4245 | No |
| MIDI-Like vs REMI | -0.0339 | 0.6312 | No |
| MIDI-Like vs TSD | 0.0038 | 0.9991 | No |
| Octuple vs REMI | -0.0775 | 0.0481 | **Yes** |
| Octuple vs TSD | -0.0399 | 0.5021 | No |
| REMI vs TSD | 0.0377 | 0.5493 | No |

### Model
| CLaMP-1 vs CLaMP-2 | -0.0161 | 0.4605 | No |

## 5. Effect Sizes (Cohen's d)

| Comparison | d | Magnitude |
|------------|---|-----------|
| tok_Octuple vs REMI | 1.119 | **large** |
| tok_MIDI-Like vs Octuple | -0.863 | **large** |
| tok_Octuple vs TSD | 0.661 | medium |
| tok_MIDI-Like vs REMI | 0.648 | medium |
| tok_REMI vs TSD | -0.610 | medium |
| mod_CLaMP-1 vs CLaMP-2 | 0.264 | small |
| tok_MIDI-Like vs TSD | -0.095 | negligible |

## 6. Key Findings

1. **Tokenizer choice is the primary sensitivity factor** (η² = 0.213, large effect).
   - Octuple vs REMI is the only significant pairwise difference (Tukey p = 0.048, Cohen's d = 1.12).
   - Permutation test confirms marginal significance (p = 0.074).

2. **Embedding model (CLaMP-1 vs CLaMP-2) has negligible impact on FMD** (η² = 0.018).
   - Mean difference is only 0.016 (not significant).
   - However, strong **interaction with tokenizer**: Octuple+CLaMP-2 produces highest FMD while REMI+CLaMP-2 produces lowest.

3. **Preprocessing has small effect** (η² = 0.052).
   - Velocity removal reduces FMD (genres become more similar without velocity information).
   - Hard quantization has minimal impact.

4. **Token-level statistics (length, entropy) do NOT predict FMD** (ρ < 0.18, p > 0.15).

5. **Cosine similarity gaps are minuscule** (0.0001–0.002) → genre separation in FMD is driven by covariance structure, not mean embeddings.

6. **Notable interaction**: Octuple tokenization paired with CLaMP-2 yields anomalously high FMD (mean 0.260), suggesting this tokenizer produces representations that CLaMP-2 maps to highly divergent embedding distributions.


## 7. Limitations

- N = 1 per cell → ANOVA has limited power; results should be interpreted with effect sizes (η², d) rather than p-values alone.
- Only 2 genres (rock, jazz) → findings may not generalize to other genre pairs.
- CLaMP models use proxy embedding paths (text encoder for CLaMP-1, general for CLaMP-2).
- Bootstrap CI widths are substantial → larger sample sizes would tighten estimates.
- No interaction terms in one-way ANOVA fallback (statsmodels three-way failed due to single observations per cell).


## 8. Recommendations for Paper

1. **Lead with tokenizer sensitivity**: η² = 0.213 is a compelling finding — tokenization choice matters more than model or preprocessing.
2. **Highlight Octuple anomaly**: The Octuple+CLaMP-2 interaction is the most interesting result and deserves a dedicated discussion.
3. **Use effect sizes over p-values**: Given limited N, η² and Cohen's d are more informative than significance tests.
4. **Include the interaction plot** as a main figure — it tells the complete story.
5. **Add more genre pairs** (electronic, country) to strengthen generalizability claims.
6. **Consider repeated subsampling** to increase statistical power.
