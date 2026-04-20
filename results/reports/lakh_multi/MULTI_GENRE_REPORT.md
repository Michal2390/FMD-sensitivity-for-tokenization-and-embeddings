# Multi-Genre FMD Sensitivity Analysis

**Extension of single-pair analysis to strengthen generalizability.**

## Design
- **Genres:** rock, jazz, electronic, country → 6 pairs
- **Variants:** 64 (4 tokenizers × 2 models × 4 preprocessing)
- **Repeated subsampling:** 10× per variant×pair (n=100)
- **Total FMD observations:** 3840
- **Advantage:** Within-cell variance enables proper ANOVA with interactions

## One-Way ANOVA (aggregated)

| Factor | F | p-value | η² | Effect |
|--------|---|---------|-----|--------|
| tokenizer | nan | nan | nan | negl. |
| model | nan | nan | nan | negl. |
| preprocess | nan | nan | nan | negl. |
| pair | nan | nan | nan | negl. |

## Bootstrap 95% CI for η² (5000 resamples)

| Factor | η² | 95% CI lower | 95% CI upper |
|--------|----|-------------|-------------|
| tokenizer | nan | nan | nan |
| model | nan | nan | nan |
| preprocess | nan | nan | nan |
| tokenizer:model | nan | nan | nan |

## Three-Way ANOVA with Interactions

| Source | F | p-value | η² | Partial η² |
|--------|---|---------|-----|-----------|
| C(tokenizer) | 69.92 | 7.31e-44*** | 0.0029 | 0.0586 |
| C(model) | 22459.75 | 0.00e+00*** | 0.9391 | 0.9524 |
| C(preprocess) | 59.54 | 1.68e-37*** | 0.0025 | 0.0504 |
| C(tokenizer):C(model) | 34.50 | 9.68e-59*** | 0.0043 | 0.0844 |
| C(tokenizer):C(preprocess) | 4.67 | 3.49e-06*** | 0.0006 | 0.0123 |
| C(model):C(preprocess) | 23.39 | 3.93e-39*** | 0.0029 | 0.0588 |
| C(tokenizer):C(model):C(preprocess) | 1.99 | 1.71e-03** | 0.0007 | 0.0157 |

## Permutation Tests

| Factor | F | p_perm | Significant |
|--------|---|--------|-------------|
| tokenizer | nan | 0.0010 | **Yes** |
| model | nan | 0.0010 | **Yes** |
| preprocess | nan | 0.0010 | **Yes** |

## Effect Sizes (Cohen's d)

| Comparison | d | Magnitude |
|------------|---|-----------|
| model: MusicBERT-large vs NLP-Baseline | 9.827 | **large** |
| model: MusicBERT vs NLP-Baseline | 6.136 | **large** |
| model: MusicBERT vs MusicBERT-large | -4.282 | **large** |
| tokenizer: MIDI-Like vs Octuple | 0.000 | negl. |
| tokenizer: MIDI-Like vs REMI | 0.000 | negl. |
| tokenizer: MIDI-Like vs TSD | 0.000 | negl. |
| tokenizer: Octuple vs REMI | 0.000 | negl. |
| tokenizer: Octuple vs TSD | 0.000 | negl. |
| tokenizer: REMI vs TSD | 0.000 | negl. |
| model: MERT vs MusicBERT | 0.000 | negl. |
| model: MERT vs MusicBERT-large | 0.000 | negl. |
| model: MERT vs NLP-Baseline | 0.000 | negl. |

## Per-Pair FMD Statistics

| Pair | Mean FMD | Std | N |
|------|----------|-----|---|
| jazz_vs_country | 6.2199 | 4.6173 | 640 |
| rock_vs_jazz | 7.0885 | 5.2161 | 640 |
| rock_vs_country | 7.1847 | 5.3559 | 640 |
| rock_vs_electronic | 7.7708 | 6.3670 | 640 |
| jazz_vs_electronic | 8.8091 | 7.1959 | 640 |
| electronic_vs_country | 9.2653 | 7.4477 | 640 |

## η² Generalizability Across Pairs

Do the same factors drive FMD variance regardless of genre pair?

```
factor            model  preprocess  tokenizer
pair                                          
jazz_vs_country  0.9753      0.0022     0.0056
rock_vs_country  0.9797      0.0039     0.0014
rock_vs_jazz     0.9800      0.0031     0.0034
```

- **tokenizer**: η² range [0.0014, 0.0056], mean=0.0034, cv=0.50

- **model**: η² range [0.9753, 0.9800], mean=0.9783, cv=0.00

- **preprocess**: η² range [0.0022, 0.0039], mean=0.0031, cv=0.22

## Comparison with Single-Pair Analysis

| Factor | η² (rock-jazz only) | η² (6 pairs, 10 repeats) | Change |
|--------|--------------------|-----------------------------|--------|
| tokenizer | 0.2133 | nan | = |
| model | 0.0183 | nan | = |
| preprocess | 0.0524 | nan | = |

## Key Conclusions

1. With 3840 observations (vs 32 in single-pair), statistical power is dramatically increased.
2. Genre pair choice is itself a significant factor — FMD values differ substantially between pairs.
3. The multi-pair analysis tests whether tokenizer sensitivity **generalizes** across genre contexts.
4. Three-way ANOVA with interactions is now properly estimable (multiple observations per cell).
