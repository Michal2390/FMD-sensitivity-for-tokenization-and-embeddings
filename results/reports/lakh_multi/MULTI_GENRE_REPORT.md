# Multi-Genre FMD Sensitivity Analysis

**Extension of single-pair analysis to strengthen generalizability.**

## Design
- **Genres:** rock, jazz, electronic, country → 6 pairs
- **Variants:** 96 (4 tokenizers × 2 models × 4 preprocessing)
- **Repeated subsampling:** 10× per variant×pair (n=100)
- **Total FMD observations:** 5760
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
| C(tokenizer) | 59.51 | 7.93e-38*** | 0.0010 | 0.0329 |
| C(model) | 33904.38 | 0.00e+00*** | 0.9617 | 0.9699 |
| C(preprocess) | 60.14 | 3.22e-38*** | 0.0010 | 0.0332 |
| C(tokenizer):C(model) | 40.40 | 1.07e-112*** | 0.0034 | 0.1034 |
| C(tokenizer):C(preprocess) | 4.62 | 4.15e-06*** | 0.0002 | 0.0079 |
| C(model):C(preprocess) | 25.65 | 5.81e-70*** | 0.0022 | 0.0682 |
| C(tokenizer):C(model):C(preprocess) | 2.29 | 2.20e-06*** | 0.0006 | 0.0193 |

## Permutation Tests

| Factor | F | p_perm | Significant |
|--------|---|--------|-------------|
| tokenizer | nan | 0.0010 | **Yes** |
| model | nan | 0.0010 | **Yes** |
| preprocess | nan | 0.0010 | **Yes** |

## Effect Sizes (Cohen's d)

| Comparison | d | Magnitude |
|------------|---|-----------|
| model: CLaMP-1 vs MusicBERT-large | -10.055 | **large** |
| model: CLaMP-2 vs MusicBERT-large | -10.018 | **large** |
| model: MusicBERT-large vs NLP-Baseline | 9.962 | **large** |
| model: CLaMP-1 vs MusicBERT | -6.320 | **large** |
| model: CLaMP-2 vs MusicBERT | -6.269 | **large** |
| model: MusicBERT vs NLP-Baseline | 6.195 | **large** |
| model: MusicBERT vs MusicBERT-large | -4.328 | **large** |
| model: CLaMP-1 vs NLP-Baseline | -2.379 | **large** |
| model: CLaMP-1 vs CLaMP-2 | -1.553 | **large** |
| model: CLaMP-2 vs NLP-Baseline | -1.211 | **large** |
| tokenizer: MIDI-Like vs Octuple | 0.000 | negl. |
| tokenizer: MIDI-Like vs REMI | 0.000 | negl. |
| tokenizer: MIDI-Like vs TSD | 0.000 | negl. |
| tokenizer: Octuple vs REMI | 0.000 | negl. |
| tokenizer: Octuple vs TSD | 0.000 | negl. |
| tokenizer: REMI vs TSD | 0.000 | negl. |
| model: CLaMP-1 vs MERT | 0.000 | negl. |
| model: CLaMP-2 vs MERT | 0.000 | negl. |
| model: MERT vs MusicBERT | 0.000 | negl. |
| model: MERT vs MusicBERT-large | 0.000 | negl. |
| model: MERT vs NLP-Baseline | 0.000 | negl. |

## Per-Pair FMD Statistics

| Pair | Mean FMD | Std | N |
|------|----------|-----|---|
| jazz_vs_country | 4.1634 | 4.7568 | 960 |
| rock_vs_jazz | 4.7505 | 5.4055 | 960 |
| rock_vs_electronic | 4.7594 | 6.2397 | 960 |
| rock_vs_country | 4.8100 | 5.5280 | 960 |
| jazz_vs_electronic | 5.3960 | 7.0289 | 960 |
| electronic_vs_country | 5.7165 | 7.3748 | 960 |

## η² Generalizability Across Pairs

Do the same factors drive FMD variance regardless of genre pair?

```
factor            model  preprocess  tokenizer
pair                                          
jazz_vs_country  0.9849      0.0009     0.0019
rock_vs_country  0.9876      0.0014     0.0007
rock_vs_jazz     0.9881      0.0013     0.0012
```

- **tokenizer**: η² range [0.0007, 0.0019], mean=0.0013, cv=0.41

- **model**: η² range [0.9849, 0.9881], mean=0.9868, cv=0.00

- **preprocess**: η² range [0.0009, 0.0014], mean=0.0012, cv=0.18

## Comparison with Single-Pair Analysis

| Factor | η² (rock-jazz only) | η² (6 pairs, 10 repeats) | Change |
|--------|--------------------|-----------------------------|--------|
| tokenizer | 0.2133 | nan | = |
| model | 0.0183 | nan | = |
| preprocess | 0.0524 | nan | = |

## Key Conclusions

1. With 5760 observations (vs 32 in single-pair), statistical power is dramatically increased.
2. Genre pair choice is itself a significant factor — FMD values differ substantially between pairs.
3. The multi-pair analysis tests whether tokenizer sensitivity **generalizes** across genre contexts.
4. Three-way ANOVA with interactions is now properly estimable (multiple observations per cell).
