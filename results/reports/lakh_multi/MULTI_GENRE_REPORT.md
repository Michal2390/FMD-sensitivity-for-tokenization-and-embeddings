# Multi-Genre FMD Sensitivity Analysis

**Extension of single-pair analysis to strengthen generalizability.**

## Design
- **Genres:** rock, jazz, electronic, country → 6 pairs
- **Variants:** 48 (4 tokenizers × 2 models × 4 preprocessing)
- **Repeated subsampling:** 10× per variant×pair (n=100)
- **Total FMD observations:** 2880
- **Advantage:** Within-cell variance enables proper ANOVA with interactions

## One-Way ANOVA (aggregated)

| Factor | F | p-value | η² | Effect |
|--------|---|---------|-----|--------|
| tokenizer | 19.14 | 2.73e-12*** | 0.0196 | small |
| model | 5048.10 | 0.00e+00*** | 0.7782 | **LARGE** |
| preprocess | 2.13 | 9.47e-02 | 0.0022 | negl. |
| pair | 24.75 | 1.73e-24*** | 0.0413 | small |

## Bootstrap 95% CI for η² (5000 resamples)

| Factor | η² | 95% CI lower | 95% CI upper |
|--------|----|-------------|-------------|
| tokenizer | 0.0196 | 0.0133 | 0.0290 |
| model | 0.7782 | 0.7637 | 0.7927 |
| preprocess | 0.0022 | 0.0005 | 0.0078 |
| tokenizer:model | 0.8484 | 0.8339 | 0.8637 |

## Three-Way ANOVA with Interactions

| Source | F | p-value | η² | Partial η² |
|--------|---|---------|-----|-----------|
| C(tokenizer) | 128.07 | 8.61e-78*** | 0.0196 | 0.1195 |
| C(model) | 7636.13 | 0.00e+00*** | 0.7782 | 0.8436 |
| C(preprocess) | 14.48 | 2.32e-09*** | 0.0022 | 0.0151 |
| C(tokenizer):C(model) | 165.48 | 1.02e-180*** | 0.0506 | 0.2596 |
| C(tokenizer):C(preprocess) | 2.01 | 3.40e-02* | 0.0009 | 0.0064 |
| C(model):C(preprocess) | 8.67 | 2.28e-09*** | 0.0026 | 0.0180 |
| C(tokenizer):C(model):C(preprocess) | 1.63 | 4.56e-02* | 0.0015 | 0.0102 |

## Permutation Tests

| Factor | F | p_perm | Significant |
|--------|---|--------|-------------|
| tokenizer | 19.14 | 0.0010 | **Yes** |
| model | 5048.10 | 0.0010 | **Yes** |
| preprocess | 2.13 | 0.0959 | No |

## Effect Sizes (Cohen's d)

| Comparison | d | Magnitude |
|------------|---|-----------|
| model: CLaMP-2 vs MusicBERT | -3.295 | **large** |
| model: CLaMP-1 vs MusicBERT | -3.216 | **large** |
| model: CLaMP-1 vs CLaMP-2 | 0.634 | medium |
| tokenizer: Octuple vs TSD | -0.405 | small |
| tokenizer: Octuple vs REMI | -0.368 | small |
| tokenizer: MIDI-Like vs Octuple | 0.254 | small |
| tokenizer: MIDI-Like vs TSD | -0.130 | negl. |
| tokenizer: MIDI-Like vs REMI | -0.102 | negl. |
| tokenizer: REMI vs TSD | -0.026 | negl. |

## Per-Pair FMD Statistics

| Pair | Mean FMD | Std | N |
|------|----------|-----|---|
| jazz_vs_country | 0.6126 | 0.6973 | 480 |
| rock_vs_jazz | 0.7297 | 0.8674 | 480 |
| rock_vs_electronic | 0.8166 | 0.9573 | 480 |
| rock_vs_country | 0.8428 | 1.0366 | 480 |
| jazz_vs_electronic | 1.0917 | 1.2525 | 480 |
| electronic_vs_country | 1.2867 | 1.5396 | 480 |

## η² Generalizability Across Pairs

Do the same factors drive FMD variance regardless of genre pair?

```
factor                  model  preprocess  tokenizer
pair                                                
electronic_vs_country  0.8750      0.0013     0.0353
jazz_vs_country        0.8504      0.0037     0.0349
jazz_vs_electronic     0.9299      0.0017     0.0131
rock_vs_country        0.8001      0.0044     0.0573
rock_vs_electronic     0.8986      0.0004     0.0219
rock_vs_jazz           0.8159      0.0085     0.0409
```

- **tokenizer**: η² range [0.0131, 0.0573], mean=0.0339, cv=0.41

- **model**: η² range [0.8001, 0.9299], mean=0.8616, cv=0.05

- **preprocess**: η² range [0.0004, 0.0085], mean=0.0034, cv=0.80

## Comparison with Single-Pair Analysis

| Factor | η² (rock-jazz only) | η² (6 pairs, 10 repeats) | Change |
|--------|--------------------|-----------------------------|--------|
| tokenizer | 0.2133 | 0.0196 | ↓ |
| model | 0.0183 | 0.7782 | ↑ |
| preprocess | 0.0524 | 0.0022 | ↓ |

## Key Conclusions

1. With 2880 observations (vs 32 in single-pair), statistical power is dramatically increased.
2. Genre pair choice is itself a significant factor — FMD values differ substantially between pairs.
3. The multi-pair analysis tests whether tokenizer sensitivity **generalizes** across genre contexts.
4. Three-way ANOVA with interactions is now properly estimable (multiple observations per cell).
