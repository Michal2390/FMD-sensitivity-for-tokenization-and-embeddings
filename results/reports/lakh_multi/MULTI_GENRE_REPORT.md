# Multi-Genre FMD Sensitivity Analysis

**Extension of single-pair analysis to strengthen generalizability.**

## Design
- **Genres:** rock, jazz, electronic, country → 6 pairs
- **Variants:** 32 (4 tokenizers × 2 models × 4 preprocessing)
- **Repeated subsampling:** 10× per variant×pair (n=100)
- **Total FMD observations:** 1920
- **Advantage:** Within-cell variance enables proper ANOVA with interactions

## One-Way ANOVA (aggregated)

| Factor | F | p-value | η² | Effect |
|--------|---|---------|-----|--------|
| tokenizer | 71.70 | 5.99e-44*** | 0.1009 | medium |
| model | 181.56 | 1.32e-39*** | 0.0865 | medium |
| preprocess | 13.23 | 1.50e-08*** | 0.0203 | small |
| pair | 105.04 | 3.77e-98*** | 0.2153 | **LARGE** |

## Three-Way ANOVA with Interactions

| Source | F | p-value | η² | Partial η² |
|--------|---|---------|-----|-----------|
| C(tokenizer) | 177.48 | 2.31e-101*** | 0.1009 | 0.2200 |
| C(model) | 456.13 | 7.98e-91*** | 0.0865 | 0.1946 |
| C(preprocess) | 35.69 | 1.97e-22*** | 0.0203 | 0.0537 |
| C(tokenizer):C(model) | 740.66 | 3.08e-318*** | 0.4212 | 0.5406 |
| C(tokenizer):C(preprocess) | 1.86 | 5.44e-02 | 0.0032 | 0.0088 |
| C(model):C(preprocess) | 1.98 | 1.16e-01 | 0.0011 | 0.0031 |
| C(tokenizer):C(model):C(preprocess) | 5.17 | 5.84e-07*** | 0.0088 | 0.0241 |

## Permutation Tests

| Factor | F | p_perm | Significant |
|--------|---|--------|-------------|
| tokenizer | 71.70 | 0.0010 | **Yes** |
| model | 181.56 | 0.0010 | **Yes** |
| preprocess | 13.23 | 0.0010 | **Yes** |

## Effect Sizes (Cohen's d)

| Comparison | d | Magnitude |
|------------|---|-----------|
| tokenizer: Octuple vs REMI | 0.895 | **large** |
| tokenizer: MIDI-Like vs REMI | 0.691 | medium |
| model: CLaMP-1 vs CLaMP-2 | 0.615 | medium |
| tokenizer: REMI vs TSD | -0.558 | medium |
| tokenizer: MIDI-Like vs Octuple | -0.273 | small |
| tokenizer: Octuple vs TSD | 0.221 | small |
| tokenizer: MIDI-Like vs TSD | 0.016 | negl. |

## Per-Pair FMD Statistics

| Pair | Mean FMD | Std | N |
|------|----------|-----|---|
| jazz_vs_country | 0.1570 | 0.0633 | 320 |
| rock_vs_electronic | 0.1766 | 0.0630 | 320 |
| rock_vs_jazz | 0.1772 | 0.0688 | 320 |
| rock_vs_country | 0.1891 | 0.0692 | 320 |
| jazz_vs_electronic | 0.2408 | 0.0810 | 320 |
| electronic_vs_country | 0.2681 | 0.1002 | 320 |

## η² Generalizability Across Pairs

Do the same factors drive FMD variance regardless of genre pair?

```
factor                  model  preprocess  tokenizer
pair                                                
electronic_vs_country  0.3799      0.0112     0.0876
jazz_vs_country        0.0125      0.0216     0.2254
jazz_vs_electronic     0.1790      0.0236     0.2146
rock_vs_country        0.1063      0.0701     0.0919
rock_vs_electronic     0.0581      0.0130     0.1834
rock_vs_jazz           0.0183      0.0485     0.2145
```

- **tokenizer**: η² range [0.0876, 0.2254], mean=0.1696, cv=0.34

- **model**: η² range [0.0125, 0.3799], mean=0.1257, cv=1.01

- **preprocess**: η² range [0.0112, 0.0701], mean=0.0313, cv=0.68

## Comparison with Single-Pair Analysis

| Factor | η² (rock-jazz only) | η² (6 pairs, 10 repeats) | Change |
|--------|--------------------|-----------------------------|--------|
| tokenizer | 0.2133 | 0.1009 | ↓ |
| model | 0.0183 | 0.0865 | ↑ |
| preprocess | 0.0524 | 0.0203 | ↓ |

## Key Conclusions

1. With 1920 observations (vs 32 in single-pair), statistical power is dramatically increased.
2. Genre pair choice is itself a significant factor — FMD values differ substantially between pairs.
3. The multi-pair analysis tests whether tokenizer sensitivity **generalizes** across genre contexts.
4. Three-way ANOVA with interactions is now properly estimable (multiple observations per cell).
