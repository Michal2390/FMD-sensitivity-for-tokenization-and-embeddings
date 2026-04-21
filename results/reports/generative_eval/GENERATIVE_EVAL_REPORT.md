# Generative Model Evaluation Report

**Validates FMD/nFMD ranking: real < Markov < random**

## Design
- Real: 120 Lakh rock MIDI files
- Markov: 100 generated (first-order chain)
- Random: 100 completely random
- Variants: 4 tokenizers × 6 models
- Total observations: 72

## Ranking Correctness (real < Markov < random)

| Metric | % Models Correct |
|--------|-----------------|
| fmd | 17% |
| nfmd_trace | 33% |
| nfmd_norm | 33% |