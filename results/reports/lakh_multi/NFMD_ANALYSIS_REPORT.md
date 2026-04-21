# Normalized FMD (nFMD) Analysis

**Scale-invariant FMD enables fair cross-model comparison.**

## Design
- **Input:** 5350 FMD observations (NaN-filtered)
- **Metrics:** raw FMD, nFMD_trace (FMD/Tr), nFMD_norm (FMD/‖μ‖²)
- **Key question:** Does η²(model) drop after normalization?

## η² Comparison: Raw vs Normalized

| Factor | η²(fmd) | η²(nfmd_trace) | η²(nfmd_norm) |
|--------|---------|----------------|---------------|
| tokenizer | 0.0010 | 0.0142 | 0.0414 |
| model | 0.9617 | 0.7079 | 0.6534 |
| preprocess | 0.0011 | 0.0038 | 0.0053 |
| pair | 0.0067 | 0.0959 | 0.0066 |

## Per-Model η² (nFMD_trace)

| Model | η²(tokenizer) | η²(preprocess) |
|-------|---------------|----------------|
| CLaMP-1 | 0.0605 | 0.0002 |
| CLaMP-2 | 0.0839 | 0.0008 |
| MERT | 0.0055 | 0.0240 |
| MusicBERT | 0.3588 | 0.0111 |
| MusicBERT-large | 0.1851 | 0.0241 |
| NLP-Baseline | 0.0184 | 0.1344 |

## Conclusions

1. Raw FMD is dominated by model scale (η²≈0.96) — this is a **scale artefact**.
2. After trace-normalization, the model effect should decrease substantially.
3. Per-model analysis reveals which models are actually sensitive to tokenization.
4. nFMD enables **fair cross-model comparison** of pipeline choices.
