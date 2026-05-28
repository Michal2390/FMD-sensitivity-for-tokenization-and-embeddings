"""Helper script to generate the new README.md for the sensitivity pivot."""
import pathlib

# Read the detailed results doc
results_content = pathlib.Path('docs/SENSITIVITY_PIVOT_RESULTS.md').read_text()

readme = """# FMD Sensitivity to Tokenization and Embedding Configuration

**Sensitivity Profiling of Frechet Music Distance for Symbolic Music Evaluation**

> An empirical study revealing how pipeline configuration choices (embedding model, tokenization, preprocessing) alter what FMD actually measures -- with practical recommendations for the music generation community.

---

## TL;DR

We profiled how 3 FMD configurations react to controlled perturbations of MIDI data. The key discovery:

| Perturbation | CLaMP2-ABC | CLaMP2-MTF | CLaMP1-ABC |
|:-------------|:----------:|:----------:|:----------:|
| Remove velocity (dynamics) | **0.510** | **0.408** | 0.022 |
| Quantize timing (16th grid) | 0.008 | 0.008 | 0.007 |
| Constant tempo (120 BPM) | 0.000 | 0.000 | 0.000 |

**CLaMP-2 sees velocity; CLaMP-1 is blind to it. No model sees tempo or microtiming. Configuration choice determines what musical aspects FMD actually evaluates.**

---

## Motivation and Research Pivot

### Original Approach: Normalized FMD (nFMD) -- Why It Failed

Our initial contribution was **Normalized FMD (nFMD)** -- an attempt to make FMD values comparable across different embedding models by normalizing for embedding scale:

- `nFMD_trace = FMD / (Tr(S1) + Tr(S2))` -- normalize by total variance
- `nFMD_norm = FMD / (||mu1|| + ||mu2||)^2` -- compensate quadratic mean scaling

We observed that raw FMD values differ by **12.8x** across models (e.g., MusicBERT FMD = 8.0 vs CLaMP-2 FMD = 0.6 for the same genre pair) simply due to embedding norm differences. nFMD was designed to fix this.

**Why nFMD is fundamentally flawed:**

1. **The problem it solves does not exist in a meaningful way.** Yes, FMD scales differ across models -- but that is because each model lives in a different feature space optimized for different objectives (contrastive vs masked LM vs audio SSL). Forcing them to the same numeric scale does not make them measure the same thing -- it just makes them *look* like they measure the same thing, which is worse.

2. **FMD/FID/FAD were never designed for cross-model comparison.** The Frechet distance measures distributional divergence *within* a fixed feature space. Comparing FMD(model_A) with FMD(model_B) is mathematically like comparing meters with kilograms -- different units from fundamentally different measurement systems.

3. **Normalization obscures rather than reveals.** After normalization, models with completely different properties appear interchangeable. The apparent hidden tokenizer effects we found (eta-sq increasing from 0.001 to 0.014 after normalization) were likely artefacts of dividing by an arbitrary scaling factor, not genuine signal recovery.

4. **Precedent agrees.** FID (Heusel et al. 2017) and FAD (Kilgour et al. 2019) -- the metrics from which FMD descends -- have never been normalized. The research community has consistently treated the embedding model as a fixed choice, not a variable to normalize across.

We implemented nFMD experimentally (5350 observations, 6 models, full ANOVA), demonstrated it numerically works to reduce model dominance (eta-sq model drops from 0.96 to 0.71), but ultimately rejected it as a paper contribution because it produces misleading results.

### The Pivot: Sensitivity Profiling

Instead of trying to make FMD cross-model comparable, we asked a more scientifically useful question:

> **Given a fixed FMD configuration, what musical properties does the metric actually measure?**

This reframing produces directly actionable knowledge for the symbolic music generation community:
- If you are evaluating whether your model produces expressive dynamics -- use CLaMP-2
- If you are evaluating harmonic/melodic structure regardless of expression -- use CLaMP-1
- If you think your FMD is measuring timing quality -- it is not (for any tested configuration)

---

## Experimental Design

### 3 Configurations (isolating variables)

| Config | Model | Tokenization | What it isolates |
|--------|-------|-------------|----------|
| **CLaMP2-ABC** | CLaMP-2 | REMI (text-like) | Baseline |
| **CLaMP2-MTF** | CLaMP-2 | MIDI-Like (full MIDI fidelity) | **tokenization effect** (same model, different tokens) |
| **CLaMP1-ABC** | CLaMP-1 | REMI (text-like) | **model effect** (same tokens, different model) |

### 3 Datasets (stylistically distinct)

| Dataset | Style | N files | Source |
|---------|-------|---------|--------|
| **MAESTRO** | Classical piano, virtuosic | 1276 | Google Magenta v3 |
| **POP909** | Pop songs | 2898 | Music-X-Lab |
| **Folk** | Traditional folk tunes | 1034 | Nottingham Dataset |

### 5 Perturbations (controlled expression removal)

| Perturbation | What it removes | Implementation |
|:-------------|:----------------|:---------------|
| `original` | Nothing (baseline) | -- |
| `no_velocity` | Dynamics/expression | All notes velocity 64 |
| `quantized_time` | Microtiming/swing | Snap to 16th-note grid |
| `constant_tempo` | Rubato/tempo variation | Remap all beats to 120 BPM |
| `all_combined` | All expression | All three above combined |

---

""" + results_content + """

---

## Running the Experiments

### Full sensitivity pivot (~27 minutes, CPU):
```bash
python main.py --mode sensitivity
```

### Individual steps:
```bash
python main.py --mode sensitivity --sensitivity-step self-similarity
python main.py --mode sensitivity --sensitivity-step ranking
python main.py --mode sensitivity --sensitivity-step perturbation
python main.py --mode sensitivity --sensitivity-step bootstrap
python main.py --mode sensitivity --sensitivity-step plots
```

### Dataset preparation:
```bash
python scripts/download_folk_dataset.py
python main.py --mode fetch-data --datasets maestro pop909
```

### Output locations:
- CSV results: `results/reports/sensitivity_pivot/`
- Plots: `results/plots/sensitivity_pivot/`
- Summary JSON: `results/reports/sensitivity_pivot/sensitivity_pivot_summary.json`

---

## Project Structure

```
src/
  preprocessing/processor.py      # MIDI preprocessing + normalize_tempo()
  tokenization/tokenizer.py       # REMI, TSD, Octuple, MIDI-Like
  embeddings/extractor.py         # CLaMP-1, CLaMP-2, MusicBERT, MERT, NLP
  metrics/fmd.py                  # Frechet Music Distance
  experiments/
    sensitivity_profiler.py       # Main pivot pipeline (7 steps)
    paper_pipeline.py             # Legacy multi-model analysis

configs/
  config.yaml                     # Main project config
  sensitivity_pivot.yaml          # Pivot experiment config

results/
  reports/sensitivity_pivot/      # CSV + JSON results
  plots/sensitivity_pivot/        # Publication figures
```

---

## Previous Work (nFMD and Multi-Model ANOVA)

Our earlier 6-model ANOVA analysis (5760 observations, Lakh MIDI) showed:
- Model choice explains 96% of FMD variance (eta-sq = 0.962)
- Tokenizer effect is negligible when pooled across models (eta-sq = 0.001)
- Cross-dataset Spearman rho = 0.975 (Lakh vs MidiCaps)

These results motivated the pivot: since model choice so thoroughly dominates, comparing raw FMD across models is meaningless -- better to fix the model and study what the metric actually captures.

Legacy run modes: `python main.py --mode paper` | `--mode lakh` | `--mode cross-validate`

---

## References

1. Retkowski, J., Stepniak, J., Modrzejewski, M. (2025). *Frechet Music Distance: A Metric for Generative Symbolic Music Evaluation.*
2. Wu, Y., et al. (2023). *CLaMP: Contrastive Language-Music Pre-training.*
3. Wu, Y., et al. (2024). *CLaMP 2: Multimodal Music Information Retrieval.*
4. Fradet, N., et al. (2024). *MidiTok: A Python Package for MIDI File Tokenization.*
5. Heusel, M., et al. (2017). *GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium.* (FID)
6. Kilgour, K., et al. (2019). *Frechet Audio Distance: A Reference-Free Metric for Evaluating Music Enhancement Algorithms.* (FAD)

## Academic Context

**Project**: Frechet Music Distance -- Sensitivity to Tokenization and Embedding Configuration  
**Institution**: Warsaw University of Technology, EITI  
**Course**: WIMU (Music Information Retrieval)  
**Authors**: Michal Fereniec, Bartlomiej Sedlak  
**Supervisor**: dr inz. Jakub Retkowski  
**Duration**: February-June 2026  
**Presentation**: June 11, 2026

---

**Status**: Results Complete | **Last Updated**: 2026-05-28
"""

pathlib.Path('README.md').write_text(readme)
print(f"README.md written: {len(readme)} chars, {readme.count(chr(10))} lines")

