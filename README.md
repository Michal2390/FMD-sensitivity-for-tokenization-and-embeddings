# 🎵 FMD Sensitivity to Tokenization and Embedding Configuration

<p align="center">
  <b>🔬 What does the Fréchet Music Distance actually measure? A sensitivity study across tokenizers and embedding models.</b>
</p>

<p align="center">
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python"></a>
  <a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch"></a>
  <a href="#"><img src="https://img.shields.io/badge/Status-Results_Complete-success?style=for-the-badge" alt="Status"></a>
  <a href="#"><img src="https://img.shields.io/badge/License-MIT-blue?style=for-the-badge" alt="License"></a>
</p>

> 📖 An FMD score is **not a property of the music alone** — it is produced by a *pipeline* (input representation → embedding model → Fréchet formula). We profile how that pipeline reshapes what FMD perceives, and give practical guidance for the music-generation community.

---

## ⚡ TL;DR

We profiled **4 FMD configurations** across **6 datasets** (15 pairs) and **5 controlled perturbations** of MIDI. Three things stand out:

1. **🎚️ Input representation decides what FMD can see.** Removing note velocity is the *only* perturbation that moves the embedding distribution above the sampling-noise floor (SNR > 1) — and only in representations that actually encode velocity (REMI/TSD tokens, CLaMP-2 MTF). ABC notation, which has no velocity channel, stays at noise.
2. **🤝 Representations that preserve performance detail rank genres the same way.** Cross-dataset rankings of **REMI and CLaMP-2/MTF agree strongly (Spearman ρ = 0.825, p < 0.001)**, while **ABC agrees with no one (ρ ≈ 0.04–0.15, p > 0.6)**.
3. **⚠️ Raw FMD is NOT comparable across embedding models.** CLaMP embeddings are L2-normalised (unit sphere → small FMD); MusicBERT's are not (→ large FMD). The ~60× gap is geometry, not music — so we compare via **scale-invariant** statistics (SNR, Spearman, CV), never raw magnitude.

<p align="center">
  <img src="results/plots/sensitivity_pivot/paper/fig2_snr_heatmap.png" alt="Perturbation SNR heatmap" width="560">
</p>

---

## 🧭 What changed (and why)

An earlier version of this study mislabelled configurations (a "CLaMP-2 + REMI" entry that actually fed ABC) and computed rank statistics over too few points. This version fixes both:

| Problem (old) | Fix (now) |
|:--|:--|
| ❌ CLaMP fed REMI tokens it cannot natively consume | ✅ REMI/TSD tokens go to **MusicBERT** (a token model); CLaMP only gets its **native** MTF/ABC |
| ❌ Rank stats over 3 points (indefensible) | ✅ **6 datasets → 15 pairs**, Spearman now interpretable with real *p*-values |
| ❌ Single perturbation numbers, no test | ✅ Each perturbation gets **SNR + bootstrap CI + permutation *p*-value** |
| ❌ "CLaMP 63× more consistent" (a scale artifact) | ✅ Cross-config claims use **scale-invariant** SNR / Spearman / CV only |

---

## 🧪 Experimental Design

### 🔧 4 Configurations (honest input semantics)

| Config | Model | Input | Isolates | Norm. |
|:------:|:-----:|:------|:---------|:-----:|
| 🅰️ **MusicBERT-REMI** | MusicBERT | MidiTok REMI tokens | tokenization baseline | no |
| 🅱️ **MusicBERT-TSD** | MusicBERT | MidiTok TSD tokens | tokenizer effect (same model) | no |
| 🅲 **CLaMP2-MTF** | CLaMP-2 | MIDI-Text Format (native) | model effect, expressive path | yes |
| 🅳 **CLaMP1-ABC** | CLaMP-1 | ABC notation (`music21`) | score-like path (no velocity) | yes |

Encoding details: [`src/embeddings/clamp_formats.py`](src/embeddings/clamp_formats.py). All models output 768-d embeddings.

### 🎶 6 Datasets → 15 pairs

`maestro` (expressive classical piano), `pop909` (pop), `classical`, `jazz`, `rock`, `rap` (Lakh-derived genre subsets). 80 files each, fixed seed. **C(6,2) = 15 pairs** — enough for an interpretable Spearman rank agreement (the code flags it interpretable at n ≥ 10).

### 🎛️ 5 Perturbations (controlled expression removal)

`original` · `no_velocity` (dynamics → 64) · `quantized_time` (16th-grid) · `constant_tempo` (120 BPM) · `all_combined`.

---

## 📊 Results

### 1️⃣ Noise floor (self-similarity) — read scales *within* a config, not across

Split-half FMD (should be ≈ 0 for a stable pipeline). MusicBERT sits ~60× higher than CLaMP **purely because its embeddings are unnormalised** — this is the reason every cross-config statement below is scale-invariant.

| Config | maestro | pop909 | classical | jazz | rock | rap |
|:--|:--:|:--:|:--:|:--:|:--:|:--:|
| MusicBERT-REMI | 2.12 | 1.57 | 1.88 | 1.80 | 1.58 | 1.71 |
| MusicBERT-TSD | 2.31 | 2.32 | 1.51 | 1.35 | 1.21 | 1.27 |
| CLaMP2-MTF | 0.036 | 0.029 | 0.027 | 0.021 | 0.023 | 0.020 |
| CLaMP1-ABC | 0.020 | 0.026 | 0.025 | 0.031 | 0.025 | 0.025 |

<p align="center">
  <img src="results/plots/sensitivity_pivot/paper/fig1_noise_floor.png" alt="Per-configuration noise floor" width="560">
</p>

### 2️⃣ Perturbation sensitivity ⭐ — SNR = FMD / noise floor

`SNR > 1` means the perturbation moves the distribution more than within-dataset sampling noise. Significance is a **two-sample permutation test** (200 permutations); CIs are bootstrap (100 resamples).

| Perturbation | MusicBERT-REMI | MusicBERT-TSD | CLaMP2-MTF | CLaMP1-ABC |
|:--|:--:|:--:|:--:|:--:|
| 🔇 **Velocity removed** | **3.16** ✓ | **1.35** ✓ | **1.18** ✓ | 0.67 |
| 📐 Timing quantized | 0.05 | 0.11 | 0.66 | 0.62 |
| ⏱️ Tempo flattened | 0.06 | 0.14 | 0.13 | 0.83 |
| 💀 All combined | **3.30** ✓ | **1.71** ✓ | **1.55** ✓ | 0.90 |

> 🎯 **Velocity is the only attribute detected above the noise floor — and only where it is represented.** ✓ = SNR > 1 **and** permutation-significant (p < 0.05). REMI/TSD tokens and MTF carry velocity → all three detected (p ≤ 0.02); **ABC has no velocity channel and stays at noise (0.67, p = 0.06).** Microtiming and tempo fall at/below the floor for every configuration. The *combined* effect simply tracks velocity. (ABC does show a tiny *systematic* shift for tempo/combined — permutation p < 0.05 — but always **below** its own sampling noise, i.e. systematic yet negligible.)

### 3️⃣ Cross-dataset ranking + Spearman agreement ⭐ — the rigorous backbone

FMD ranks the 15 dataset pairs differently depending on the pipeline. Rank agreement between configurations (scale-invariant, **n = 15**):

| Pair of configurations | Spearman ρ | p | Reading |
|:--|:--:|:--:|:--|
| MusicBERT-REMI ↔ **CLaMP2-MTF** | **0.825** | **< 0.001** | 🟢 strong agreement |
| MusicBERT-TSD ↔ CLaMP2-MTF | 0.525 | 0.044 | 🟡 moderate |
| MusicBERT-REMI ↔ MusicBERT-TSD | 0.436 | 0.104 | ⚪ tokenizer changes ranking |
| MusicBERT-TSD ↔ CLaMP1-ABC | 0.321 | 0.243 | ⚪ none |
| MusicBERT-REMI ↔ CLaMP1-ABC | 0.146 | 0.603 | ⚪ none |
| CLaMP2-MTF ↔ **CLaMP1-ABC** | 0.039 | 0.889 | 🔴 no agreement |

> 🔑 **Representations that preserve performance detail (REMI, MTF) rank genres the same way; ABC ranks them differently and nearly flatly.** What FMD calls "stylistic distance" depends on the pipeline.

<p align="center">
  <img src="results/plots/sensitivity_pivot/paper/fig5_spearman_heatmap.png" alt="Spearman rank agreement" width="460">
</p>

### 4️⃣ Bootstrap stability (MAESTRO vs POP909, 50 resamples, N = 80)

Coefficient of variation (CV) is scale-invariant and *is* comparable across configs:

| Config | FMD mean ± std | 95% CI | CV |
|:--|:--:|:--:|:--:|
| MusicBERT-REMI | 3.51 ± 0.53 | [2.66, 4.64] | 15.1% |
| MusicBERT-TSD | 3.92 ± 0.79 | [2.74, 5.47] | 20.2% |
| CLaMP2-MTF | 0.048 ± 0.007 | [0.039, 0.060] | 14.0% |
| CLaMP1-ABC | 0.024 ± 0.003 | [0.019, 0.030] | 11.0% |

> 📊 All four are comparably stable (CV 11–20%); MusicBERT-TSD is the noisiest. **Do not compare the mean columns across the CLaMP/MusicBERT boundary** — different spaces.

---

## 🏆 Conclusions

| # | Finding | Evidence |
|:-:|:--|:--|
| 1️⃣ | **Input representation sets a ceiling on what FMD can detect** | velocity SNR 1.2–3.2 (p ≤ 0.02) in REMI/TSD/MTF vs 0.67 in ABC |
| 2️⃣ | **Rankings agree only between detail-preserving representations** | REMI↔MTF ρ=0.825 (p<0.001); ABC↔all ρ≈0 |
| 3️⃣ | **Tokenizer alone reshapes sensitivity** | REMI vs TSD velocity SNR 3.16 vs 1.35; ranking ρ=0.44 |
| 4️⃣ | **Raw FMD is not cross-model comparable** | noise floors differ ~60× from normalisation, not music |

### 🎯 Practical recommendations

| Goal | ✅ Use | ❌ Avoid | Why |
|:--|:--|:--|:--|
| 🎹 Dynamics / expression | MusicBERT-REMI or CLaMP-2/MTF | CLaMP-1/ABC | ABC discards velocity entirely |
| 🎼 Score / folk structure | CLaMP-1/ABC | — | velocity-invariant by design |
| 📊 Stylistic-distance ranking | CLaMP-2/MTF | CLaMP-1/ABC | ABC ranks genres nearly flatly |
| 🔁 Cross-model comparison | SNR / Spearman / CV | raw FMD magnitude | scale is geometry, not music |

---

## 🚀 Reproducing the results

```bash
# Full 7-step pipeline (≈30–60 min on a single GPU)
python main.py --mode sensitivity

# Individual steps
python main.py --mode sensitivity --sensitivity-step self-similarity
python main.py --mode sensitivity --sensitivity-step ranking
python main.py --mode sensitivity --sensitivity-step perturbation
python main.py --mode sensitivity --sensitivity-step bootstrap

# Figures + LaTeX tables for the paper (regenerated from the result CSVs)
python scripts/generate_draft_figures.py
python scripts/generate_draft_tables.py
```

Config: [`configs/sensitivity_pivot.yaml`](configs/sensitivity_pivot.yaml). Install deps with `pip install -r requirements.txt` (includes `unidecode`, required by the CLaMP M3 patcher).

### 📁 Output

```
results/reports/sensitivity_pivot/      # CSV + JSON results
  ├── self_similarity.csv               # noise floor, 6 datasets × 4 configs
  ├── cross_dataset_fmd.csv             # 15 pairs × 4 configs
  ├── spearman_ranking_agreement.csv    # rank agreement (n=15, interpretable)
  ├── perturbation_sensitivity.csv      # FMD, SNR, CI, permutation p
  ├── bootstrap_stability.csv           # mean ± std, CI, CV
  └── tables/*.tex                      # auto-generated LaTeX tables
results/plots/sensitivity_pivot/paper/  # publication figures (fig1…fig6)
```

The scientific write-up is in [`draft.tex`](draft.tex); a plain-language findings summary is in [`docs/PAPER_FINDINGS.md`](docs/PAPER_FINDINGS.md).

---

## 🗂️ Project structure

```
main.py                                 # CLI entry point (modes: sensitivity, full, fetch-data, tests, demo, lint)
configs/
  ├── config.yaml                       # infrastructure config (models, tokenizers, FMD)
  └── sensitivity_pivot.yaml            # the experiment: configs, datasets, perturbations, seeds
src/
  ├── experiments/
  │   ├── sensitivity_profiler.py       # the 7-step pipeline (noise floor, SNR, permutation, Spearman, bootstrap)
  │   └── study_config.py               # validates model↔input pairings (rejects e.g. CLaMP+REMI)
  ├── embeddings/{extractor.py, clamp_formats.py}   # MusicBERT / CLaMP-1/2 encoders, MTF/ABC builders
  ├── metrics/fmd.py                    # Fréchet Music Distance (standard formula; no cross-model normalisation)
  ├── preprocessing/processor.py        # MIDI loading + the three perturbations
  ├── tokenization/tokenizer.py         # MidiTok REMI / TSD / Octuple / MIDI-Like
  └── data/                             # dataset manager + genre/MidiCaps loaders
scripts/{generate_draft_figures.py, generate_draft_tables.py}   # paper figures & tables from CSVs
tests/                                  # unit tests (pytest) — all green
draft.tex                               # the paper
results/reports|plots/sensitivity_pivot/   # CSV/JSON results + figures
```

> The repository is focused on the sensitivity study. An earlier multi-model / normalised-FMD (nFMD) exploration was retired because raw FMD is not comparable across embedding spaces; its code, results, and notes have been removed to keep the project unambiguous.

---

## 📖 References

1. Retkowski, J., Stępniak, J., Modrzejewski, M. (2025). *Fréchet Music Distance: A Metric for Generative Symbolic Music Evaluation.*
2. Zeng, M., et al. (2021). *MusicBERT: Symbolic Music Understanding with Large-Scale Pre-Training.* (Findings of ACL)
3. Wu, Y., et al. (2023). *CLaMP: Contrastive Language-Music Pre-training.*
4. Wu, Y., et al. (2024). *CLaMP 2: Multimodal Music Information Retrieval.*
5. Fradet, N., et al. (2021). *MidiTok: A Python Package for MIDI File Tokenization.*
6. Huang, Y.-S., Yang, Y.-H. (2020). *Pop Music Transformer* (REMI).
7. Heusel, M., et al. (2017). *GANs Trained by a Two Time-Scale Update Rule* (FID).
8. Kilgour, K., et al. (2019). *Fréchet Audio Distance* (FAD).

---

## 🎓 Academic Context

| | |
|:--|:--|
| 🏫 **Institution** | Warsaw University of Technology, EITI |
| 📚 **Course** | WIMU (Music Information Retrieval) |
| 👨‍💻 **Authors** | Michał Fereniec, Bartosz Sędzikowski |
| 👨‍🏫 **Supervisor** | mgr inż. Tomasz Radzikowski |
| 📅 **Duration** | February–June 2026 |

<p align="center">
  <b>✅ Status: Results Complete</b> | 📅 Last Updated: 2026-06-09
</p>
