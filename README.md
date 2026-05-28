# 🎵 FMD Sensitivity to Tokenization and Embedding Configuration

<p align="center">
  <b>🔬 Sensitivity Profiling of Fréchet Music Distance for Symbolic Music Evaluation</b>
</p>

<p align="center">
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python"></a>
  <a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch"></a>
  <a href="#"><img src="https://img.shields.io/badge/Status-Results_Complete-success?style=for-the-badge" alt="Status"></a>
  <a href="#"><img src="https://img.shields.io/badge/License-MIT-blue?style=for-the-badge" alt="License"></a>
</p>

> 📖 An empirical study revealing how pipeline configuration choices (embedding model, tokenization, preprocessing) alter what FMD actually measures - with practical recommendations for the music generation community.

---

## ⚡ TL;DR

We profiled how 3 FMD configurations react to controlled perturbations of MIDI data. **The key discovery:**

| Perturbation | CLaMP2-ABC | CLaMP2-MTF | CLaMP1-ABC |
|:-------------|:----------:|:----------:|:----------:|
| 🔴 Remove velocity (dynamics) | **0.510** | **0.408** | 0.022 |
| 🟡 Quantize timing (16th grid) | 0.008 | 0.008 | 0.007 |
| 🟢 Constant tempo (120 BPM) | 0.000 | 0.000 | 0.000 |

> 🎯 **CLaMP-2 sees velocity; CLaMP-1 is blind to it. No model sees tempo or microtiming. Configuration choice determines what musical aspects FMD actually evaluates.**

---

## 🔄 Motivation and Research Pivot

### ❌ Original Approach: Normalized FMD (nFMD) - Why It Failed

Our initial contribution was **Normalized FMD (nFMD)** - an attempt to make FMD values comparable across different embedding models by normalizing for embedding scale:

- `nFMD_trace = FMD / (Tr(Σ₁) + Tr(Σ₂))` - normalize by total variance
- `nFMD_norm = FMD / (‖μ₁‖ + ‖μ₂‖)²` - compensate quadratic mean scaling

We observed that raw FMD values differ by **12.8×** across models (e.g., MusicBERT FMD ≈ 8.0 vs CLaMP-2 FMD ≈ 0.6 for the same genre pair) simply due to embedding norm differences.

**🚫 Why nFMD is fundamentally flawed:**

| # | Argument | Explanation |
|:-:|:---------|:------------|
| 1️⃣ | **The problem doesn't exist meaningfully** | Each model lives in a different feature space. Forcing common scale ≠ measuring the same thing |
| 2️⃣ | **FMD/FID/FAD were never designed for cross-model comparison** | Fréchet distance works *within* a fixed space. Cross-model = comparing meters with kilograms |
| 3️⃣ | **Normalization obscures rather than reveals** | Apparent "hidden effects" (η² 0.001→0.014) are likely division artefacts |
| 4️⃣ | **Precedent agrees** | FID and FAD have never been normalized in the literature |

We implemented nFMD experimentally (5350 obs, 6 models, full ANOVA), but **rejected it** because it produces misleading results.

### ✅ The Pivot: Sensitivity Profiling

Instead of cross-model normalization, we asked:

> 💡 **"Given a fixed FMD configuration, what musical properties does the metric actually measure?"**

This produces **directly actionable knowledge:**
- 🎹 Evaluating expressive dynamics? → Use **CLaMP-2**
- 🎼 Evaluating harmonic/melodic structure? → Use **CLaMP-1**
- ⏱️ Evaluating timing quality? → **None of these configs can do that**

---

## 🧪 Experimental Design

### 🔧 3 Configurations (isolating variables)

| Config | Model | Tokenization | Isolates |
|:------:|:-----:|:-------------|:---------|
| 🅰️ **CLaMP2-ABC** | CLaMP-2 | REMI (text-like) | Baseline |
| 🅱️ **CLaMP2-MTF** | CLaMP-2 | MIDI-Like (full fidelity) | 🔀 **Tokenization effect** |
| 🅲 **CLaMP1-ABC** | CLaMP-1 | REMI (text-like) | 🧠 **Model effect** |

### 🎶 3 Datasets (stylistically distinct)

| Dataset | Style | N files | Source |
|:--------|:------|:-------:|:-------|
| 🎹 **MAESTRO** | Classical piano, virtuosic | 1,276 | Google Magenta v3 |
| 🎤 **POP909** | Pop songs | 2,898 | Music-X-Lab |
| 🪕 **Folk** | Traditional folk tunes | 1,034 | Nottingham Dataset |

### 🎛️ 5 Perturbations (controlled expression removal)

| Perturbation | What it removes | Implementation |
|:-------------|:----------------|:---------------|
| ✨ `original` | Nothing (baseline) | - |
| 🔇 `no_velocity` | Dynamics/expression | All notes → velocity 64 |
| 📐 `quantized_time` | Microtiming/swing | Snap to 16th-note grid |
| ⏱️ `constant_tempo` | Rubato/tempo variation | Remap beats → 120 BPM |
| 💀 `all_combined` | All expression | All three combined |

---

## 📊 Results

### ✅ Step 3: Self-Similarity Sanity Check

Split-half FMD (should be ≈ 0 if stable):

| Dataset | CLaMP2-ABC | CLaMP2-MTF | CLaMP1-ABC |
|:--------|:----------:|:----------:|:----------:|
| 🎹 MAESTRO | 0.039 | 0.027 | 0.022 |
| 🎤 POP909 | 0.070 | 0.056 | 0.038 |
| 🪕 Folk | 0.039 | 0.019 | 0.059 |

> ✅ All values near zero (0.019–0.070). **Noise floor established.** Any FMD < 0.07 = sampling noise.

---

### 📈 Step 4: Cross-Dataset Ranking

| Pair | CLaMP2-ABC | CLaMP2-MTF | CLaMP1-ABC |
|:-----|:----------:|:----------:|:----------:|
| MAESTRO ↔ POP909 | 0.265 | 0.170 | 0.145 |
| MAESTRO ↔ Folk | 0.572 | 0.477 | 0.294 |
| POP909 ↔ Folk | **0.814** | 0.310 | 0.213 |

**Spearman τ (ranking agreement):**

| Pair | τ | Interpretation |
|:-----|:-:|:---------------|
| CLaMP2-ABC vs CLaMP2-MTF | 0.50 | ⚠️ Weak agreement |
| CLaMP2-ABC vs CLaMP1-ABC | 0.50 | ⚠️ Weak agreement |
| CLaMP2-MTF vs CLaMP1-ABC | **1.00** | ✅ **Perfect agreement** |

> 🔑 **Key Finding:** Tokenization choice (REMI vs MIDI-Like) can **invert** which datasets FMD considers most similar/different!

---

### 🔬 Step 5: Perturbation Sensitivity ⭐ MAIN RESULT

FMD(original, perturbed) on MAESTRO. Higher = more sensitive:

| Perturbation | CLaMP2-ABC | CLaMP2-MTF | CLaMP1-ABC | Verdict |
|:-------------|:----------:|:----------:|:----------:|:--------|
| 🔇 **no_velocity** | **0.510** | **0.408** | 0.022 | 🔴 CLaMP-2 sees it; CLaMP-1 blind |
| 📐 quantized_time | 0.008 | 0.008 | 0.007 | ⚪ Nothing sees it |
| ⏱️ constant_tempo | 0.000 | 0.000 | 0.000 | ⚪ Nothing sees it |
| 💀 all_combined | 0.514 | 0.408 | 0.026 | = velocity alone |

#### 🔍 Analysis:

| Finding | Details |
|:--------|:--------|
| 🔴 **Velocity dominates** | CLaMP-2: FMD = 0.4–0.5 (7–25× noise floor). CLaMP-1: FMD = 0.02 (at noise floor - blind!) |
| 🟡 **Timing invisible** | FMD ≈ 0.008 for all configs. 16th-note quantization is undetectable |
| 🟢 **Tempo invisible** | FMD = 0.000. Rubato vs metronomic = identical embeddings |
| ⚫ **Combined = velocity** | all_combined ≈ no_velocity. One-dimensional sensitivity |

> 💡 **Key insight:** If you claim "lower FMD = more expressive music" - that only holds for velocity with CLaMP-2. Tempo and timing are completely invisible to all tested configurations.

---

### 📉 Step 6: Bootstrap Stability

10× bootstrap (MAESTRO vs POP909, N=200):

| Config | FMD mean ± std | 95% CI | CV |
|:-------|:--------------:|:------:|:--:|
| CLaMP2-ABC | 0.286 ± 0.025 | [0.253, 0.329] | 8.9% |
| CLaMP2-MTF | 0.184 ± 0.015 | [0.159, 0.211] | 8.3% |
| CLaMP1-ABC | 0.157 ± 0.013 | [0.140, 0.180] | 8.6% |

> 📊 All configs equally stable (CV ≈ 8–9%). Non-overlapping CIs confirm robust differences.

---

## 🏆 Conclusions for Publication

### 📝 Main Contributions

| # | Contribution | Key Number |
|:-:|:-------------|:-----------|
| 1️⃣ | **Sensitivity profiling methodology** - perturbation-based framework | 3 configs × 5 perturbations |
| 2️⃣ | **Only velocity is detected** by FMD (not tempo, not timing) | FMD = 0.0 for tempo |
| 3️⃣ | **Model determines sensitivity** - CLaMP-2 sees velocity; CLaMP-1 doesn't | 0.51 vs 0.02 |
| 4️⃣ | **Tokenization affects ranking** - can invert dataset distance ordering | τ = 0.5 |
| 5️⃣ | **Rejection of nFMD** - negative result preventing dead-end research | η² artefact |

### 🎯 Practical Recommendations

| Goal | ✅ Use | ❌ Avoid | Why |
|:-----|:-------|:---------|:----|
| 🎹 Dynamics/expression | CLaMP-2 | CLaMP-1 | FMD = 0.4–0.5 vs 0.02 |
| 🎼 Harmonic structure | CLaMP-1 | - | Velocity-invariant |
| 📊 Consistent ranking | CLaMP-2 MTF / CLaMP-1 | CLaMP-2 ABC | τ = 1.0 vs 0.5 |
| ⏱️ Tempo/timing eval | - | All configs | FMD = 0.0 always |

---

## 🚀 Running the Experiments

### Full pipeline (~27 min, CPU):
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

### 📦 Dataset preparation:
```bash
python scripts/download_folk_dataset.py          # 🪕 Nottingham folk
python main.py --mode fetch-data --datasets maestro pop909
```

### 📁 Output:
```
results/reports/sensitivity_pivot/
├── 📄 self_similarity.csv
├── 📄 cross_dataset_fmd.csv
├── 📄 spearman_ranking_agreement.csv
├── 📄 perturbation_sensitivity.csv
├── 📄 bootstrap_stability.csv
└── 📄 sensitivity_pivot_summary.json

results/plots/sensitivity_pivot/
├── 🖼️ perturbation_heatmap.png
├── 🖼️ cross_dataset_bar.png
├── 🖼️ bootstrap_stability.png
└── 🖼️ self_similarity.png
```

---

## 🗂️ Project Structure

```
📦 src/
  ├── 🔧 preprocessing/processor.py     # MIDI preprocessing + normalize_tempo()
  ├── 🎹 tokenization/tokenizer.py      # REMI, TSD, Octuple, MIDI-Like
  ├── 🧠 embeddings/extractor.py        # CLaMP-1/2, MusicBERT, MERT, NLP
  ├── 📐 metrics/fmd.py                 # Fréchet Music Distance
  └── 🔬 experiments/
      ├── sensitivity_profiler.py       # ← Main pivot pipeline
      └── paper_pipeline.py             # Legacy multi-model analysis

📦 configs/
  ├── config.yaml                       # Main project config
  └── sensitivity_pivot.yaml            # Pivot experiment config

📦 results/
  ├── reports/sensitivity_pivot/        # CSV + JSON results
  └── plots/sensitivity_pivot/          # Publication figures
```

---

## 📚 Previous Work

<details>
<summary><b>📂 nFMD & Multi-Model ANOVA (Weeks 1–11)</b></summary>

Our earlier 6-model ANOVA analysis (5760 observations, Lakh MIDI) showed:
- 🏆 Model choice explains **96%** of FMD variance (η² = 0.962)
- 📉 Tokenizer effect is negligible when pooled (η² = 0.001)
- ✅ Cross-dataset Spearman ρ = 0.975 (Lakh vs MidiCaps)

These results motivated the pivot: since model choice so thoroughly dominates, comparing raw FMD across models is meaningless.

Legacy modes: `python main.py --mode paper` | `--mode lakh` | `--mode cross-validate`

</details>

---

## 📖 References

1. Retkowski, J., Stępniak, J., Modrzejewski, M. (2025). *Fréchet Music Distance: A Metric for Generative Symbolic Music Evaluation.*
2. Wu, Y., et al. (2023). *CLaMP: Contrastive Language-Music Pre-training.*
3. Wu, Y., et al. (2024). *CLaMP 2: Multimodal Music Information Retrieval.*
4. Fradet, N., et al. (2024). *MidiTok: A Python Package for MIDI File Tokenization.*
5. Heusel, M., et al. (2017). *GANs Trained by a Two Time-Scale Update Rule.* (FID)
6. Kilgour, K., et al. (2019). *Fréchet Audio Distance.* (FAD)

---

## 🎓 Academic Context

| | |
|:--|:--|
| 🏫 **Institution** | Warsaw University of Technology, EITI |
| 📚 **Course** | WIMU (Music Information Retrieval) |
| 👨‍💻 **Authors** | Michał Fereniec, Bartłomiej Sędłak |
| 👨‍🏫 **Supervisor** | mgr inż. Tomasz Radzikowski |
| 📅 **Duration** | February–June 2026 |
| 🎤 **Presentation** | May 28, 2026 |

---

<p align="center">
  <b>✅ Status: Results Complete</b> | 📅 Last Updated: 2026-05-28
</p>
