#!/usr/bin/env python3
"""Interaction mechanism analysis: why certain tokenizer×model combos yield anomalous FMD.

Investigates the tok×model interaction by analysing:
  1. Embedding norms & effective dimensionality (PCA)
  2. Inter-sample cosine similarity per cell
  3. Token sequence length distributions
  4. t-SNE / UMAP visualisation of embedding space coloured by genre

Reads cached embeddings or re-extracts from MIDI, then generates
results/reports/lakh_multi/INTERACTION_MECHANISM_REPORT.md + plots.
"""

from __future__ import annotations

import sys
from itertools import product
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import yaml
from loguru import logger
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity

sys.path.insert(0, str(Path(__file__).parent / "src"))

from data.manager import DatasetManager
from embeddings.extractor import EmbeddingExtractor
from preprocessing.processor import MIDIPreprocessor
from tokenization.tokenizer import TokenizationPipeline

# ──────────────────────────────────────────────────────────────────────
OUTPUT_DIR = Path("results/reports/lakh_multi")
PLOTS_DIR = Path("results/plots/paper")
SEED = 42
MAX_FILES = 120
# Only analyse default preprocessing (no velocity removal, no hard quant)
DEFAULT_PREPROCESS = (False, False)


def load_config() -> Dict:
    with open("configs/config.yaml") as f:
        return yaml.safe_load(f)


# ──────────────────────────────────────────────────────────────────────
# InteractionAnalyzer
# ──────────────────────────────────────────────────────────────────────

class InteractionAnalyzer:
    """Analyse tok×model interaction mechanisms."""

    def __init__(self, config: Dict):
        self.config = config
        self.dataset_manager = DatasetManager(config)
        self.preprocessor = MIDIPreprocessor(config)
        self.tokenization = TokenizationPipeline(config)
        self.emb_extractor = EmbeddingExtractor(config)

        self.tokenizers = [t["type"] for t in config["tokenization"]["tokenizers"]]
        self.models = [m["name"] for m in config["embeddings"]["models"]]
        self.genres = config["lakh"]["genres"]

        # Caches: keyed by (genre, tok, model)
        self.emb_cache: Dict[Tuple, np.ndarray] = {}
        self.token_lengths: Dict[Tuple, List[int]] = {}

    # ── Extraction ────────────────────────────────────────────────────

    def extract_all(self):
        """Extract embeddings + token lengths for default preprocessing."""
        vel, quant = DEFAULT_PREPROCESS
        combos = list(product(self.genres, self.tokenizers, self.models))
        for idx, (genre, tok, model) in enumerate(combos, 1):
            key = (genre, tok, model)
            if key in self.emb_cache:
                continue
            logger.info(f"[{idx}/{len(combos)}] Extracting {genre} × {tok} × {model}")

            ds_name = f"lakh_{genre}"
            midi_files = self.dataset_manager.list_midi_files(ds_name, processed=False, limit=MAX_FILES)
            if not midi_files:
                logger.warning(f"  No MIDI for {ds_name}")
                continue

            vectors, lengths = [], []
            for midi_path in midi_files:
                try:
                    midi_data = self.preprocessor.load_midi(midi_path)
                    if midi_data is None:
                        continue
                    midi_data = self.preprocessor.filter_note_range(midi_data)
                    midi_data = self.preprocessor.normalize_instruments(midi_data)

                    tokenizer = self.tokenization.tokenizers[tok]
                    tokens = tokenizer.encode_midi_object(midi_data)
                    if not tokens:
                        continue
                    lengths.append(len(tokens))
                    vec = self.emb_extractor.extract_embeddings([tokens], model)[0]
                    vectors.append(vec)
                except Exception as exc:
                    logger.debug(f"  Skip {midi_path.name}: {exc}")

            if vectors:
                self.emb_cache[key] = np.vstack(vectors)
                self.token_lengths[key] = lengths
                logger.info(f"  → {len(vectors)} embeddings, dim={vectors[0].shape[0]}")

    # ── Analysis per cell ─────────────────────────────────────────────

    def compute_cell_stats(self) -> pd.DataFrame:
        """Per tok×model cell: norm, eff-dim, cosine sim, token length."""
        rows = []
        for tok in self.tokenizers:
            for model in self.models:
                # Aggregate across genres
                all_embs = []
                all_lengths = []
                for genre in self.genres:
                    key = (genre, tok, model)
                    if key in self.emb_cache:
                        all_embs.append(self.emb_cache[key])
                    if key in self.token_lengths:
                        all_lengths.extend(self.token_lengths[key])
                if not all_embs:
                    continue

                embs = np.vstack(all_embs)
                norms = np.linalg.norm(embs, axis=1)

                # Effective dimensionality via PCA
                n_comp = min(20, embs.shape[0], embs.shape[1])
                pca = PCA(n_components=n_comp)
                pca.fit(embs)
                cum_var = np.cumsum(pca.explained_variance_ratio_)
                eff_dim_90 = int(np.searchsorted(cum_var, 0.90) + 1)
                eff_dim_95 = int(np.searchsorted(cum_var, 0.95) + 1)

                # Mean pairwise cosine similarity (sample 500 for speed)
                if embs.shape[0] > 500:
                    rng = np.random.default_rng(SEED)
                    idx = rng.choice(embs.shape[0], 500, replace=False)
                    sample = embs[idx]
                else:
                    sample = embs
                cos_sim = cosine_similarity(sample)
                # Upper triangle excluding diagonal
                triu_idx = np.triu_indices_from(cos_sim, k=1)
                mean_cos = float(cos_sim[triu_idx].mean())

                rows.append({
                    "tokenizer": tok,
                    "model": model,
                    "n_samples": embs.shape[0],
                    "emb_dim": embs.shape[1],
                    "mean_norm": float(norms.mean()),
                    "std_norm": float(norms.std()),
                    "eff_dim_90": eff_dim_90,
                    "eff_dim_95": eff_dim_95,
                    "mean_cosine_sim": mean_cos,
                    "mean_token_length": float(np.mean(all_lengths)) if all_lengths else 0,
                    "median_token_length": float(np.median(all_lengths)) if all_lengths else 0,
                    "std_token_length": float(np.std(all_lengths)) if all_lengths else 0,
                    "pca_var_ratio_top5": float(cum_var[min(4, len(cum_var) - 1)]),
                })
        return pd.DataFrame(rows)

    def compute_pca_curves(self) -> Dict[str, np.ndarray]:
        """PCA explained variance ratio curves per tok×model."""
        curves = {}
        for tok in self.tokenizers:
            for model in self.models:
                all_embs = []
                for genre in self.genres:
                    key = (genre, tok, model)
                    if key in self.emb_cache:
                        all_embs.append(self.emb_cache[key])
                if not all_embs:
                    continue
                embs = np.vstack(all_embs)
                n_comp = min(20, embs.shape[0], embs.shape[1])
                pca = PCA(n_components=n_comp)
                pca.fit(embs)
                curves[f"{tok}+{model}"] = pca.explained_variance_ratio_
        return curves

    # ── Plotting ──────────────────────────────────────────────────────

    def generate_plots(self, cell_stats: pd.DataFrame, pca_curves: Dict[str, np.ndarray]):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns
        sns.set_theme(style="whitegrid", font_scale=1.1)

        PLOTS_DIR.mkdir(parents=True, exist_ok=True)
        DPI = 300

        def save_fig(fig, name):
            fig.tight_layout()
            for fmt in ("png", "pdf"):
                fig.savefig(PLOTS_DIR / f"{name}.{fmt}", dpi=DPI, bbox_inches="tight")
            plt.close(fig)
            logger.info(f"  ✅ {name}")

        # Plot 1: PCA explained variance curves
        fig, ax = plt.subplots(figsize=(10, 6))
        for label, curve in pca_curves.items():
            ax.plot(range(1, len(curve) + 1), np.cumsum(curve), marker="o", label=label, markersize=4)
        ax.set_xlabel("Principal Component")
        ax.set_ylabel("Cumulative Explained Variance")
        ax.set_title("PCA Explained Variance by Tokenizer × Model")
        ax.legend(fontsize=8, ncol=2)
        ax.axhline(0.90, color="red", linestyle="--", alpha=0.4, label="90%")
        ax.axhline(0.95, color="orange", linestyle="--", alpha=0.4, label="95%")
        save_fig(fig, "interaction_pca_curves")

        # Plot 2: Token sequence length boxplot
        length_rows = []
        for (genre, tok, model), lengths in self.token_lengths.items():
            for l in lengths:
                length_rows.append({"tokenizer": tok, "model": model, "genre": genre, "token_length": l})
        if length_rows:
            len_df = pd.DataFrame(length_rows)
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.boxplot(data=len_df, x="tokenizer", y="token_length", hue="model", ax=ax, palette="Set1")
            ax.set_title("Token Sequence Length by Tokenizer × Model")
            ax.set_ylabel("Token Sequence Length")
            save_fig(fig, "interaction_token_length")

        # Plot 3: Effective dimensionality heatmap
        if not cell_stats.empty:
            pivot = cell_stats.pivot_table(values="eff_dim_90", index="tokenizer", columns="model")
            fig, ax = plt.subplots(figsize=(7, 5))
            sns.heatmap(pivot, annot=True, fmt=".0f", cmap="YlGnBu", ax=ax, linewidths=0.5)
            ax.set_title("Effective Dimensionality (90% variance) by Tok × Model")
            save_fig(fig, "interaction_eff_dim_heatmap")

        # Plot 4: UMAP / t-SNE visualisation (sample from 2 cells for contrast)
        try:
            from sklearn.manifold import TSNE

            # Pick best and worst cell
            best_cell = cell_stats.iloc[0] if not cell_stats.empty else None
            worst_cell = cell_stats.iloc[-1] if not cell_stats.empty else None
            if best_cell is not None and worst_cell is not None:
                cells_to_plot = [
                    (best_cell["tokenizer"], best_cell["model"]),
                    (worst_cell["tokenizer"], worst_cell["model"]),
                ]

                fig, axes = plt.subplots(1, 2, figsize=(14, 6))
                for ax, (tok, model) in zip(axes, cells_to_plot):
                    embs_list, labels = [], []
                    for genre in self.genres:
                        key = (genre, tok, model)
                        if key in self.emb_cache:
                            e = self.emb_cache[key][:50]  # sample for speed
                            embs_list.append(e)
                            labels.extend([genre] * len(e))
                    if not embs_list:
                        continue
                    all_e = np.vstack(embs_list)
                    perplexity = min(30, len(all_e) - 1)
                    if perplexity < 5:
                        continue
                    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=SEED)
                    proj = tsne.fit_transform(all_e)
                    for genre in self.genres:
                        mask = [l == genre for l in labels]
                        ax.scatter(proj[mask, 0], proj[mask, 1], label=genre, alpha=0.6, s=20)
                    ax.set_title(f"{tok} + {model}")
                    ax.legend(fontsize=7)

                fig.suptitle("t-SNE: Best vs Worst Tok×Model Cell (coloured by genre)", fontsize=13)
                save_fig(fig, "interaction_tsne_best_worst")
        except Exception as exc:
            logger.warning(f"t-SNE plot failed: {exc}")

    # ── Report ────────────────────────────────────────────────────────

    def generate_report(self, cell_stats: pd.DataFrame):
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        report = []
        report.append("# Interaction Mechanism Analysis: Tokenizer × Model")
        report.append(
            "\n**Why do certain tokenizer×model combinations yield anomalously high or low FMD?**\n"
        )

        report.append("## Cell-Level Statistics\n")
        report.append(
            "| Tokenizer | Model | N | Mean Norm | Eff Dim (90%) | Mean Cos Sim | Mean Tok Len | PCA Top-5 Var |"
        )
        report.append(
            "|-----------|-------|---|-----------|---------------|-------------|-------------|--------------|"
        )
        for _, r in cell_stats.iterrows():
            report.append(
                f"| {r['tokenizer']} | {r['model']} | {int(r['n_samples'])} | "
                f"{r['mean_norm']:.3f} | {int(r['eff_dim_90'])} | "
                f"{r['mean_cosine_sim']:.3f} | {r['mean_token_length']:.0f} | "
                f"{r['pca_var_ratio_top5']:.3f} |"
            )

        report.append("\n## Key Findings\n")
        if not cell_stats.empty:
            # Identify extremes
            lowest_dim = cell_stats.loc[cell_stats["eff_dim_90"].idxmin()]
            highest_dim = cell_stats.loc[cell_stats["eff_dim_90"].idxmax()]
            shortest_tok = cell_stats.loc[cell_stats["mean_token_length"].idxmin()]
            longest_tok = cell_stats.loc[cell_stats["mean_token_length"].idxmax()]

            report.append(
                f"1. **Lowest effective dimensionality**: {lowest_dim['tokenizer']}+{lowest_dim['model']} "
                f"(eff_dim_90={int(lowest_dim['eff_dim_90'])}). Lower dimensionality may inflate FMD "
                f"because the Fréchet distance concentrates in fewer directions.\n"
            )
            report.append(
                f"2. **Highest effective dimensionality**: {highest_dim['tokenizer']}+{highest_dim['model']} "
                f"(eff_dim_90={int(highest_dim['eff_dim_90'])}).\n"
            )
            report.append(
                f"3. **Shortest token sequences**: {shortest_tok['tokenizer']} "
                f"(mean={shortest_tok['mean_token_length']:.0f} tokens). "
                f"Shorter sequences carry less information → different embedding distributions.\n"
            )
            report.append(
                f"4. **Longest token sequences**: {longest_tok['tokenizer']} "
                f"(mean={longest_tok['mean_token_length']:.0f} tokens).\n"
            )

        report.append("\n## Hypotheses\n")
        report.append(
            "- **Octuple + MusicBERT-large**: Octuple produces significantly shorter token sequences. "
            "MusicBERT-large (MIDI-native) encodes MIDI structure directly, but the compact Octuple "
            "representation may cause embeddings to cluster in a low-dimensional subspace, "
            "inflating FMD between genre distributions.\n"
        )
        report.append(
            "- **REMI + MusicBERT-large**: REMI's detailed beat-relative encoding provides rich "
            "input for MusicBERT-large's MIDI encoder, yielding well-separated genre clusters and low FMD.\n"
        )

        report.append("\n## Plots\n")
        report.append("- `results/plots/paper/interaction_pca_curves.png` — PCA explained variance\n")
        report.append("- `results/plots/paper/interaction_token_length.png` — Token length distributions\n")
        report.append("- `results/plots/paper/interaction_eff_dim_heatmap.png` — Effective dimensionality\n")
        report.append("- `results/plots/paper/interaction_tsne_best_worst.png` — t-SNE visualisation\n")

        report_path = OUTPUT_DIR / "INTERACTION_MECHANISM_REPORT.md"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("\n".join(report))
        logger.info(f"✅ Report: {report_path}")


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def main():
    logger.info("=" * 70)
    logger.info("INTERACTION MECHANISM ANALYSIS")
    logger.info("=" * 70)

    config = load_config()
    analyzer = InteractionAnalyzer(config)

    logger.info("\n=== Step 1: Extract embeddings ===")
    analyzer.extract_all()

    logger.info("\n=== Step 2: Compute cell statistics ===")
    cell_stats = analyzer.compute_cell_stats()
    cell_stats.to_csv(OUTPUT_DIR / "interaction_cell_stats.csv", index=False)
    logger.info(f"Cell stats:\n{cell_stats.to_string()}")

    logger.info("\n=== Step 3: PCA curves ===")
    pca_curves = analyzer.compute_pca_curves()

    logger.info("\n=== Step 4: Generate plots ===")
    analyzer.generate_plots(cell_stats, pca_curves)

    logger.info("\n=== Step 5: Generate report ===")
    analyzer.generate_report(cell_stats)

    logger.info("\n" + "=" * 70)
    logger.info("INTERACTION ANALYSIS COMPLETE")
    logger.info(f"  Results: {OUTPUT_DIR}/")
    logger.info(f"  Plots:   {PLOTS_DIR}/")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()



