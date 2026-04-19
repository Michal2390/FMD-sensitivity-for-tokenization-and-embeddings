# Interaction Mechanism Analysis: Tokenizer × Model

**Why do certain tokenizer×model combinations yield anomalously high or low FMD?**

## Cell-Level Statistics

| Tokenizer | Model | N | Mean Norm | Eff Dim (90%) | Mean Cos Sim | Mean Tok Len | PCA Top-5 Var |
|-----------|-------|---|-----------|---------------|-------------|-------------|--------------|
| REMI | CLaMP-1 | 475 | 7.784 | 19 | 0.980 | 21117 | 0.671 |
| REMI | CLaMP-2 | 475 | 2.757 | 21 | 0.964 | 21117 | 0.613 |
| REMI | MusicBERT | 475 | 15.855 | 21 | 0.964 | 21117 | 0.542 |
| TSD | CLaMP-1 | 475 | 7.846 | 20 | 0.980 | 20910 | 0.677 |
| TSD | CLaMP-2 | 475 | 2.636 | 21 | 0.941 | 20910 | 0.523 |
| TSD | MusicBERT | 475 | 15.266 | 21 | 0.960 | 20910 | 0.553 |
| Octuple | CLaMP-1 | 473 | 7.761 | 15 | 0.984 | 21120 | 0.733 |
| Octuple | CLaMP-2 | 473 | 2.675 | 21 | 0.843 | 21120 | 0.516 |
| Octuple | MusicBERT | 473 | 13.814 | 21 | 0.971 | 21120 | 0.589 |
| MIDI-Like | CLaMP-1 | 475 | 7.404 | 16 | 0.981 | 26875 | 0.724 |
| MIDI-Like | CLaMP-2 | 475 | 2.773 | 21 | 0.890 | 26875 | 0.687 |
| MIDI-Like | MusicBERT | 475 | 14.394 | 11 | 0.950 | 26875 | 0.835 |

## Key Findings

1. **Lowest effective dimensionality**: MIDI-Like+MusicBERT (eff_dim_90=11). Lower dimensionality may inflate FMD because the Fréchet distance concentrates in fewer directions.

2. **Highest effective dimensionality**: REMI+CLaMP-2 (eff_dim_90=21).

3. **Shortest token sequences**: TSD (mean=20910 tokens). Shorter sequences carry less information → different embedding distributions.

4. **Longest token sequences**: MIDI-Like (mean=26875 tokens).


## Hypotheses

- **Octuple + CLaMP-2**: Octuple produces significantly shorter token sequences. CLaMP-2 (MIDI-native) encodes MIDI structure directly, but the compact Octuple representation may cause embeddings to cluster in a low-dimensional subspace, inflating FMD between genre distributions.

- **REMI + CLaMP-2**: REMI's detailed beat-relative encoding provides rich input for CLaMP-2's MIDI encoder, yielding well-separated genre clusters and low FMD.


## Plots

- `results/plots/paper/interaction_pca_curves.png` — PCA explained variance

- `results/plots/paper/interaction_token_length.png` — Token length distributions

- `results/plots/paper/interaction_eff_dim_heatmap.png` — Effective dimensionality

- `results/plots/paper/interaction_tsne_best_worst.png` — t-SNE visualisation
