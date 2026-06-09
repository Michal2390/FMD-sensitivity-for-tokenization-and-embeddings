# FMD Sensitivity Study - Methodology Status

This document replaces the previous sensitivity-pivot result note. The older
tables included a `CLaMP2-REMI` configuration, which is not a valid main-study
configuration: CLaMP-2 does not consume MidiTok REMI tokens as its native music
input. Those numbers must not be used in the paper.

## Paper Title

**FMD Sensitivity to Tokenization and Embedding Configuration**

## Valid Configuration Families

| Family | Valid examples | Scientific interpretation |
|--------|----------------|---------------------------|
| MidiTok tokenizer sensitivity | `MusicBERT-REMI`, `MusicBERT-TSD`, `MusicBERT-Octuple`, `MusicBERT-MIDI-Like` | Same embedding model, different tokenization schemes. |
| Model-native representation sensitivity | `CLaMP2-MTF`, `CLaMP1-ABC`, optionally `MERT-audio` | Different model/input representation pipelines. |

Invalid for main paper results:

- `CLaMP2-REMI`
- `CLaMP1-REMI`
- Any result that zero-pads or SVD-aligns embedding spaces before FMD
- Any result based on synthetic fallback embeddings
- Any claim that nFMD makes FMD fairly comparable across embedding models

## Current Pivot Configurations

- **MusicBERT-REMI** - MusicBERT over MidiTok REMI tokens.
- **MusicBERT-TSD** - MusicBERT over MidiTok TSD tokens.
- **CLaMP2-MTF** - CLaMP-2 native MTF/M3 representation from MIDI.
- **CLaMP1-ABC** - CLaMP-1 ABC notation via `music21`.

## Statistical Design

The paper must not rely on significance tests over three configuration values.
The defended analysis should use repeated observations from:

- multiple dataset pairs,
- bootstrap resampling over pieces,
- perturbation profiles per configuration,
- optional mixed-effects models with dataset pair as a random effect.

Recommended primary outputs:

- raw FMD per dataset pair and configuration,
- bootstrap confidence intervals,
- rank agreement across dataset pairs,
- perturbation sensitivity heatmaps,
- clear separation between tokenizer effects and model/representation effects.

## Interpretation Rules

- Raw FMD is valid within a single embedding space.
- Cross-configuration comparisons are interpreted as ranking/sensitivity changes,
  not as universally calibrated distances.
- ABC does not preserve velocity; low velocity sensitivity for `CLaMP1-ABC` is an
  expected representation property, not evidence that velocity is irrelevant.
- MTF preserves more MIDI message information and should be evaluated as a
  model-native representation, not as a tokenizer.

## Regeneration

Regenerate results only after the strict paper pipeline passes without synthetic
fallbacks or invalid configuration warnings:

```bash
python main.py --mode sensitivity
```
