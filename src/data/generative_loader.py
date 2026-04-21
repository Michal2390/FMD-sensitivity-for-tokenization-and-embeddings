"""Generative model baselines for FMD evaluation.

Provides:
1. MarkovMIDIGenerator — simple first-order Markov chain trained on real MIDI
2. RandomMIDIGenerator — random note sequences (negative control)
3. SymbotunesFetcher — download FolkRNN/GPT-2 outputs from Symbotunes

These baselines let us test whether FMD/nFMD correctly ranks:
  real >> Markov >> random
across all embedding models and tokenizers.
"""

from __future__ import annotations

import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from loguru import logger

try:
    import pretty_midi
    _HAS_PRETTY_MIDI = True
except ImportError:
    _HAS_PRETTY_MIDI = False


class MarkovMIDIGenerator:
    """First-order Markov chain MIDI generator.

    Learns pitch transition probabilities from a set of real MIDI files,
    then generates new MIDI files by sampling from the learned chain.
    """

    def __init__(self, order: int = 1, seed: int = 42):
        self.order = order
        self.seed = seed
        self.transition: Dict[Tuple[int, ...], Dict[int, int]] = defaultdict(lambda: defaultdict(int))
        self.start_states: List[Tuple[int, ...]] = []
        self._fitted = False

    def fit(self, midi_files: List[Path], max_files: int = 500) -> "MarkovMIDIGenerator":
        """Learn transition probabilities from real MIDI files.

        Args:
            midi_files: List of paths to MIDI files.
            max_files: Maximum files to use for training.
        """
        if not _HAS_PRETTY_MIDI:
            raise ImportError("pretty_midi required for MarkovMIDIGenerator")

        rng = random.Random(self.seed)
        files = list(midi_files)[:max_files]
        rng.shuffle(files)

        n_processed = 0
        for path in files:
            try:
                pm = pretty_midi.PrettyMIDI(str(path))
                for instrument in pm.instruments:
                    if instrument.is_drum:
                        continue
                    pitches = [n.pitch for n in instrument.notes]
                    if len(pitches) < self.order + 1:
                        continue
                    # Record start state
                    self.start_states.append(tuple(pitches[:self.order]))
                    # Build transitions
                    for i in range(len(pitches) - self.order):
                        state = tuple(pitches[i:i + self.order])
                        next_pitch = pitches[i + self.order]
                        self.transition[state][next_pitch] += 1
                n_processed += 1
            except Exception:
                continue

        self._fitted = True
        logger.info(f"MarkovMIDIGenerator fitted on {n_processed} files, "
                    f"{len(self.transition)} unique states")
        return self

    def generate(
        self,
        n_files: int = 100,
        notes_per_file: int = 64,
        output_dir: Optional[Path] = None,
    ) -> List[Path]:
        """Generate MIDI files using the learned Markov chain.

        Args:
            n_files: Number of files to generate.
            notes_per_file: Notes per generated file.
            output_dir: Directory to save MIDI files.

        Returns:
            List of paths to generated MIDI files.
        """
        if not self._fitted:
            raise RuntimeError("Call fit() before generate()")
        if not _HAS_PRETTY_MIDI:
            raise ImportError("pretty_midi required")

        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

        rng = random.Random(self.seed + 1)
        np_rng = np.random.default_rng(self.seed + 1)
        paths = []

        for i in range(n_files):
            pm = pretty_midi.PrettyMIDI()
            instrument = pretty_midi.Instrument(program=0)

            # Pick start state
            if self.start_states:
                state = list(rng.choice(self.start_states))
            else:
                state = [60] * self.order  # fallback to middle C

            pitches = list(state)
            for _ in range(notes_per_file - self.order):
                key = tuple(pitches[-self.order:])
                if key in self.transition and self.transition[key]:
                    next_pitches = list(self.transition[key].keys())
                    weights = list(self.transition[key].values())
                    total = sum(weights)
                    probs = [w / total for w in weights]
                    next_pitch = np_rng.choice(next_pitches, p=probs)
                else:
                    # Random fallback
                    next_pitch = rng.randint(36, 84)
                pitches.append(int(next_pitch))

            # Convert pitches to MIDI notes
            t = 0.0
            for pitch in pitches:
                duration = rng.choice([0.25, 0.5, 0.5, 1.0])
                note = pretty_midi.Note(
                    velocity=80,
                    pitch=max(0, min(127, pitch)),
                    start=t,
                    end=t + duration,
                )
                instrument.notes.append(note)
                t += duration

            pm.instruments.append(instrument)

            if output_dir:
                path = output_dir / f"markov_{i:04d}.mid"
                pm.write(str(path))
                paths.append(path)

        logger.info(f"Generated {len(paths)} Markov MIDI files")
        return paths


class RandomMIDIGenerator:
    """Random MIDI generator (negative control baseline).

    Generates completely random note sequences — should have
    the highest FMD against any real music dataset.
    """

    def __init__(self, seed: int = 42):
        self.seed = seed

    def generate(
        self,
        n_files: int = 100,
        notes_per_file: int = 64,
        output_dir: Optional[Path] = None,
    ) -> List[Path]:
        """Generate random MIDI files.

        Args:
            n_files: Number of files to generate.
            notes_per_file: Notes per file.
            output_dir: Directory to save files.

        Returns:
            List of paths.
        """
        if not _HAS_PRETTY_MIDI:
            raise ImportError("pretty_midi required")

        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

        rng = np.random.default_rng(self.seed)
        paths = []

        for i in range(n_files):
            pm = pretty_midi.PrettyMIDI()
            instrument = pretty_midi.Instrument(program=0)

            t = 0.0
            for _ in range(notes_per_file):
                pitch = int(rng.integers(21, 109))
                duration = float(rng.choice([0.125, 0.25, 0.5, 1.0]))
                velocity = int(rng.integers(40, 120))
                note = pretty_midi.Note(
                    velocity=velocity,
                    pitch=pitch,
                    start=t,
                    end=t + duration,
                )
                instrument.notes.append(note)
                t += duration

            pm.instruments.append(instrument)

            if output_dir:
                path = output_dir / f"random_{i:04d}.mid"
                pm.write(str(path))
                paths.append(path)

        logger.info(f"Generated {len(paths)} random MIDI files")
        return paths

