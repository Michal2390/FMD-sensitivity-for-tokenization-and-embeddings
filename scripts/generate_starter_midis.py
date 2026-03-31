"""Generate large, style-diverse synthetic MIDI corpora for experiments.

Data source note:
- Files are generated procedurally in this script (not downloaded from external copyrighted sets).
"""

from __future__ import annotations

import argparse
import math
import random
from pathlib import Path

import pretty_midi


STYLE_CONFIG = {
    "classical": {
        "program": 0,
        "tempo": (84, 126),
        "scales": [[0, 2, 4, 5, 7, 9, 11], [0, 2, 3, 5, 7, 8, 10]],
        "rhythm": [1.0, 0.5, 0.5, 1.0, 1.0],
        "velocity": (48, 96),
        "octaves": [4, 5],
    },
    "jazz": {
        "program": 0,
        "tempo": (96, 154),
        "scales": [[0, 2, 3, 5, 7, 9, 10], [0, 1, 3, 5, 6, 8, 10]],
        "rhythm": [0.66, 0.34, 0.5, 0.5, 1.0],
        "velocity": (55, 108),
        "octaves": [3, 4, 5],
    },
    "rock": {
        "program": 29,
        "tempo": (104, 176),
        "scales": [[0, 2, 4, 5, 7, 9, 10]],
        "rhythm": [0.5, 0.5, 0.5, 0.5, 1.0],
        "velocity": (72, 122),
        "octaves": [2, 3, 4],
    },
    "rap": {
        "program": 38,
        "tempo": (74, 126),
        "scales": [[0, 2, 3, 5, 7, 8, 10]],
        "rhythm": [0.25, 0.25, 0.5, 0.5, 0.5, 1.0],
        "velocity": (68, 124),
        "octaves": [2, 3],
    },
    "pop": {
        "program": 0,
        "tempo": (100, 140),
        "scales": [[0, 2, 4, 5, 7, 9, 11], [0, 2, 4, 5, 7, 9, 10]],
        "rhythm": [0.5, 0.5, 1.0, 1.0],
        "velocity": (60, 112),
        "octaves": [3, 4, 5],
    },
}


def _bounded_pitch(value: int) -> int:
    return max(21, min(108, value))


def _create_track(style: str, idx: int, bars: int, rng: random.Random) -> pretty_midi.PrettyMIDI:
    cfg = STYLE_CONFIG[style]
    tempo = rng.randint(cfg["tempo"][0], cfg["tempo"][1])
    midi = pretty_midi.PrettyMIDI(initial_tempo=tempo)

    melody = pretty_midi.Instrument(program=cfg["program"], is_drum=False)
    chords = pretty_midi.Instrument(program=0, is_drum=False)
    drums = pretty_midi.Instrument(program=0, is_drum=True)

    scale = rng.choice(cfg["scales"])
    base_pc = rng.choice([0, 2, 4, 5, 7, 9])
    rhythm = cfg["rhythm"]
    min_vel, max_vel = cfg["velocity"]

    beat = 60.0 / float(tempo)
    t = 0.0
    total_beats = bars * 4

    # Melody with style-specific rhythmic jitter and contour.
    while t < total_beats * beat:
        dur_beats = rng.choice(rhythm)
        dur = dur_beats * beat
        octave = rng.choice(cfg["octaves"])
        degree = rng.choice(scale)
        contour = int(2.0 * math.sin((idx + t) * 0.8))
        pitch = _bounded_pitch(12 * octave + base_pc + degree + contour)
        vel = rng.randint(min_vel, max_vel)
        melody.notes.append(pretty_midi.Note(velocity=vel, pitch=pitch, start=t, end=t + dur))
        t += dur

    # Simple harmonic bed, one chord per bar.
    t = 0.0
    for _ in range(bars):
        root = _bounded_pitch(48 + base_pc + rng.choice([0, 2, 5, 7]))
        triad = [root, _bounded_pitch(root + 4), _bounded_pitch(root + 7)]
        for p in triad:
            chords.notes.append(
                pretty_midi.Note(velocity=rng.randint(45, 84), pitch=p, start=t, end=t + 4 * beat)
            )
        t += 4 * beat

    # Lightweight drum groove to add rhythmic diversity.
    t = 0.0
    while t < total_beats * beat:
        if rng.random() < 0.85:
            drums.notes.append(pretty_midi.Note(velocity=rng.randint(45, 96), pitch=36, start=t, end=t + 0.1))
        if rng.random() < 0.55:
            drums.notes.append(pretty_midi.Note(velocity=rng.randint(35, 92), pitch=38, start=t + 0.5 * beat, end=t + 0.6 * beat))
        if rng.random() < 0.65:
            drums.notes.append(pretty_midi.Note(velocity=rng.randint(30, 88), pitch=42, start=t, end=t + 0.1))
        t += beat

    midi.instruments.extend([melody, chords, drums])
    return midi


def generate_dataset(root: Path, dataset: str, style: str, count: int, bars: int, seed: int) -> int:
    target = root / dataset
    target.mkdir(parents=True, exist_ok=True)
    generated = 0

    for idx in range(count):
        rng = random.Random(seed + idx + abs(hash((dataset, style))) % 100_000)
        midi = _create_track(style=style, idx=idx, bars=bars, rng=rng)
        out_file = target / f"{dataset}_{idx:04d}.mid"
        midi.write(str(out_file))
        generated += 1

    return generated


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate diverse synthetic MIDI corpora")
    parser.add_argument("--count-per-dataset", type=int, default=120, help="Number of MIDI files per dataset")
    parser.add_argument("--bars", type=int, default=16, help="Bars per generated piece")
    parser.add_argument("--seed", type=int, default=20260331, help="Global generation seed")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path("data/raw")
    style_for_dataset = {
        "maestro": "classical",
        "midicaps": "jazz",
        "pop909": "pop",
        "jazz": "jazz",
        "rock": "rock",
        "classical": "classical",
        "pop": "pop",
        "rap": "rap",
    }

    generated_total = 0
    for dataset, style in style_for_dataset.items():
        generated_total += generate_dataset(
            root=root,
            dataset=dataset,
            style=style,
            count=args.count_per_dataset,
            bars=args.bars,
            seed=args.seed,
        )

    print(
        f"Generated {generated_total} synthetic MIDI files across {len(style_for_dataset)} datasets in data/raw/*"
    )


if __name__ == "__main__":
    main()



