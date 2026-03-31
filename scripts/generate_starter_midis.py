"""Generate starter MIDI corpora for local experimentation.

Creates small synthetic sets in:
- data/raw/maestro
- data/raw/midicaps
- data/raw/pop909
"""

from pathlib import Path
import random

import pretty_midi


def make_piece(style: str, idx: int, out_path: Path) -> None:
    midi = pretty_midi.PrettyMIDI(initial_tempo=120)
    instrument = pretty_midi.Instrument(program=0, is_drum=False)

    random.seed(1000 + idx)

    if style == "maestro":
        # Classical-like arpeggios and wider pitch span.
        base = random.choice([48, 50, 52, 53])
        pitches = [base, base + 4, base + 7, base + 12, base + 7, base + 4]
        duration = 0.35
    elif style == "pop909":
        # Pop-like repetitive hooks with tighter range.
        base = random.choice([60, 62, 64])
        pitches = [base, base + 2, base + 4, base + 2, base, base + 4, base + 2, base]
        duration = 0.25
    else:  # midicaps
        # Caption-like mixed patterns / more variety.
        base = random.choice([55, 57, 59, 60])
        pitches = [base, base + 3, base + 7, base + 10, base + 5, base + 8]
        duration = 0.3

    t = 0.0
    for p in pitches * 6:
        vel = random.randint(60, 108)
        note = pretty_midi.Note(velocity=vel, pitch=max(21, min(108, p)), start=t, end=t + duration)
        instrument.notes.append(note)
        t += duration

    midi.instruments.append(instrument)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    midi.write(str(out_path))


def main() -> None:
    root = Path("data/raw")
    spec = {
        "maestro": 10,
        "midicaps": 10,
        "pop909": 10,
    }

    for dataset, count in spec.items():
        target = root / dataset
        target.mkdir(parents=True, exist_ok=True)
        for idx in range(count):
            out_file = target / f"{dataset}_{idx:02d}.mid"
            make_piece(dataset, idx, out_file)

    print("Generated starter MIDI files in data/raw/*")


if __name__ == "__main__":
    main()

