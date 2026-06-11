"""CLaMP input format conversion (MTF, ABC, REMI token text).

Implements M3 bar-patching as used by sander-wood/clamp2 (M3Patchilizer).
"""

from __future__ import annotations

import os
import re
import tempfile
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F

PATCH_SIZE_CLAMP2 = 64
PATCH_SIZE_CLAMP1 = 49  # 6272 / 128
PATCH_LENGTH = 512
MAX_PATCHES = 512


class M3Patchilizer:
    """Bar-patcher from CLaMP-2 / M3 (ported from sander-wood/clamp2)."""

    def __init__(self):
        self.delimiters = ["|:", "::", ":|", "[|", "||", "|]", "|"]
        self.regex_pattern = "(" + "|".join(map(re.escape, self.delimiters)) + ")"
        self.pad_token_id = 0
        self.bos_token_id = 1
        self.eos_token_id = 2
        self.mask_token_id = 3

    def split_bars(self, body: str) -> List[str]:
        bars = re.split(self.regex_pattern, "".join(body))
        bars = list(filter(None, bars))
        if bars and bars[0] in self.delimiters:
            if len(bars) > 1:
                bars[1] = bars[0] + bars[1]
            bars = bars[1:]
        if len(bars) < 2:
            return bars
        return [bars[i * 2] + bars[i * 2 + 1] for i in range(len(bars) // 2)]

    def bar2patch(self, bar: str, patch_size: int) -> List[int]:
        patch = [self.bos_token_id] + [ord(c) for c in bar] + [self.eos_token_id]
        patch = patch[:patch_size]
        patch += [self.pad_token_id] * (patch_size - len(patch))
        return patch

    def encode(
        self,
        item: str,
        patch_size: int = PATCH_SIZE_CLAMP2,
        add_special_patches: bool = False,
        truncate: bool = True,
    ) -> List[List[int]]:
        from unidecode import unidecode

        item = unidecode(item)
        lines = re.findall(r".*?\n|.*$", item)
        lines = list(filter(None, lines))
        patches: List[str] = []

        if lines and lines[0].split(" ")[0] == "ticks_per_beat":
            patch = ""
            for line in lines:
                msg_type = line.split(" ")[0]
                payload = " ".join(line.split(" ")[1:])
                if (
                    patch.startswith(msg_type)
                    and len(patch) + len(payload) <= patch_size - 2
                ):
                    patch = patch[:-1] + "\t" + payload
                else:
                    if patch:
                        patches.append(patch)
                    patch = line
            if patch:
                patches.append(patch)
        else:
            for line in lines:
                if len(line) > 1 and ((line[0].isalpha() and line[1] == ":") or line.startswith("%%")):
                    patches.append(line)
                else:
                    bars = self.split_bars(line)
                    if bars:
                        bars[-1] += "\n"
                    patches.extend(bars)

        if add_special_patches:
            bos_patch = chr(self.bos_token_id) * patch_size
            eos_patch = chr(self.eos_token_id) * patch_size
            patches = [bos_patch] + patches + [eos_patch]

        if len(patches) > PATCH_LENGTH and truncate:
            patches = patches[:PATCH_LENGTH]

        if not patches:
            patches = [""]

        return [self.bar2patch(p, patch_size) for p in patches]


def _msg_to_str(msg) -> str:
    parts = [msg.type] + [str(v) for k, v in msg.dict().items()]
    return " ".join(parts)


def pretty_midi_to_mtf_text(midi_data, m3_compatible: bool = True) -> str:
    """Convert PrettyMIDI to CLaMP MTF text (lossless MIDI message format)."""
    import mido

    with tempfile.NamedTemporaryFile(suffix=".mid", delete=False) as tmp:
        path = tmp.name
        midi_data.write(path)

    try:
        mid = mido.MidiFile(path)
        msg_list = [f"ticks_per_beat {mid.ticks_per_beat}"]
        merged = getattr(mid, "merged_track", None)
        if merged is None:
            merged = mido.merge_tracks(mid.tracks)
        for msg in merged:
            if m3_compatible and msg.is_meta:
                if msg.type in {
                    "text",
                    "copyright",
                    "track_name",
                    "instrument_name",
                    "lyrics",
                    "marker",
                    "cue_marker",
                    "device_name",
                }:
                    continue
            msg_list.append(_msg_to_str(msg))
        return "\n".join(msg_list)
    finally:
        os.unlink(path)


_ABC_PITCH_CLASS = ["C", "^C", "D", "^D", "E", "F", "^F", "G", "^G", "A", "^A", "B"]


def _abc_pitch(midi_pitch: int) -> str:
    """MIDI pitch -> ABC pitch token (middle C, MIDI 60, is 'C')."""
    octave = midi_pitch // 12 - 1
    name = _ABC_PITCH_CLASS[midi_pitch % 12]
    if octave >= 5:
        return name.lower() + "'" * (octave - 5)
    return name + "," * (4 - octave)


def pretty_midi_to_abc_text(midi_data, max_groups: int = 4096) -> str:
    """Render PrettyMIDI as ABC notation text (deterministic, dependency-free).

    music21 cannot export ABC (its ``write('abc')`` silently emits the
    object repr), so ABC is rendered directly: a standard header
    (X/M/L/Q/K) followed by notes quantized to a sixteenth-note grid
    (L:1/16), simultaneous onsets grouped as ABC chords, rests for gaps,
    and bar lines derived from the meter. By construction the rendering
    carries pitch, rhythm, meter and the global tempo marking -- and, like
    the ABC standard itself, no velocity channel. The output is a pure
    function of the note content, so re-encoding the same file yields the
    identical string.
    """
    if midi_data.time_signature_changes:
        ts = midi_data.time_signature_changes[0]
        num, den = ts.numerator, ts.denominator
    else:
        num, den = 4, 4

    tempos = midi_data.get_tempo_changes()
    tempo = float(tempos[1][0]) if len(tempos[1]) > 0 else 120.0
    sixteenth = 60.0 / tempo / 4.0  # seconds per L:1/16 unit

    header = f"X:1\nM:{num}/{den}\nL:1/16\nQ:{int(round(tempo))}\nK:C\n"

    notes = [
        (note.start, note.pitch, note.end)
        for inst in midi_data.instruments
        if not inst.is_drum
        for note in inst.notes
    ]
    if not notes:
        return header + "z16 |\n"

    # Quantize onsets/durations to the sixteenth grid and group chords.
    groups: Dict[int, List[int]] = {}
    durations: Dict[int, int] = {}
    for start, pitch, end in sorted(notes):
        onset = int(round(start / sixteenth))
        dur = max(1, int(round((end - start) / sixteenth)))
        groups.setdefault(onset, []).append(pitch)
        durations[onset] = max(durations.get(onset, 1), dur)

    units_per_bar = max(1, num * 16 // den)
    body_parts: List[str] = []
    prev_end = 0
    prev_bar = 0
    for onset in sorted(groups)[:max_groups]:
        gap = onset - prev_end
        if gap > 0:
            body_parts.append(f"z{gap}" if gap > 1 else "z")
        bar = onset // units_per_bar
        if bar > prev_bar:
            body_parts.append("|")
            if bar % 4 == 0:
                body_parts.append("\n")
            prev_bar = bar
        pitches = sorted(set(groups[onset]))
        dur = durations[onset]
        dur_s = "" if dur == 1 else str(dur)
        if len(pitches) == 1:
            body_parts.append(_abc_pitch(pitches[0]) + dur_s)
        else:
            body_parts.append("[" + "".join(_abc_pitch(p) for p in pitches) + "]" + dur_s)
        prev_end = onset + dur

    return header + " ".join(body_parts) + " |\n"


def tokens_to_symbolic_text(tokens: List[int], miditok_tokenizer) -> str:
    """Convert MidiTok token IDs to a line-oriented text representation."""
    try:
        vocab = getattr(miditok_tokenizer, "vocab", None)
        if vocab is not None and hasattr(vocab, "ids_to_tokens"):
            names = vocab.ids_to_tokens(tokens)
            return "\n".join(str(t) for t in names)
        if isinstance(vocab, dict):
            inv = {v: k for k, v in vocab.items()}
            return "\n".join(str(inv.get(t, f"UNK_{t}")) for t in tokens)
    except Exception:
        pass
    return "\n".join(f"tok_{t}" for t in tokens)


def text_to_patch_tensor(
    text: str,
    patch_size: int,
    max_patches: int = MAX_PATCHES,
) -> torch.Tensor:
    """Encode text to integer patch tensor (1, N, patch_size)."""
    patchilizer = M3Patchilizer()
    patches = patchilizer.encode(text, patch_size=patch_size, truncate=True)
    if not patches:
        patches = [[0] * patch_size]
    patches = patches[:max_patches]
    return torch.tensor(patches, dtype=torch.long).unsqueeze(0)


def patches_to_hidden(
    patches_long: torch.Tensor,
    patch_embedding: torch.nn.Module,
    encoder: torch.nn.Module,
    projection: torch.nn.Module,
    device: str,
) -> np.ndarray:
    """Run M3-style one-hot patch encoding through CLaMP music encoder."""
    patches_long = patches_long.to(device)
    n_patches = patches_long.shape[1]
    x = F.one_hot(patches_long.clamp(0, 127), 128).float()
    x = x.reshape(x.shape[0], n_patches, -1)
    x = patch_embedding(x)
    out = encoder(inputs_embeds=x)
    hidden = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]
    mask = torch.ones(1, n_patches, device=device).unsqueeze(-1)
    pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1)
    embedding = projection(pooled)
    embedding = F.normalize(embedding, dim=-1)
    return embedding[0].detach().cpu().numpy().astype(np.float32)
