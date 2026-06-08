"""CLaMP input format conversion (MTF, ABC, REMI token text).

Implements M3 bar-patching as used by sander-wood/clamp2 (M3Patchilizer).
"""

from __future__ import annotations

import os
import re
import tempfile
from typing import List, Optional

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


def pretty_midi_to_abc_text(midi_data) -> str:
    """Convert PrettyMIDI to ABC notation text via music21."""
    import music21

    with tempfile.NamedTemporaryFile(suffix=".mid", delete=False) as tmp:
        path = tmp.name
        midi_data.write(path)

    abc_path = tempfile.mktemp(suffix=".abc")
    try:
        score = music21.converter.parse(path)
        score.write("abc", fp=abc_path)
        with open(abc_path, "r", encoding="utf-8", errors="replace") as f:
            return f.read()
    except Exception:
        # Minimal fallback ABC from note events
        notes = []
        for inst in midi_data.instruments:
            if inst.is_drum:
                continue
            for note in inst.notes:
                notes.append((note.start, note.pitch, note.end - note.start))
        notes.sort(key=lambda x: x[0])
        body = " ".join(f"{p}/{max(1, int(d * 4))}" for _, p, d in notes[:256])
        return f"X:1\nT:Generated\nM:4/4\nL:1/8\nK:C\n{body}\n"
    finally:
        if os.path.exists(abc_path):
            os.unlink(abc_path)
        os.unlink(path)


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
