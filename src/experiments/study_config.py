"""Validation helpers for paper-grade FMD sensitivity studies.

The paper pipeline distinguishes tokenizers from model-native input formats.
CLaMP models are not MidiTok tokenizers: CLaMP-2 uses its MTF/M3 music input,
and CLaMP-1 uses ABC notation. Keeping this contract centralized prevents the
invalid CLaMP+REMI configuration that motivated the methodology review.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


MIDITOK_FORMAT = "MIDITOK"
MTF_FORMAT = "MTF"
ABC_FORMAT = "ABC"

MIDITOK_TOKENIZERS = {"REMI", "TSD", "Octuple", "MIDI-Like"}


@dataclass(frozen=True)
class EmbeddingInputSpec:
    """Validated model/input-format/tokenizer contract."""

    model: str
    input_format: str
    tokenizer: Optional[str] = None


def infer_input_format(model: str, tokenizer: Optional[str] = None) -> str:
    """Infer the only valid default input format for a model.

    MidiTok-aware symbolic models consume tokenizer IDs/text. CLaMP models use
    model-native music representations and must not be relabelled as REMI/TSD.
    """
    if model == "CLaMP-2":
        return MTF_FORMAT
    if model == "CLaMP-1":
        return ABC_FORMAT
    if tokenizer:
        return MIDITOK_FORMAT
    return MIDITOK_FORMAT


def validate_embedding_input(
    model: str,
    input_format: Optional[str] = None,
    tokenizer: Optional[str] = None,
) -> EmbeddingInputSpec:
    """Validate a paper-study embedding configuration.

    Raises:
        ValueError: if the combination would make an invalid scientific claim.
    """
    resolved_format = (input_format or infer_input_format(model, tokenizer)).upper()
    if resolved_format == "REMI":
        # REMI is a tokenizer, not an embedding input family. Keep old YAMLs from
        # accidentally encoding the previous CLaMP2-REMI mistake.
        resolved_format = MIDITOK_FORMAT
        tokenizer = tokenizer or "REMI"

    if model == "CLaMP-2":
        if resolved_format != MTF_FORMAT or tokenizer:
            raise ValueError(
                "CLaMP-2 must be configured as input_format=MTF with no MidiTok tokenizer. "
                "It does not consume REMI/TSD/Octuple/MIDI-Like tokenizers."
            )
        return EmbeddingInputSpec(model=model, input_format=MTF_FORMAT, tokenizer=None)

    if model == "CLaMP-1":
        if resolved_format != ABC_FORMAT or tokenizer:
            raise ValueError(
                "CLaMP-1 must be configured as input_format=ABC with no MidiTok tokenizer. "
                "ABC is a notation conversion, not a MidiTok tokenizer."
            )
        return EmbeddingInputSpec(model=model, input_format=ABC_FORMAT, tokenizer=None)

    if resolved_format != MIDITOK_FORMAT:
        raise ValueError(
            f"Model {model} is configured with unsupported input_format={resolved_format}. "
            "Use MIDITOK plus an explicit tokenizer for tokenizer-sensitivity studies."
        )
    if not tokenizer:
        raise ValueError(f"Model {model} requires a MidiTok tokenizer in paper studies")
    if tokenizer not in MIDITOK_TOKENIZERS:
        raise ValueError(f"Unknown MidiTok tokenizer for paper study: {tokenizer}")

    return EmbeddingInputSpec(model=model, input_format=MIDITOK_FORMAT, tokenizer=tokenizer)


def requires_midi_data(spec: EmbeddingInputSpec) -> bool:
    """Return whether an embedding config needs the original MIDI object."""
    return spec.input_format in {MTF_FORMAT, ABC_FORMAT}
