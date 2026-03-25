"""MIDI tokenization module using MidiTok."""

from pathlib import Path
from typing import Dict, List
from loguru import logger
import pretty_midi
from abc import ABC, abstractmethod


class TokenizerBase(ABC):
    """Base class for MIDI tokenizers."""

    def __init__(self, config: Dict, tokenizer_type: str):
        """
        Initialize tokenizer.

        Args:
            config: Configuration dictionary
            tokenizer_type: Type of tokenizer (REMI, TSD, Octuple, MIDI-Like)
        """
        self.config = config
        self.tokenizer_type = tokenizer_type
        logger.info(f"Initialized {tokenizer_type} tokenizer")

    @abstractmethod
    def encode(self, midi_data: pretty_midi.PrettyMIDI) -> List[int]:
        """
        Encode MIDI data to token sequence.

        Args:
            midi_data: PrettyMIDI object

        Returns:
            List of token IDs
        """
        pass

    @abstractmethod
    def decode(self, tokens: List[int]) -> pretty_midi.PrettyMIDI:
        """
        Decode token sequence back to MIDI data.

        Args:
            tokens: List of token IDs

        Returns:
            PrettyMIDI object
        """
        pass

    def get_token_sequence_length(self, midi_data: pretty_midi.PrettyMIDI) -> int:
        """
        Get length of token sequence for MIDI data.

        Args:
            midi_data: PrettyMIDI object

        Returns:
            Length of token sequence
        """
        tokens = self.encode(midi_data)
        return len(tokens)


class REMITokenizer(TokenizerBase):
    """REMI (Relative Event-based MIDI Representation)."""

    def __init__(self, config: Dict):
        """Initialize REMI tokenizer."""
        super().__init__(config, "REMI")
        tokenizer_config = next(
            (t for t in config["tokenization"]["tokenizers"] if t["type"] == "REMI"), {}
        )
        self.params = tokenizer_config.get("params", {})

    def encode(self, midi_data: pretty_midi.PrettyMIDI) -> List[int]:
        """Encode MIDI to REMI tokens."""
        # Placeholder: Will use MidiTok library
        logger.debug("Encoding MIDI to REMI tokens")
        return [0] * 100  # Placeholder

    def decode(self, tokens: List[int]) -> pretty_midi.PrettyMIDI:
        """Decode REMI tokens to MIDI."""
        logger.debug("Decoding REMI tokens to MIDI")
        return pretty_midi.PrettyMIDI()  # Placeholder


class TSDTokenizer(TokenizerBase):
    """TSD (Time-Shift-Duration representation)."""

    def __init__(self, config: Dict):
        """Initialize TSD tokenizer."""
        super().__init__(config, "TSD")
        tokenizer_config = next(
            (t for t in config["tokenization"]["tokenizers"] if t["type"] == "TSD"), {}
        )
        self.params = tokenizer_config.get("params", {})

    def encode(self, midi_data: pretty_midi.PrettyMIDI) -> List[int]:
        """Encode MIDI to TSD tokens."""
        logger.debug("Encoding MIDI to TSD tokens")
        return [0] * 100  # Placeholder

    def decode(self, tokens: List[int]) -> pretty_midi.PrettyMIDI:
        """Decode TSD tokens to MIDI."""
        logger.debug("Decoding TSD tokens to MIDI")
        return pretty_midi.PrettyMIDI()  # Placeholder


class OctupleTokenizer(TokenizerBase):
    """Octuple (8-track symbolic representation)."""

    def __init__(self, config: Dict):
        """Initialize Octuple tokenizer."""
        super().__init__(config, "Octuple")
        tokenizer_config = next(
            (t for t in config["tokenization"]["tokenizers"] if t["type"] == "Octuple"), {}
        )
        self.params = tokenizer_config.get("params", {})

    def encode(self, midi_data: pretty_midi.PrettyMIDI) -> List[int]:
        """Encode MIDI to Octuple tokens."""
        logger.debug("Encoding MIDI to Octuple tokens")
        return [0] * 100  # Placeholder

    def decode(self, tokens: List[int]) -> pretty_midi.PrettyMIDI:
        """Decode Octuple tokens to MIDI."""
        logger.debug("Decoding Octuple tokens to MIDI")
        return pretty_midi.PrettyMIDI()  # Placeholder


class MIDILikeTokenizer(TokenizerBase):
    """MIDI-Like (Event-based MIDI representation)."""

    def __init__(self, config: Dict):
        """Initialize MIDI-Like tokenizer."""
        super().__init__(config, "MIDI-Like")
        tokenizer_config = next(
            (t for t in config["tokenization"]["tokenizers"] if t["type"] == "MIDI-Like"), {}
        )
        self.params = tokenizer_config.get("params", {})

    def encode(self, midi_data: pretty_midi.PrettyMIDI) -> List[int]:
        """Encode MIDI to MIDI-Like tokens."""
        logger.debug("Encoding MIDI to MIDI-Like tokens")
        return [0] * 100  # Placeholder

    def decode(self, tokens: List[int]) -> pretty_midi.PrettyMIDI:
        """Decode MIDI-Like tokens to MIDI."""
        logger.debug("Decoding MIDI-Like tokens to MIDI")
        return pretty_midi.PrettyMIDI()  # Placeholder


class TokenizationFactory:
    """Factory for creating tokenizers."""

    _tokenizers = {
        "REMI": REMITokenizer,
        "TSD": TSDTokenizer,
        "Octuple": OctupleTokenizer,
        "MIDI-Like": MIDILikeTokenizer,
    }

    @classmethod
    def create_tokenizer(cls, config: Dict, tokenizer_type: str) -> TokenizerBase:
        """
        Create a tokenizer instance.

        Args:
            config: Configuration dictionary
            tokenizer_type: Type of tokenizer

        Returns:
            Tokenizer instance
        """
        if tokenizer_type not in cls._tokenizers:
            raise ValueError(f"Unknown tokenizer type: {tokenizer_type}")

        return cls._tokenizers[tokenizer_type](config)

    @classmethod
    def get_available_tokenizers(cls) -> List[str]:
        """Get list of available tokenizers."""
        return list(cls._tokenizers.keys())


class TokenizationPipeline:
    """Pipeline for tokenizing MIDI data."""

    def __init__(self, config: Dict):
        """
        Initialize tokenization pipeline.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.tokenizers = {
            tok_type: TokenizationFactory.create_tokenizer(config, tok_type)
            for tok_type in TokenizationFactory.get_available_tokenizers()
        }
        logger.info(f"Tokenization pipeline initialized with {len(self.tokenizers)} tokenizers")

    def tokenize_midi(self, midi_data: pretty_midi.PrettyMIDI, tokenizer_type: str) -> List[int]:
        """
        Tokenize MIDI data using specified tokenizer.

        Args:
            midi_data: PrettyMIDI object
            tokenizer_type: Type of tokenizer to use

        Returns:
            List of token IDs
        """
        if tokenizer_type not in self.tokenizers:
            raise ValueError(f"Unknown tokenizer type: {tokenizer_type}")

        tokenizer = self.tokenizers[tokenizer_type]
        return tokenizer.encode(midi_data)

    def tokenize_dataset(
        self, midi_files: List[Path], output_dir: Path, tokenizer_type: str
    ) -> Dict:
        """
        Tokenize multiple MIDI files.

        Args:
            midi_files: List of paths to MIDI files
            output_dir: Directory to save tokenized data
            tokenizer_type: Type of tokenizer to use

        Returns:
            Dictionary with tokenization statistics
        """
        stats = {
            "tokenizer": tokenizer_type,
            "total": len(midi_files),
            "successful": 0,
            "failed": 0,
            "token_lengths": [],
        }

        output_dir.mkdir(parents=True, exist_ok=True)

        tokenizer = self.tokenizers[tokenizer_type]

        for i, midi_path in enumerate(midi_files, 1):
            logger.info(f"Tokenizing file {i}/{len(midi_files)}: {midi_path.name}")

            try:
                midi_data = pretty_midi.PrettyMIDI(str(midi_path))
                tokens = tokenizer.encode(midi_data)

                # Save tokenized data
                output_file = output_dir / (midi_path.stem + f"_{tokenizer_type}.txt")
                with open(output_file, "w") as f:
                    f.write(" ".join(map(str, tokens)))

                stats["successful"] += 1
                stats["token_lengths"].append(len(tokens))
            except Exception as e:
                logger.error(f"Error tokenizing {midi_path}: {e}")
                stats["failed"] += 1

        if stats["token_lengths"]:
            stats["avg_token_length"] = sum(stats["token_lengths"]) / len(stats["token_lengths"])
            stats["min_token_length"] = min(stats["token_lengths"])
            stats["max_token_length"] = max(stats["token_lengths"])

        logger.info(
            f"Tokenization complete: {stats['successful']} successful, {stats['failed']} failed"
        )
        return stats
