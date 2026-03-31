"""MIDI tokenization module using MidiTok."""

from pathlib import Path
from typing import Dict, List, Optional
from loguru import logger
import pretty_midi
from abc import ABC, abstractmethod
from miditok import REMI, TSD, Octuple, MIDILike, TokenizerConfig


def create_tokenizer_config(params: Dict) -> TokenizerConfig:
    """
    Create MidiTok TokenizerConfig from parameters.

    Args:
        params: Dictionary with tokenizer parameters

    Returns:
        TokenizerConfig object
    """
    # Extract parameters with defaults
    beat_res = params.get("beat_res", 480)
    nb_velocities = params.get("nb_velocities", 32)

    # Map beat_res to time signatures
    # 480 MIDI ticks per quarter note is standard
    beat_res_config = {(0, 4): 8}  # 8 positions per quarter note

    config = TokenizerConfig(
        pitch_range=(21, 109),  # MIDI range A0 to C8
        beat_res=beat_res_config,
        nb_velocities=nb_velocities,
        use_chords=False,
        use_rests=False,
        use_tempos=True,
        use_time_signatures=True,
        use_programs=True,
        nb_tempos=32,
        tempo_range=(40, 250),
    )

    return config


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
        self.miditok_tokenizer = None
        logger.info(f"Initialized {tokenizer_type} tokenizer")

    @abstractmethod
    def _create_miditok_tokenizer(self, params: Dict):
        """
        Create MidiTok tokenizer instance.

        Args:
            params: Tokenizer-specific parameters
        """
        pass

    @staticmethod
    def _normalize_token_ids(token_ids) -> List[int]:
        """Normalize MidiTok token IDs to a flat list of ints."""
        if token_ids is None:
            return []

        # Multi-vocabulary tokenizers (e.g. Octuple) can return nested ids.
        if isinstance(token_ids, list) and token_ids and isinstance(token_ids[0], (list, tuple)):
            flat_ids: List[int] = []
            for event in token_ids:
                flat_ids.extend(int(value) for value in event)
            return flat_ids

        if isinstance(token_ids, list):
            return [int(value) for value in token_ids]

        return []

    def encode(self, midi_path: Path) -> List[int]:
        """
        Encode MIDI file to token sequence.

        Args:
            midi_path: Path to MIDI file

        Returns:
            List of token IDs
        """
        if self.miditok_tokenizer is None:
            raise ValueError("MidiTok tokenizer not initialized")

        try:
            tokens = self.miditok_tokenizer(midi_path)
            if hasattr(tokens, "ids"):
                return self._normalize_token_ids(tokens.ids)
            if isinstance(tokens, list) and len(tokens) > 0:
                if hasattr(tokens[0], "ids"):
                    return self._normalize_token_ids(tokens[0].ids)
                return self._normalize_token_ids(tokens[0])
            return []
        except Exception as e:
            logger.error(f"Error encoding MIDI {midi_path}: {e}")
            return []

    def encode_midi_object(self, midi_data: pretty_midi.PrettyMIDI) -> List[int]:
        """
        Encode PrettyMIDI object to token sequence.

        Args:
            midi_data: PrettyMIDI object

        Returns:
            List of token IDs
        """
        # Save to temporary file and encode
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.mid', delete=False) as tmp:
            tmp_path = Path(tmp.name)
            midi_data.write(str(tmp_path))

        tokens = self.encode(tmp_path)

        # Clean up
        tmp_path.unlink()

        return tokens

    def decode(self, tokens: List[int], output_path: Optional[Path] = None) -> pretty_midi.PrettyMIDI:
        """
        Decode token sequence back to MIDI data.

        Args:
            tokens: List of token IDs
            output_path: Optional path to save MIDI file

        Returns:
            PrettyMIDI object
        """
        if self.miditok_tokenizer is None:
            raise ValueError("MidiTok tokenizer not initialized")

        try:
            # Create TokSequence from token IDs
            from miditok import TokSequence
            tok_seq = TokSequence(ids=tokens)

            # Decode to MIDI
            midi = self.miditok_tokenizer.decode(tok_seq)

            # Save if output path provided
            if output_path:
                midi.dump_midi(str(output_path))

            # Convert to PrettyMIDI
            import tempfile
            with tempfile.NamedTemporaryFile(suffix='.mid', delete=False) as tmp:
                tmp_path = Path(tmp.name)
                midi.dump_midi(str(tmp_path))
                pretty_midi_obj = pretty_midi.PrettyMIDI(str(tmp_path))
                tmp_path.unlink()

            return pretty_midi_obj

        except Exception as e:
            logger.error(f"Error decoding tokens: {e}")
            return pretty_midi.PrettyMIDI()

    def get_token_sequence_length(self, midi_path: Path) -> int:
        """
        Get length of token sequence for MIDI file.

        Args:
            midi_path: Path to MIDI file

        Returns:
            Length of token sequence
        """
        tokens = self.encode(midi_path)
        return len(tokens)

    def get_vocab_size(self) -> int:
        """
        Get vocabulary size of the tokenizer.

        Returns:
            Size of token vocabulary
        """
        if self.miditok_tokenizer is None:
            return 0

        vocab = self.miditok_tokenizer.vocab

        # Multi-vocabulary tokenizers can expose a list/dict of vocabularies.
        if isinstance(vocab, list):
            return int(sum(len(v) for v in vocab))

        if isinstance(vocab, dict):
            first_value = next(iter(vocab.values()), None)
            if isinstance(first_value, dict):
                return int(sum(len(v) for v in vocab.values()))
            return int(len(vocab))

        return int(len(vocab))


class REMITokenizer(TokenizerBase):
    """REMI (Relative Event-based MIDI Representation)."""

    def __init__(self, config: Dict):
        """Initialize REMI tokenizer."""
        super().__init__(config, "REMI")
        tokenizer_config = next(
            (t for t in config["tokenization"]["tokenizers"] if t["type"] == "REMI"), {}
        )
        self.params = tokenizer_config.get("params", {})
        self._create_miditok_tokenizer(self.params)

    def _create_miditok_tokenizer(self, params: Dict):
        """Create MidiTok REMI tokenizer."""
        tok_config = create_tokenizer_config(params)
        self.miditok_tokenizer = REMI(tok_config)
        logger.debug(f"REMI tokenizer created with vocab size: {self.get_vocab_size()}")


class TSDTokenizer(TokenizerBase):
    """TSD (Time-Shift-Duration representation)."""

    def __init__(self, config: Dict):
        """Initialize TSD tokenizer."""
        super().__init__(config, "TSD")
        tokenizer_config = next(
            (t for t in config["tokenization"]["tokenizers"] if t["type"] == "TSD"), {}
        )
        self.params = tokenizer_config.get("params", {})
        self._create_miditok_tokenizer(self.params)

    def _create_miditok_tokenizer(self, params: Dict):
        """Create MidiTok TSD tokenizer."""
        tok_config = create_tokenizer_config(params)
        self.miditok_tokenizer = TSD(tok_config)
        logger.debug(f"TSD tokenizer created with vocab size: {self.get_vocab_size()}")


class OctupleTokenizer(TokenizerBase):
    """Octuple (8-track symbolic representation)."""

    def __init__(self, config: Dict):
        """Initialize Octuple tokenizer."""
        super().__init__(config, "Octuple")
        tokenizer_config = next(
            (t for t in config["tokenization"]["tokenizers"] if t["type"] == "Octuple"), {}
        )
        self.params = tokenizer_config.get("params", {})
        self._create_miditok_tokenizer(self.params)

    def _create_miditok_tokenizer(self, params: Dict):
        """Create MidiTok Octuple tokenizer."""
        tok_config = create_tokenizer_config(params)
        self.miditok_tokenizer = Octuple(tok_config)
        logger.debug(f"Octuple tokenizer created with vocab size: {self.get_vocab_size()}")


class MIDILikeTokenizer(TokenizerBase):
    """MIDI-Like (Event-based MIDI representation)."""

    def __init__(self, config: Dict):
        """Initialize MIDI-Like tokenizer."""
        super().__init__(config, "MIDI-Like")
        tokenizer_config = next(
            (t for t in config["tokenization"]["tokenizers"] if t["type"] == "MIDI-Like"), {}
        )
        self.params = tokenizer_config.get("params", {})
        self._create_miditok_tokenizer(self.params)

    def _create_miditok_tokenizer(self, params: Dict):
        """Create MidiTok MIDI-Like tokenizer."""
        tok_config = create_tokenizer_config(params)
        self.miditok_tokenizer = MIDILike(tok_config)
        logger.debug(f"MIDI-Like tokenizer created with vocab size: {self.get_vocab_size()}")


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

    def tokenize_midi(self, midi_path: Path, tokenizer_type: str) -> List[int]:
        """
        Tokenize MIDI file using specified tokenizer.

        Args:
            midi_path: Path to MIDI file
            tokenizer_type: Type of tokenizer to use

        Returns:
            List of token IDs
        """
        if tokenizer_type not in self.tokenizers:
            raise ValueError(f"Unknown tokenizer type: {tokenizer_type}")

        tokenizer = self.tokenizers[tokenizer_type]
        return tokenizer.encode(midi_path)

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
        from tqdm import tqdm
        import json

        stats = {
            "tokenizer": tokenizer_type,
            "vocab_size": self.tokenizers[tokenizer_type].get_vocab_size(),
            "total": len(midi_files),
            "successful": 0,
            "failed": 0,
            "token_lengths": [],
            "failed_files": [],
        }

        output_dir.mkdir(parents=True, exist_ok=True)

        tokenizer = self.tokenizers[tokenizer_type]

        for midi_path in tqdm(midi_files, desc=f"Tokenizing with {tokenizer_type}"):
            try:
                tokens = tokenizer.encode(midi_path)

                if len(tokens) == 0:
                    stats["failed"] += 1
                    stats["failed_files"].append(str(midi_path))
                    logger.warning(f"Empty token sequence for {midi_path}")
                    continue

                # Save tokenized data
                output_file = output_dir / (midi_path.stem + f"_{tokenizer_type}.json")
                with open(output_file, "w") as f:
                    json.dump({"tokens": tokens, "file": str(midi_path)}, f)

                stats["successful"] += 1
                stats["token_lengths"].append(len(tokens))

            except Exception as e:
                logger.error(f"Error tokenizing {midi_path}: {e}")
                stats["failed"] += 1
                stats["failed_files"].append(str(midi_path))

        if stats["token_lengths"]:
            stats["avg_token_length"] = sum(stats["token_lengths"]) / len(stats["token_lengths"])
            stats["min_token_length"] = min(stats["token_lengths"])
            stats["max_token_length"] = max(stats["token_lengths"])
            stats["total_tokens"] = sum(stats["token_lengths"])

        logger.info(
            f"Tokenization complete: {stats['successful']} successful, {stats['failed']} failed"
        )

        # Save statistics to a filename that does not collide with token files glob pattern.
        stats_file = output_dir / f"tokenization_stats_{tokenizer_type}_summary.json"
        with open(stats_file, "w") as f:
            json.dump(stats, f, indent=2)

        return stats

    def load_tokens(self, token_file: Path) -> Dict:
        """
        Load tokens from file.

        Args:
            token_file: Path to token file

        Returns:
            Dictionary with tokens and metadata
        """
        import json
        with open(token_file, "r") as f:
            return json.load(f)

    def compare_tokenizers(self, midi_path: Path) -> Dict:
        """
        Compare all tokenizers on a single MIDI file.

        Args:
            midi_path: Path to MIDI file

        Returns:
            Dictionary with comparison results
        """
        results = {}

        for tok_type, tokenizer in self.tokenizers.items():
            try:
                tokens = tokenizer.encode(midi_path)
                results[tok_type] = {
                    "token_count": len(tokens),
                    "vocab_size": tokenizer.get_vocab_size(),
                    "compression_ratio": len(tokens) / tokenizer.get_vocab_size() if tokenizer.get_vocab_size() > 0 else 0,
                }
            except Exception as e:
                logger.error(f"Error with {tok_type}: {e}")
                results[tok_type] = {"error": str(e)}

        return results
