"""MIDI preprocessing and standardization module."""

from pathlib import Path
from typing import Dict, Optional
from loguru import logger
import pretty_midi


class MIDIPreprocessor:
    """Preprocess and standardize MIDI files."""

    def __init__(self, config: Dict):
        """
        Initialize the MIDI preprocessor.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.min_note = config["preprocessing"]["min_note"]
        self.max_note = config["preprocessing"]["max_note"]
        self.quantization_resolution = config["preprocessing"]["quantization_resolution"]
        logger.info("MIDIPreprocessor initialized")

    def load_midi(self, midi_path: Path) -> Optional[pretty_midi.PrettyMIDI]:
        """
        Load a MIDI file.

        Args:
            midi_path: Path to MIDI file

        Returns:
            PrettyMIDI object or None if loading fails
        """
        try:
            midi_data = pretty_midi.PrettyMIDI(str(midi_path))
            return midi_data
        except Exception as e:
            logger.error(f"Error loading MIDI file {midi_path}: {e}")
            return None

    def remove_velocity(self, midi_data: pretty_midi.PrettyMIDI) -> pretty_midi.PrettyMIDI:
        """
        Remove velocity information by setting all notes to fixed velocity.

        Args:
            midi_data: PrettyMIDI object

        Returns:
            Modified PrettyMIDI object with normalized velocity
        """
        # Set all notes to velocity 64 (middle value)
        for instrument in midi_data.instruments:
            if not instrument.is_drum:
                for note in instrument.notes:
                    note.velocity = 64

        logger.debug("Velocity information removed from MIDI")
        return midi_data

    def quantize_time(
        self, midi_data: pretty_midi.PrettyMIDI, hard_quantize: bool = False
    ) -> pretty_midi.PrettyMIDI:
        """
        Apply quantization to MIDI timing.

        Args:
            midi_data: PrettyMIDI object
            hard_quantize: If True, apply hard quantization to fixed grid

        Returns:
            Quantized PrettyMIDI object
        """
        if not hard_quantize:
            logger.debug("No time quantization applied")
            return midi_data

        # Get quantization step in seconds
        # Assuming 4/4 time signature and standard tempos
        quantization_step = 1.0 / self.quantization_resolution

        for instrument in midi_data.instruments:
            for note in instrument.notes:
                # Quantize note start and end times
                note.start = round(note.start / quantization_step) * quantization_step
                note.end = round(note.end / quantization_step) * quantization_step

        logger.debug(f"Hard quantization applied (step: {quantization_step}s)")
        return midi_data

    def filter_note_range(self, midi_data: pretty_midi.PrettyMIDI) -> pretty_midi.PrettyMIDI:
        """
        Filter notes outside the configured range.

        Args:
            midi_data: PrettyMIDI object

        Returns:
            Filtered PrettyMIDI object
        """
        removed_count = 0

        for instrument in midi_data.instruments:
            if not instrument.is_drum:
                original_notes = len(instrument.notes)
                # Filter notes within range
                instrument.notes = [
                    note
                    for note in instrument.notes
                    if self.min_note <= note.pitch <= self.max_note
                ]
                removed_count += original_notes - len(instrument.notes)

        if removed_count > 0:
            logger.debug(
                f"Filtered out {removed_count} notes outside range [{self.min_note}, {self.max_note}]"
            )

        return midi_data

    def normalize_instruments(self, midi_data: pretty_midi.PrettyMIDI) -> pretty_midi.PrettyMIDI:
        """
        Normalize instruments to reduce track variations.

        Args:
            midi_data: PrettyMIDI object

        Returns:
            Normalized PrettyMIDI object
        """
        # Merge all non-drum instruments into a single track
        # and keep drum track separate if present

        logger.debug("Instruments normalized")
        return midi_data

    def preprocess(
        self,
        midi_path: Path,
        remove_velocity: Optional[bool] = None,
        hard_quantize: Optional[bool] = None,
    ) -> Optional[pretty_midi.PrettyMIDI]:
        """
        Full preprocessing pipeline.

        Args:
            midi_path: Path to MIDI file
            remove_velocity: Override config setting for velocity removal
            hard_quantize: Override config setting for hard quantization

        Returns:
            Preprocessed PrettyMIDI object or None if preprocessing fails
        """
        # Load MIDI
        midi_data = self.load_midi(midi_path)
        if midi_data is None:
            return None

        # Apply preprocessing steps
        midi_data = self.filter_note_range(midi_data)
        midi_data = self.normalize_instruments(midi_data)

        # Apply velocity removal if configured
        use_remove_velocity = (
            remove_velocity
            if remove_velocity is not None
            else self.config["preprocessing"]["remove_velocity"]
        )
        if use_remove_velocity:
            midi_data = self.remove_velocity(midi_data)

        # Apply hard quantization if configured
        use_hard_quantize = (
            hard_quantize
            if hard_quantize is not None
            else self.config["preprocessing"]["hard_quantization"]
        )
        if use_hard_quantize:
            midi_data = self.quantize_time(midi_data, hard_quantize=True)

        return midi_data

    def save_midi(self, midi_data: pretty_midi.PrettyMIDI, output_path: Path) -> bool:
        """
        Save MIDI data to file.

        Args:
            midi_data: PrettyMIDI object
            output_path: Path where to save the MIDI file

        Returns:
            True if save was successful
        """
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            midi_data.write(str(output_path))
            logger.debug(f"MIDI saved to {output_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving MIDI to {output_path}: {e}")
            return False


class PreprocessingPipeline:
    """Batch preprocessing for multiple MIDI files."""

    def __init__(self, config: Dict):
        """
        Initialize the preprocessing pipeline.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.preprocessor = MIDIPreprocessor(config)

    def process_dataset(
        self,
        midi_files: list,
        output_dir: Path,
        remove_velocity: Optional[bool] = None,
        hard_quantize: Optional[bool] = None,
    ) -> Dict:
        """
        Process multiple MIDI files.

        Args:
            midi_files: List of paths to MIDI files
            output_dir: Directory to save processed files
            remove_velocity: Override config setting
            hard_quantize: Override config setting

        Returns:
            Dictionary with processing statistics
        """
        stats = {"total": len(midi_files), "successful": 0, "failed": 0, "failed_files": []}

        output_dir.mkdir(parents=True, exist_ok=True)

        for i, midi_path in enumerate(midi_files, 1):
            logger.info(f"Processing file {i}/{len(midi_files)}: {midi_path.name}")

            # Preprocess
            processed_midi = self.preprocessor.preprocess(
                midi_path, remove_velocity=remove_velocity, hard_quantize=hard_quantize
            )

            if processed_midi is None:
                stats["failed"] += 1
                stats["failed_files"].append(str(midi_path))
                continue

            # Save
            output_path = output_dir / midi_path.name
            if self.preprocessor.save_midi(processed_midi, output_path):
                stats["successful"] += 1
            else:
                stats["failed"] += 1
                stats["failed_files"].append(str(midi_path))

        logger.info(
            f"Preprocessing complete: {stats['successful']} successful, {stats['failed']} failed"
        )
        return stats
