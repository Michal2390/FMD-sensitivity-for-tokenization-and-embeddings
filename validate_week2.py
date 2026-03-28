"""
Validation script for Week 2 implementation.

This script validates the preprocessing and tokenization pipeline
by processing sample MIDI files and checking the results.

Usage:
    python validate_week2.py [--create-sample] [--run-tests]
"""

import argparse
from pathlib import Path
import yaml
import pretty_midi
from loguru import logger

from src.preprocessing.processor import PreprocessingPipeline
from src.tokenization.tokenizer import TokenizationPipeline


def create_sample_midi(output_path: Path):
    """
    Create a sample MIDI file for testing.

    Args:
        output_path: Path where to save the sample MIDI
    """
    logger.info("Creating sample MIDI file...")

    midi = pretty_midi.PrettyMIDI(initial_tempo=120)

    # Create a piano instrument
    piano = pretty_midi.Instrument(program=0, name="Piano")

    # Add a C major scale
    pitches = [60, 62, 64, 65, 67, 69, 71, 72]  # C D E F G A B C
    velocities = [80, 85, 90, 85, 80, 75, 70, 80]

    for i, (pitch, velocity) in enumerate(zip(pitches, velocities)):
        note = pretty_midi.Note(
            velocity=velocity,
            pitch=pitch,
            start=i * 0.5,
            end=(i + 0.8) * 0.5
        )
        piano.notes.append(note)

    # Add a simple chord progression
    chords = [
        [60, 64, 67],  # C major
        [62, 65, 69],  # D minor
        [64, 67, 71],  # E minor
        [60, 64, 67],  # C major
    ]

    for i, chord_notes in enumerate(chords):
        start_time = 4.0 + i * 2.0
        for pitch in chord_notes:
            note = pretty_midi.Note(
                velocity=70,
                pitch=pitch,
                start=start_time,
                end=start_time + 1.8
            )
            piano.notes.append(note)

    midi.instruments.append(piano)

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    midi.write(str(output_path))

    logger.info(f"✓ Sample MIDI created: {output_path}")
    logger.info(f"  - Duration: {midi.get_end_time():.2f}s")
    logger.info(f"  - Notes: {len(piano.notes)}")

    return output_path


def validate_preprocessing(config: dict, midi_files: list):
    """
    Validate preprocessing functionality.

    Args:
        config: Configuration dictionary
        midi_files: List of MIDI files to process

    Returns:
        True if validation passed
    """
    logger.info("\n" + "=" * 60)
    logger.info("VALIDATING PREPROCESSING")
    logger.info("=" * 60)

    pipeline = PreprocessingPipeline(config)

    # Test 1: Basic preprocessing
    logger.info("\n[Test 1] Basic preprocessing (no velocity removal, no quantization)")
    output_dir = Path("data/processed/validation_test1")
    stats = pipeline.process_dataset(
        midi_files,
        output_dir,
        remove_velocity=False,
        hard_quantize=False
    )

    if stats["successful"] != len(midi_files):
        logger.error(f"✗ Expected {len(midi_files)} successful, got {stats['successful']}")
        return False
    logger.info(f"✓ Processed {stats['successful']}/{stats['total']} files")

    # Test 2: With velocity removal
    logger.info("\n[Test 2] Preprocessing with velocity removal")
    output_dir = Path("data/processed/validation_test2")
    stats = pipeline.process_dataset(
        midi_files,
        output_dir,
        remove_velocity=True,
        hard_quantize=False
    )

    if stats["successful"] != len(midi_files):
        logger.error(f"✗ Expected {len(midi_files)} successful, got {stats['successful']}")
        return False
    logger.info(f"✓ Processed {stats['successful']}/{stats['total']} files")

    # Test 3: With hard quantization
    logger.info("\n[Test 3] Preprocessing with hard quantization")
    output_dir = Path("data/processed/validation_test3")
    stats = pipeline.process_dataset(
        midi_files,
        output_dir,
        remove_velocity=False,
        hard_quantize=True
    )

    if stats["successful"] != len(midi_files):
        logger.error(f"✗ Expected {len(midi_files)} successful, got {stats['successful']}")
        return False
    logger.info(f"✓ Processed {stats['successful']}/{stats['total']} files")

    logger.info("\n✓ All preprocessing tests passed!")
    return True


def validate_tokenization(config: dict, midi_files: list):
    """
    Validate tokenization functionality.

    Args:
        config: Configuration dictionary
        midi_files: List of MIDI files to tokenize

    Returns:
        True if validation passed
    """
    logger.info("\n" + "=" * 60)
    logger.info("VALIDATING TOKENIZATION")
    logger.info("=" * 60)

    pipeline = TokenizationPipeline(config)

    tokenizer_types = ["REMI", "TSD", "Octuple", "MIDI-Like"]

    for tok_type in tokenizer_types:
        logger.info(f"\n[Test {tok_type}] Tokenizing with {tok_type}")

        output_dir = Path(f"data/embeddings/validation_{tok_type}")
        stats = pipeline.tokenize_dataset(
            midi_files,
            output_dir,
            tok_type
        )

        if stats["successful"] != len(midi_files):
            logger.error(f"✗ Expected {len(midi_files)} successful, got {stats['successful']}")
            return False

        logger.info(f"✓ Tokenized {stats['successful']}/{stats['total']} files")
        logger.info(f"  - Vocab size: {stats['vocab_size']}")
        logger.info(f"  - Avg token length: {stats.get('avg_token_length', 0):.1f}")

    logger.info("\n✓ All tokenization tests passed!")
    return True


def compare_tokenizers_on_sample(config: dict, midi_file: Path):
    """
    Compare all tokenizers on a sample file.

    Args:
        config: Configuration dictionary
        midi_file: Path to MIDI file
    """
    logger.info("\n" + "=" * 60)
    logger.info("COMPARING TOKENIZERS")
    logger.info("=" * 60)

    pipeline = TokenizationPipeline(config)
    results = pipeline.compare_tokenizers(midi_file)

    logger.info(f"\nTokenizer comparison for: {midi_file.name}")
    logger.info("-" * 60)
    logger.info(f"{'Tokenizer':<15} {'Tokens':<10} {'Vocab':<10} {'Ratio':<10}")
    logger.info("-" * 60)

    for tok_type, result in results.items():
        if "error" not in result:
            logger.info(
                f"{tok_type:<15} "
                f"{result['token_count']:<10} "
                f"{result['vocab_size']:<10} "
                f"{result['compression_ratio']:<10.3f}"
            )
        else:
            logger.warning(f"{tok_type:<15} ERROR: {result['error']}")


def main():
    """Main validation function."""
    parser = argparse.ArgumentParser(description="Validate Week 2 implementation")
    parser.add_argument(
        "--create-sample",
        action="store_true",
        help="Create a sample MIDI file for testing"
    )
    parser.add_argument(
        "--run-tests",
        action="store_true",
        help="Run validation tests"
    )

    args = parser.parse_args()

    # Load configuration
    config_path = Path("configs/config.yaml")
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Create sample if requested
    sample_path = Path("data/raw/sample_validation.mid")
    if args.create_sample or not sample_path.exists():
        create_sample_midi(sample_path)

    # Find MIDI files
    data_dir = Path(config["data"]["raw_data_dir"])
    midi_files = list(data_dir.glob("**/*.mid")) + list(data_dir.glob("**/*.midi"))

    if not midi_files:
        logger.warning("No MIDI files found in data/raw")
        logger.info("Creating sample MIDI for testing...")
        sample_path = create_sample_midi(sample_path)
        midi_files = [sample_path]

    logger.info(f"\nFound {len(midi_files)} MIDI file(s) for validation")

    # Run tests if requested
    if args.run_tests:
        # Validate preprocessing
        preprocess_ok = validate_preprocessing(config, midi_files)

        if not preprocess_ok:
            logger.error("\n✗ Preprocessing validation FAILED")
            return

        # Validate tokenization
        tokenization_ok = validate_tokenization(config, midi_files)

        if not tokenization_ok:
            logger.error("\n✗ Tokenization validation FAILED")
            return

        # Compare tokenizers
        compare_tokenizers_on_sample(config, midi_files[0])

        logger.info("\n" + "=" * 60)
        logger.info("✓✓✓ ALL VALIDATIONS PASSED ✓✓✓")
        logger.info("=" * 60)
        logger.info("\nWeek 2 implementation is working correctly!")
        logger.info("Ready to proceed to Week 3 (CLaMP integration)")

    else:
        logger.info("\nUse --run-tests to run validation tests")
        logger.info("Use --create-sample to create a sample MIDI file")


if __name__ == "__main__":
    main()
