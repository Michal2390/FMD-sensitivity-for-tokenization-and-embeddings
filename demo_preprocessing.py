"""
Demo script for MIDI preprocessing and tokenization.

This script demonstrates the preprocessing and tokenization pipeline
for Week 2 of the FMD sensitivity analysis project.

Usage:
    python demo_preprocessing.py [--midi-file PATH] [--show-comparison]
"""

import argparse
from pathlib import Path
import yaml
from loguru import logger

from src.preprocessing.processor import MIDIPreprocessor, PreprocessingPipeline
from src.tokenization.tokenizer import TokenizationPipeline


def load_config():
    """Load configuration from config.yaml."""
    config_path = Path("configs/config.yaml")
    with open(config_path) as f:
        return yaml.safe_load(f)


def demo_preprocessing(midi_path: Path, config: dict):
    """
    Demonstrate MIDI preprocessing.

    Args:
        midi_path: Path to MIDI file
        config: Configuration dictionary
    """
    logger.info("=" * 60)
    logger.info("MIDI PREPROCESSING DEMO")
    logger.info("=" * 60)

    preprocessor = MIDIPreprocessor(config)

    # Load MIDI
    logger.info(f"\n1. Loading MIDI file: {midi_path.name}")
    midi_data = preprocessor.load_midi(midi_path)

    if midi_data is None:
        logger.error("Failed to load MIDI file")
        return None

    # Display basic info
    logger.info(f"   - Duration: {midi_data.get_end_time():.2f} seconds")
    logger.info(f"   - Number of instruments: {len(midi_data.instruments)}")
    logger.info(f"   - Total notes: {sum(len(inst.notes) for inst in midi_data.instruments)}")

    # Filter note range
    logger.info("\n2. Filtering note range [21-108]")
    midi_data = preprocessor.filter_note_range(midi_data)
    logger.info(f"   - Notes after filtering: {sum(len(inst.notes) for inst in midi_data.instruments)}")

    # Normalize instruments
    logger.info("\n3. Normalizing instruments")
    original_inst_count = len(midi_data.instruments)
    midi_data = preprocessor.normalize_instruments(midi_data)
    logger.info(f"   - Instruments: {original_inst_count} -> {len(midi_data.instruments)}")

    # Optional: Remove velocity
    logger.info("\n4. Velocity handling")
    if config["preprocessing"]["remove_velocity"]:
        logger.info("   - Removing velocity information")
        midi_data = preprocessor.remove_velocity(midi_data)
    else:
        logger.info("   - Keeping velocity information")

    # Optional: Quantize time
    logger.info("\n5. Time quantization")
    if config["preprocessing"]["hard_quantization"]:
        logger.info("   - Applying hard quantization")
        midi_data = preprocessor.quantize_time(midi_data, hard_quantize=True)
    else:
        logger.info("   - No quantization applied")

    logger.info("\n✓ Preprocessing complete")
    return midi_data


def demo_tokenization(midi_path: Path, config: dict, show_comparison: bool = False):
    """
    Demonstrate MIDI tokenization.

    Args:
        midi_path: Path to MIDI file
        config: Configuration dictionary
        show_comparison: Whether to show comparison of all tokenizers
    """
    logger.info("\n" + "=" * 60)
    logger.info("MIDI TOKENIZATION DEMO")
    logger.info("=" * 60)

    pipeline = TokenizationPipeline(config)

    if show_comparison:
        # Compare all tokenizers
        logger.info(f"\nComparing all tokenizers on: {midi_path.name}")
        results = pipeline.compare_tokenizers(midi_path)

        logger.info("\nTokenization Comparison:")
        logger.info("-" * 60)
        logger.info(f"{'Tokenizer':<15} {'Token Count':<12} {'Vocab Size':<12} {'Ratio':<10}")
        logger.info("-" * 60)

        for tok_type, result in results.items():
            if "error" not in result:
                logger.info(
                    f"{tok_type:<15} "
                    f"{result['token_count']:<12} "
                    f"{result['vocab_size']:<12} "
                    f"{result['compression_ratio']:<10.3f}"
                )
            else:
                logger.warning(f"{tok_type:<15} ERROR: {result['error']}")

    else:
        # Demonstrate REMI tokenization in detail
        logger.info(f"\nTokenizing with REMI: {midi_path.name}")

        tokenizer = pipeline.tokenizers["REMI"]

        # Encode
        logger.info("\n1. Encoding MIDI to tokens...")
        tokens = tokenizer.encode(midi_path)
        logger.info(f"   - Token sequence length: {len(tokens)}")
        logger.info(f"   - Vocabulary size: {tokenizer.get_vocab_size()}")
        logger.info(f"   - First 20 tokens: {tokens[:20]}")

        # Decode
        logger.info("\n2. Decoding tokens back to MIDI...")
        decoded_midi = tokenizer.decode(tokens)
        logger.info(f"   - Reconstructed duration: {decoded_midi.get_end_time():.2f} seconds")
        logger.info(f"   - Reconstructed instruments: {len(decoded_midi.instruments)}")

        logger.info("\n✓ Tokenization complete")


def demo_full_pipeline(midi_path: Path, config: dict):
    """
    Demonstrate the full preprocessing and tokenization pipeline.

    Args:
        midi_path: Path to MIDI file
        config: Configuration dictionary
    """
    import tempfile

    logger.info("\n" + "=" * 60)
    logger.info("FULL PIPELINE DEMO")
    logger.info("=" * 60)

    # Step 1: Preprocessing
    logger.info("\nStep 1: Preprocessing")
    preprocess_pipeline = PreprocessingPipeline(config)

    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir) / "preprocessed"
        output_path = output_dir / midi_path.name

        success = preprocess_pipeline.process_single_file(
            midi_path,
            output_path,
            remove_velocity=False,
            hard_quantize=False
        )

        if not success:
            logger.error("Preprocessing failed")
            return

        logger.info(f"✓ Preprocessed MIDI saved to: {output_path}")

        # Step 2: Tokenization
        logger.info("\nStep 2: Tokenization")
        tokenization_pipeline = TokenizationPipeline(config)

        tokens_dir = Path(tmpdir) / "tokens"
        stats = tokenization_pipeline.tokenize_dataset(
            [output_path],
            tokens_dir,
            "REMI"
        )

        logger.info(f"\nTokenization Statistics:")
        logger.info(f"  - Files processed: {stats['successful']}/{stats['total']}")
        logger.info(f"  - Average token length: {stats.get('avg_token_length', 0):.1f}")
        logger.info(f"  - Vocabulary size: {stats['vocab_size']}")

        logger.info("\n✓ Full pipeline complete")


def main():
    """Main demo function."""
    parser = argparse.ArgumentParser(description="Demo for MIDI preprocessing and tokenization")
    parser.add_argument(
        "--midi-file",
        type=str,
        help="Path to MIDI file to process",
    )
    parser.add_argument(
        "--show-comparison",
        action="store_true",
        help="Show comparison of all tokenizers",
    )
    parser.add_argument(
        "--full-pipeline",
        action="store_true",
        help="Run full preprocessing and tokenization pipeline",
    )

    args = parser.parse_args()

    # Load configuration
    config = load_config()

    # Find MIDI file
    if args.midi_file:
        midi_path = Path(args.midi_file)
        if not midi_path.exists():
            logger.error(f"MIDI file not found: {midi_path}")
            return
    else:
        # Look for sample MIDI in data/raw
        data_dir = Path(config["data"]["raw_data_dir"])
        if data_dir.exists():
            midi_files = list(data_dir.glob("**/*.mid")) + list(data_dir.glob("**/*.midi"))
            if midi_files:
                midi_path = midi_files[0]
                logger.info(f"Using sample MIDI file: {midi_path}")
            else:
                logger.error("No MIDI files found in data/raw directory")
                logger.info("Please provide a MIDI file with --midi-file")
                return
        else:
            logger.error(f"Data directory not found: {data_dir}")
            logger.info("Please provide a MIDI file with --midi-file")
            return

    # Run demos
    if args.full_pipeline:
        demo_full_pipeline(midi_path, config)
    else:
        processed_midi = demo_preprocessing(midi_path, config)
        if processed_midi:
            demo_tokenization(midi_path, config, show_comparison=args.show_comparison)

    logger.info("\n" + "=" * 60)
    logger.info("DEMO COMPLETE")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
