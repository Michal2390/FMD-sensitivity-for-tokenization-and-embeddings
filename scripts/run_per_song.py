#!/usr/bin/env python3
"""Run per-song FMD sensitivity analysis for a single MIDI file.

Usage (from repo root):
  python scripts\run_per_song.py --midi-file data\raw\lakh_rock\000af203cb1d081a52cea33aaca5fec3.mid \
      --tokenizers REMI TSD Octuple "MIDI-Like" --models "CLaMP-1" "CLaMP-2" MusicBERT --segments 8 --axis both

The script will load the project config, create a PaperExperimentRunner and run
run_single_song_analysis with the requested tokenizers/models. Results (CSV/JSON/MD)
are written to results/reports/per_song/<song_label>/.
"""

from pathlib import Path
import sys
import argparse
import json

# Make src importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from utils.config import load_config, setup_logging, get_logger
from experiments.paper_pipeline import PaperExperimentRunner


def parse_args():
    p = argparse.ArgumentParser(description="Run per-song FMD sensitivity analysis")
    p.add_argument("--midi-file", required=True, help="Path to MIDI file")
    p.add_argument("--tokenizers", nargs="+", default=None, help="List of tokenizers to compare")
    p.add_argument("--models", nargs="+", default=None, help="List of embedding models (used for tokenizer comparison)")
    p.add_argument("--segments", type=int, default=8, help="Number of segments to split the song into")
    p.add_argument("--model", default=None, help="Fixed model used for tokenizer comparison (default: first model from config)")
    p.add_argument("--remove-velocity", action="store_true")
    p.add_argument("--hard-quantization", action="store_true")
    p.add_argument("--output-dir", default=None, help="Optional output directory")
    return p.parse_args()


def main():
    args = parse_args()
    config = load_config("configs/config.yaml")
    setup_logging(config.get("logging", {}).get("level", "INFO"), config.get("logging", {}).get("log_file", "logs/experiment.log"))
    logger = get_logger("run_per_song")

    midi_path = Path(args.midi_file)
    if not midi_path.exists():
        logger.error(f"MIDI file not found: {midi_path}")
        sys.exit(2)

    runner = PaperExperimentRunner(config)

    tokenizers = args.tokenizers
    models = args.models

    logger.info(f"Running per-song analysis: {midi_path}\n tokenizers={tokenizers}\n models={models}")

    result = runner.run_single_song_analysis(
        midi_path=midi_path,
        tokenizers=tokenizers,
        models=models,
        n_segments=args.segments,
        remove_velocity=args.remove_velocity,
        hard_quantization=args.hard_quantization,
        axis="tokenizer",
        output_dir=Path(args.output_dir) if args.output_dir else None,
    )

    # Print summary paths
    outputs = result.get("outputs", {})
    out = {"midi": str(midi_path), "outputs": outputs}
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
