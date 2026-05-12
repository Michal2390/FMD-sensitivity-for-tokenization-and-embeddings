#!/usr/bin/env python3
"""Run the embedding-architecture audit for FMD cross-model comparability.

This experiment complements the original sensitivity study: it preserves the
core question about tokenization and embedding choice, but adds a deeper audit
of embedding geometry, model integrity, and exploratory cross-model calibration.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from experiments.embedding_architecture_audit import EmbeddingArchitectureAudit
from utils.config import load_config


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument(
		"--config",
		default="configs/config.yaml",
		help="Path to YAML configuration file.",
	)
	parser.add_argument(
		"--max-files",
		type=int,
		default=None,
		help="Optional cap on MIDI files per genre/tokenizer/model cell.",
	)
	parser.add_argument(
		"--seed",
		type=int,
		default=42,
		help="Random seed used for baselines and shared-space calibration.",
	)
	parser.add_argument(
		"--print-json",
		action="store_true",
		help="Print the output manifest as JSON at the end.",
	)
	parser.add_argument(
		"--genres",
		default="",
		help="Comma-separated subset of genres for quicker smoke-runs.",
	)
	parser.add_argument(
		"--tokenizers",
		default="",
		help="Comma-separated subset of tokenizers for quicker smoke-runs.",
	)
	parser.add_argument(
		"--models",
		default="",
		help="Comma-separated subset of embedding models for quicker smoke-runs.",
	)
	return parser.parse_args()


def _parse_csv_arg(raw: str) -> list[str]:
  return [item.strip() for item in raw.split(",") if item.strip()]


def main() -> None:
	args = parse_args()
	config = load_config(args.config)
	max_files = args.max_files or int(config.get("paper", {}).get("max_files_per_dataset", 120))

	audit = EmbeddingArchitectureAudit(
		config=config,
		max_files=max_files,
		seed=args.seed,
	)

	genres = _parse_csv_arg(args.genres)
	tokenizers = _parse_csv_arg(args.tokenizers)
	models = _parse_csv_arg(args.models)
	if genres:
		audit.genres = [g for g in audit.genres if g in genres]
	if tokenizers:
		audit.tokenizers = [t for t in audit.tokenizers if t in tokenizers]
	if models:
		audit.models = [m for m in audit.models if m in models]
	outputs = audit.run()

	if args.print_json:
		print(json.dumps(outputs, indent=2))
	else:
		print("\nArchitecture audit outputs:")
		for key, value in outputs.items():
			print(f"- {key}: {value}")


if __name__ == "__main__":
	main()


