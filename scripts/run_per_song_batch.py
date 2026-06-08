#!/usr/bin/env python3
"""Stream per-song FMD summaries into two global CSV files.

This batch runner does not create per-song folders. It processes each MIDI file
in memory and appends one row per song to:

- summary_tokenizer_for_model_<model>.csv
- summary_model_for_tokenizer_<tokenizer>.csv
"""

from __future__ import annotations

import argparse
import csv
import statistics
import sys
import os
import traceback
from contextlib import contextmanager
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

# Make src importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from experiments.paper_pipeline import PaperExperimentRunner, PipelineVariant
from utils.config import get_logger, load_config, setup_logging


@contextmanager
def suppress_stdout_stderr():
    """Redirect stdout and stderr to devnull (silences C-extension and subprocess prints)."""
    try:
        sys.stdout.flush()
        sys.stderr.flush()
    except Exception:
        pass
    devnull_fd = os.open(os.devnull, os.O_RDWR)
    stdout_fd = os.dup(1)
    stderr_fd = os.dup(2)
    try:
        os.dup2(devnull_fd, 1)
        os.dup2(devnull_fd, 2)
        os.close(devnull_fd)
        yield
    finally:
        os.dup2(stdout_fd, 1)
        os.dup2(stderr_fd, 2)
        os.close(stdout_fd)
        os.close(stderr_fd)

def parse_args():
    p = argparse.ArgumentParser(description="Batch per-song runner that writes two global CSV files")
    p.add_argument("--base-dir", default="data/raw", help="Base directory containing Lakh split folders")
    p.add_argument("--pattern", default="lakh_*", help="Glob pattern to match split folders")
    p.add_argument("--n-per-genre", type=int, default=3, help="How many songs to run per genre (ignored when --all is set)")
    p.add_argument("--all", action="store_true", help="Process all MIDI files in each folder (ignore --n-per-genre)")
    p.add_argument("--model", default=None, help="Fixed model used for the tokenizer-comparison CSV")
    p.add_argument("--tokenizer", default=None, help="Fixed tokenizer used for the model-comparison CSV")
    p.add_argument("--tokenizers", nargs="+", default=None, help="Tokenizers to test (default from config)")
    p.add_argument("--models", nargs="+", default=None, help="Models to test (default from config)")
    p.add_argument("--segments", type=int, default=8, help="How many time windows to build per song")
    p.add_argument(
        "--output-dir",
        default=None,
        help="Directory for the two aggregate CSV files (default: results/reports/per_song)",
    )
    return p.parse_args()


def _sanitize(name: str) -> str:
    return str(name).replace(" ", "_").replace("-", "_").replace("/", "_").replace("(", "").replace(")", "")


def _pairwise_summary(
    pair_rows: Iterable[Dict],
    items: Sequence[str],
    left_key: str,
    right_key: str,
) -> Tuple[Dict[str, object], List[float]]:
    by_item: Dict[str, List[float]] = defaultdict(list)
    for row in pair_rows:
        try:
            value = float(row.get("fmd"))
        except Exception:
            continue
        left = row.get(left_key)
        right = row.get(right_key)
        if left in items:
            by_item[str(left)].append(value)
        if right in items:
            by_item[str(right)].append(value)

    summary: Dict[str, object] = {}
    means: List[float] = []
    for item in items:
        values = by_item.get(item, [])
        key = _sanitize(item)
        if values:
            mean_val = statistics.mean(values)
            summary[f"{key}_mean_fmd"] = mean_val
            summary[f"{key}_std_fmd"] = statistics.pstdev(values) if len(values) > 1 else 0.0
            summary[f"{key}_n_pairs"] = len(values)
            means.append(mean_val)
        else:
            summary[f"{key}_mean_fmd"] = None
            summary[f"{key}_std_fmd"] = None
            summary[f"{key}_n_pairs"] = 0

    if means:
        summary["mean_of_means"] = statistics.mean(means)
        summary["std_of_means"] = statistics.pstdev(means) if len(means) > 1 else 0.0
        summary["range_of_means"] = max(means) - min(means)
    else:
        summary["mean_of_means"] = None
        summary["std_of_means"] = None
        summary["range_of_means"] = None

    return summary, means


def _write_rows(path: Path, fieldnames: Sequence[str], rows: Sequence[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()

    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _append_row(path: Path, fieldnames: Sequence[str], row: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = path.exists()
    with open(path, "a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def _empty_tokenizer_row(song_label: str, midi_path: Path, selected_model: str, tokenizers: Sequence[str]) -> Dict[str, object]:
    row: Dict[str, object] = {
        "song_label": song_label,
        "midi_path": str(midi_path),
        "status": "insufficient_segments",
        "error": "",
        "selected_model": selected_model,
        "segments_built": None,
        "segments_used": 0,
    }
    for tok in tokenizers:
        key = _sanitize(tok)
        row[f"{key}_mean_fmd"] = None
        row[f"{key}_std_fmd"] = None
        row[f"{key}_n_pairs"] = 0
    row["mean_of_means"] = None
    row["std_of_means"] = None
    row["range_of_means"] = None
    return row


def _empty_model_row(song_label: str, midi_path: Path, selected_tokenizer: str, models: Sequence[str]) -> Dict[str, object]:
    row: Dict[str, object] = {
        "song_label": song_label,
        "midi_path": str(midi_path),
        "status": "insufficient_segments",
        "error": "",
        "selected_tokenizer": selected_tokenizer,
        "segments_built": None,
        "segments_used": 0,
    }
    for model in models:
        key = _sanitize(model)
        row[f"{key}_mean_fmd"] = None
        row[f"{key}_std_fmd"] = None
        row[f"{key}_n_pairs"] = 0
    row["mean_of_means"] = None
    row["std_of_means"] = None
    row["range_of_means"] = None
    return row


def _build_song_rows(
    runner: PaperExperimentRunner,
    midi_path: Path,
    tokenizers: Sequence[str],
    models: Sequence[str],
    selected_model: str,
    selected_tokenizer: str,
    n_segments: int,
) -> Dict[str, object]:
    """Build a single row comparing tokenizers for a fixed model (tokenizer CSV only)."""
    song_label = runner._segment_label(midi_path)
    base_variant = PipelineVariant(tokenizer=tokenizers[0], model=models[0], remove_velocity=False, hard_quantization=False)
    midi_data = runner._preprocess_midi_file(midi_path, base_variant)
    if midi_data is None:
        raise RuntimeError(f"Failed to preprocess MIDI file: {midi_path}")

    segments = runner._slice_midi_into_segments(midi_data, n_segments=n_segments)
    segment_indexes = [int(segment["index"]) for segment in segments]

    tokenizer_row = _empty_tokenizer_row(song_label, midi_path, selected_model, tokenizers)
    tokenizer_row["segments_built"] = len(segments)

    if len(segments) < 2:
        tokenizer_row["status"] = "insufficient_segments"
        return tokenizer_row

    token_maps: Dict[str, Dict[int, List[int]]] = {
        tokenizer_name: runner._tokenize_segments(segments, tokenizer_name) for tokenizer_name in tokenizers
    }

    # Tokenizer axis: fixed model, compare tokenizers on the segments they all share.
    common_indices = [
        idx for idx in segment_indexes if all(idx in token_maps[tokenizer_name] for tokenizer_name in tokenizers)
    ]
    tokenizer_row["segments_used"] = len(common_indices)
    if len(common_indices) >= 2:
        sequences_by_tokenizer = {
            tokenizer_name: [token_maps[tokenizer_name][idx] for idx in common_indices]
            for tokenizer_name in tokenizers
        }
        distributions = {
            tokenizer_name: runner.embeddings.extract_embeddings(sequences, selected_model)
            for tokenizer_name, sequences in sequences_by_tokenizer.items()
        }
        pair_rows, _ = runner._pairwise_fmd_rows(distributions, axis="tokenizer", group_name=selected_model)
        summary, means = _pairwise_summary(pair_rows, tokenizers, "left", "right")
        tokenizer_row.update(summary)
        tokenizer_row["status"] = "ok" if means else "no_pairwise"
    else:
        tokenizer_row["status"] = "insufficient_segments"

    return tokenizer_row


def main():
    args = parse_args()
    config = load_config("configs/config.yaml")
    setup_logging(config.get("logging", {}).get("level", "INFO"), config.get("logging", {}).get("log_file", "logs/experiment.log"))
    logger = get_logger("run_per_song_batch")

    base = Path(args.base_dir)
    folders = sorted(list(base.glob(args.pattern)))
    if not folders:
        logger.error(f"No folders matching pattern {args.pattern} under {base}")
        sys.exit(2)

    runner = PaperExperimentRunner(config)

    tokenizers = args.tokenizers or [t["type"] for t in config.get("tokenization", {}).get("tokenizers", [])]
    models = args.models or [m["name"] for m in config.get("embeddings", {}).get("models", [])]
    if not tokenizers:
        raise RuntimeError("No tokenizers configured")
    if not models:
        raise RuntimeError("No models configured")

    selected_model = args.model or models[0]
    selected_tokenizer = args.tokenizer or tokenizers[0]
    if selected_model not in models:
        raise RuntimeError(f"Selected model '{selected_model}' must be included in --models")
    if selected_tokenizer not in tokenizers:
        raise RuntimeError(f"Selected tokenizer '{selected_tokenizer}' must be included in --tokenizers")

    outdir = Path(args.output_dir) if args.output_dir else Path(config.get("results", {}).get("reports_dir", "results/reports")) / "per_song"
    outdir.mkdir(parents=True, exist_ok=True)

    tokenizer_csv = outdir / f"summary_tokenizer_for_model_{_sanitize(selected_model)}.csv"

    tokenizer_fieldnames = [
        "song_label",
        "midi_path",
        "status",
        "error",
        "selected_model",
        "segments_built",
        "segments_used",
    ]
    for tok in tokenizers:
        key = _sanitize(tok)
        tokenizer_fieldnames.extend([f"{key}_mean_fmd", f"{key}_std_fmd", f"{key}_n_pairs"])
    tokenizer_fieldnames.extend(["mean_of_means", "std_of_means", "range_of_means"])

    # Start fresh on each run; rows are appended song-by-song during processing.
    if tokenizer_csv.exists():
        tokenizer_csv.unlink()

    for folder in folders:
        midi_files = sorted(list(folder.glob("*.mid")) + list(folder.glob("*.midi")))
        if not midi_files:
            logger.warning(f"No MIDI files in {folder}, skipping")
            continue
        if args.all:
            to_run = midi_files
        else:
            to_run = midi_files[: max(1, args.n_per_genre)]
        for midi in to_run:
            try:
                logger.info(f"Processing {midi} ({folder.name})")
                # Silence noisy native outputs while building rows
                with suppress_stdout_stderr():
                    tokenizer_row = _build_song_rows(
                        runner=runner,
                        midi_path=midi,
                        tokenizers=tokenizers,
                        models=models,
                        selected_model=selected_model,
                        selected_tokenizer=selected_tokenizer,
                        n_segments=args.segments,
                    )
                logger.info(f"Finished {midi.name} -> status={tokenizer_row.get('status')}")
                _append_row(tokenizer_csv, tokenizer_fieldnames, tokenizer_row)
            except Exception as exc:
                tb = traceback.format_exc()
                # Log full traceback to configured logger (file + console)
                logger.exception(f"Failed on {midi}")
                song_label = runner._segment_label(midi)
                # Truncate traceback for CSV field to keep file readable
                short_tb = tb[-2000:] if len(tb) > 2000 else tb
                _append_row(
                    tokenizer_csv,
                    tokenizer_fieldnames,
                    _empty_tokenizer_row(song_label, midi, selected_model, tokenizers) | {"status": "failed", "error": short_tb},
                )

    print(f"Wrote: {tokenizer_csv}")


if __name__ == "__main__":
    main()
