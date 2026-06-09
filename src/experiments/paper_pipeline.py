"""Paper-oriented experiment pipeline for FMD sensitivity analysis."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from hashlib import md5
from itertools import combinations, product
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple
import csv
import json

import numpy as np
import yaml
from loguru import logger
from scipy.stats import kendalltau, spearmanr

try:
    from data.manager import DatasetManager
except Exception as _e:
    DatasetManager = None
    logger.warning(f"DatasetManager import failed: {_e}. Using fallback dataset manager.")

try:
    from data.lakh_genre_loader import LakhGenreLoader
except Exception as _e:
    LakhGenreLoader = None
    logger.warning(f"LakhGenreLoader import failed: {_e}. Lakh genre helper not available.")
try:
    from embeddings.extractor import EmbeddingExtractor
except Exception as _e:
    EmbeddingExtractor = None
    logger.warning(f"Embeddings extractor import failed: {_e}. Falling back to synthetic extractor.")
from metrics.fmd import FMDRanking, FrechetMusicDistance
from experiments.study_config import (
    MIDITOK_FORMAT,
    validate_embedding_input,
)
try:
    from preprocessing.processor import MIDIPreprocessor
except Exception as _e:
    MIDIPreprocessor = None
    logger.warning(f"MIDIPreprocessor import failed: {_e}. Using fallback preprocessor.")

try:
    from tokenization.tokenizer import TokenizationPipeline
except Exception as _e:
    TokenizationPipeline = None
    logger.warning(f"TokenizationPipeline import failed: {_e}. Using fallback tokenization pipeline.")


@dataclass(frozen=True)
class PipelineVariant:
    """Single pipeline configuration used in experiments."""

    model: str
    input_format: str
    tokenizer: str | None
    remove_velocity: bool
    hard_quantization: bool

    @property
    def name(self) -> str:
        """Compact string label for reports."""
        label = self.input_format if self.tokenizer is None else self.tokenizer
        return (
            f"input={label}|model={self.model}|"
            f"vel={'off' if self.remove_velocity else 'on'}|"
            f"quant={'on' if self.hard_quantization else 'off'}"
        )


class PaperExperimentRunner:
    """Runs comparisons needed for an article draft."""

    def __init__(self, config: Dict):
        self.config = config
        paper_cfg = config.get("paper", {})
        fallback_mode = str(paper_cfg.get("fallback_mode", "synthetic")).strip().lower()
        strict_runtime = fallback_mode in {"strict", "hard_strict"}

        # Dataset manager (fallback if project-specific manager not available)
        if DatasetManager is not None:
            try:
                self.dataset_manager = DatasetManager(config)
            except Exception as _e:
                if strict_runtime:
                    raise RuntimeError("DatasetManager is required in strict paper mode") from _e
                logger.warning(f"Failed to initialize DatasetManager: {_e}; using fallback dataset manager.")
                self.dataset_manager = self._fallback_dataset_manager()
        else:
            if strict_runtime:
                raise RuntimeError("DatasetManager import failed in strict paper mode")
            logger.warning("DatasetManager not available; using fallback dataset manager.")
            self.dataset_manager = self._fallback_dataset_manager()

        # Preprocessor (fallback if pretty_midi or project preprocessor is missing)
        if MIDIPreprocessor is not None:
            try:
                self.preprocessor = MIDIPreprocessor(config)
            except Exception as _e:
                if strict_runtime:
                    raise RuntimeError("MIDIPreprocessor is required in strict paper mode") from _e
                logger.warning(f"Failed to initialize MIDIPreprocessor: {_e}; using fallback preprocessor.")
                self.preprocessor = self._fallback_preprocessor()
        else:
            if strict_runtime:
                raise RuntimeError("MIDIPreprocessor import failed in strict paper mode")
            logger.warning("MIDIPreprocessor not available; using fallback preprocessor.")
            self.preprocessor = self._fallback_preprocessor()

        # Tokenization pipeline (fallback to simple deterministic tokenizers)
        if TokenizationPipeline is not None:
            try:
                self.tokenization = TokenizationPipeline(config)
            except Exception as _e:
                if strict_runtime:
                    raise RuntimeError("TokenizationPipeline is required in strict paper mode") from _e
                logger.warning(f"Failed to initialize TokenizationPipeline: {_e}; using fallback tokenization.")
                self.tokenization = self._fallback_tokenization_pipeline()
        else:
            if strict_runtime:
                raise RuntimeError("TokenizationPipeline import failed in strict paper mode")
            logger.warning("TokenizationPipeline not available; using fallback tokenization pipeline.")
            self.tokenization = self._fallback_tokenization_pipeline()

        # Embeddings extractor may require heavy dependencies (torch, transformers).
        # If the real extractor is not importable, use a deterministic synthetic fallback.
        if EmbeddingExtractor is not None:
            try:
                self.embeddings = EmbeddingExtractor(config)
            except Exception as _e:
                if strict_runtime:
                    raise RuntimeError("EmbeddingExtractor is required in strict paper mode") from _e
                logger.warning(f"Failed to initialize EmbeddingExtractor: {_e}; using synthetic fallback.")
                self.embeddings = self._synthetic_fallback_extractor()
        else:
            if strict_runtime:
                raise RuntimeError("EmbeddingExtractor import failed in strict paper mode")
            logger.warning("EmbeddingExtractor not available; using synthetic fallback extractor.")
            self.embeddings = self._synthetic_fallback_extractor()

        self.fmd = FrechetMusicDistance(config)

        self.max_files = int(paper_cfg.get("max_files_per_dataset", 8))
        self.synthetic_fallback_samples = int(paper_cfg.get("synthetic_fallback_samples", 12))
        self.seed = int(paper_cfg.get("seed", 42))
        self.dataset_names = self._dataset_names()
        self.compare_all_pairs = bool(paper_cfg.get("compare_all_pairs", False))
        self.fallback_mode = fallback_mode
        self.progress_log_every = int(paper_cfg.get("progress_log_every", 25))
        self.bootstrap_cfg = paper_cfg.get("bootstrap_ci", {})
        self.bootstrap_enabled = bool(self.bootstrap_cfg.get("enabled", True))
        self.bootstrap_resamples = int(self.bootstrap_cfg.get("n_resamples", 50))
        self.bootstrap_ci = float(self.bootstrap_cfg.get("confidence", 0.95))
        self.bootstrap_seed = int(self.bootstrap_cfg.get("seed", self.seed))

        self.genre_aliases = self._load_genre_aliases(paper_cfg)
        self.top_variants_per_pair = int(paper_cfg.get("top_variants_per_pair", 3))

        reports_dir = config.get("results", {}).get("reports_dir", "results/reports")
        self.output_dir = Path(reports_dir) / "paper"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def build_variants(
        self,
        tokenizers: Sequence[str] | None = None,
        models: Sequence[str] | None = None,
        preprocessing_grid: Sequence[Tuple[bool, bool]] | None = None,
    ) -> List[PipelineVariant]:
        """Build experiment grid tokenizer x model x preprocessing."""
        if tokenizers is None:
            tokenizers = [t["type"] for t in self.config["tokenization"]["tokenizers"]]
        if models is None:
            models = [m["name"] for m in self.config["embeddings"]["models"]]
        if preprocessing_grid is None:
            preprocessing_grid = [(False, False), (True, False), (False, True), (True, True)]

        variants: List[PipelineVariant] = []
        for model, (remove_vel, hard_quant) in product(models, preprocessing_grid):
            if model in {"CLaMP-1", "CLaMP-2"}:
                spec = validate_embedding_input(model=model)
                variants.append(
                    PipelineVariant(
                        model=spec.model,
                        input_format=spec.input_format,
                        tokenizer=spec.tokenizer,
                        remove_velocity=remove_vel,
                        hard_quantization=hard_quant,
                    )
                )
                continue

            for tok in tokenizers:
                spec = validate_embedding_input(model=model, input_format=MIDITOK_FORMAT, tokenizer=tok)
                variants.append(
                    PipelineVariant(
                        model=spec.model,
                        input_format=spec.input_format,
                        tokenizer=spec.tokenizer,
                        remove_velocity=remove_vel,
                        hard_quantization=hard_quant,
                    )
                )
        logger.info(f"Built {len(variants)} variants")
        return variants

    @staticmethod
    def _parse_pairs(raw_pairs: Iterable) -> List[Tuple[str, str]]:
        """Parse YAML pairs that can be tuple-like strings or arrays."""
        pairs: List[Tuple[str, str]] = []
        for entry in raw_pairs:
            if isinstance(entry, (list, tuple)) and len(entry) == 2:
                pairs.append((str(entry[0]), str(entry[1])))
                continue

            text = str(entry).strip().strip("()")
            if "," not in text:
                continue
            left, right = [p.strip().strip("'\"") for p in text.split(",", 1)]
            if left and right:
                pairs.append((left, right))
        return pairs

    def _dataset_names(self) -> List[str]:
        return [d["name"] for d in self.config["data"]["datasets"]]

    def _load_genre_aliases(self, paper_cfg: Dict) -> Dict[str, str]:
        """Load alias mapping from config and optional standalone YAML file."""
        aliases: Dict[str, str] = {}

        inline_aliases = paper_cfg.get("genre_aliases", {})
        if isinstance(inline_aliases, dict):
            aliases.update(
                {
                    str(alias).strip().lower(): str(dataset).strip()
                    for alias, dataset in inline_aliases.items()
                    if str(alias).strip() and str(dataset).strip()
                }
            )

        mapping_file = paper_cfg.get("genre_mapping_file")
        if not mapping_file:
            return aliases

        mapping_path = Path(mapping_file)
        if not mapping_path.exists():
            logger.warning(f"Genre mapping file not found: {mapping_path}")
            return aliases

        with open(mapping_path, "r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle) or {}

        file_aliases = payload.get("aliases", payload)
        if isinstance(file_aliases, dict):
            for alias, dataset in file_aliases.items():
                alias_text = str(alias).strip().lower()
                dataset_text = str(dataset).strip()
                if alias_text and dataset_text:
                    aliases[alias_text] = dataset_text

        return aliases

    def _resolve_dataset_name(self, name_or_alias: str) -> str | None:
        label = str(name_or_alias).strip()
        if not label:
            return None

        label_lower = label.lower()
        if label in self.dataset_names:
            return label
        if label_lower in self.genre_aliases:
            resolved = self.genre_aliases[label_lower]
            return resolved if resolved in self.dataset_names else None
        if label_lower in [x.lower() for x in self.dataset_names]:
            by_lower = {x.lower(): x for x in self.dataset_names}
            return by_lower[label_lower]
        return None

    def _resolve_special_pairs(self) -> List[Dict]:
        paper_cfg = self.config.get("paper", {})
        parsed = self._parse_pairs(paper_cfg.get("special_pairs", []))

        resolved: List[Dict] = []
        for left, right in parsed:
            ds_left = self._resolve_dataset_name(left)
            ds_right = self._resolve_dataset_name(right)
            if not ds_left or not ds_right:
                logger.warning(f"Skipping special pair ({left}, {right}) - unresolved dataset alias")
                continue
            resolved.append(
                {
                    "genre_a": left,
                    "genre_b": right,
                    "dataset_a": ds_left,
                    "dataset_b": ds_right,
                }
            )

        return resolved

    def _list_dataset_midis(self, dataset_name: str) -> List[Path]:
        files = self.dataset_manager.list_midi_files(dataset_name, processed=False, limit=self.max_files)
        if files:
            return files

        # Fallback: some repos keep MIDI files directly in data/raw.
        raw_root = Path(self.config["data"]["raw_data_dir"])
        flat = sorted(list(raw_root.glob("*.mid")) + list(raw_root.glob("*.midi")))
        return flat[: self.max_files]

    def _preprocess_midi_file(self, midi_path: Path, variant: PipelineVariant):
        midi_data = self.preprocessor.load_midi(midi_path)
        if midi_data is None:
            return None
        if variant.remove_velocity:
            midi_data = self.preprocessor.remove_velocity(midi_data)
        if variant.hard_quantization:
            midi_data = self.preprocessor.quantize_time(midi_data)
        midi_data = self.preprocessor.filter_note_range(midi_data)
        midi_data = self.preprocessor.normalize_instruments(midi_data)
        return midi_data

    def _tokens_for_variant(self, midi_data, variant: PipelineVariant) -> List[int]:
        if variant.input_format != MIDITOK_FORMAT:
            return []
        if not variant.tokenizer:
            raise ValueError(f"Variant {variant.name} requires a MidiTok tokenizer")
        tokenizer = self.tokenization.tokenizers[variant.tokenizer]
        return tokenizer.encode_midi_object(midi_data)

    def _miditok_tokenizer_for_variant(self, variant: PipelineVariant):
        if variant.input_format != MIDITOK_FORMAT or not variant.tokenizer:
            return None
        tokenizer = self.tokenization.tokenizers[variant.tokenizer]
        return getattr(tokenizer, "miditok_tokenizer", None)

    def _extract_variant_embeddings(
        self,
        token_sequences: Sequence[List[int]],
        midi_data_list: Sequence | None,
        variant: PipelineVariant,
    ) -> np.ndarray:
        input_formats = [variant.input_format] * len(token_sequences)
        return self.embeddings.extract_embeddings(
            token_sequences=list(token_sequences),
            model_name=variant.model,
            midi_data_list=list(midi_data_list) if midi_data_list is not None else None,
            input_formats=input_formats,
            miditok_tokenizer=self._miditok_tokenizer_for_variant(variant),
        )

    @staticmethod
    def _segment_label(path: Path) -> str:
        text = path.stem.strip().lower()
        safe = []
        for ch in text:
            if ch.isalnum() or ch in {"-", "_"}:
                safe.append(ch)
            else:
                safe.append("_")
        label = "".join(safe).strip("_")
        return label or "song"

    @staticmethod
    def _slice_midi_into_segments(midi_data, n_segments: int) -> List[Dict]:
        """Slice midi_data into segments. Works with pretty_midi when available,
        otherwise uses a lightweight SimpleNamespace-based representation compatible
        with fallback tokenizers.
        """
        try:
            import pretty_midi
            using_pretty_midi = True
        except Exception:
            pretty_midi = None
            using_pretty_midi = False

        notes = []
        for instrument in getattr(midi_data, "instruments", []):
            notes.extend(getattr(instrument, "notes", []))

        if not notes:
            return []

        end_time = max(getattr(note, "end", 0) for note in notes)
        if end_time <= 0:
            return []

        n_segments = max(1, int(n_segments))
        edges = np.linspace(0.0, float(end_time), num=n_segments + 1)
        segments: List[Dict] = []

        from types import SimpleNamespace

        for idx in range(len(edges) - 1):
            start = float(edges[idx])
            end = float(edges[idx + 1])
            note_count = 0

            if using_pretty_midi:
                segment = pretty_midi.PrettyMIDI()
                for instrument in midi_data.instruments:
                    seg_inst = pretty_midi.Instrument(
                        program=getattr(instrument, "program", 0),
                        is_drum=getattr(instrument, "is_drum", False),
                        name=getattr(instrument, "name", ""),
                    )
                    for note in getattr(instrument, "notes", []):
                        if getattr(note, "end", 0) <= start or getattr(note, "start", 0) >= end:
                            continue
                        clipped_start = max(getattr(note, "start", 0), start) - start
                        clipped_end = min(getattr(note, "end", 0), end) - start
                        if clipped_end <= clipped_start:
                            continue
                        seg_inst.notes.append(
                            pretty_midi.Note(
                                velocity=int(getattr(note, "velocity", 64)),
                                pitch=int(getattr(note, "pitch", 60)),
                                start=float(clipped_start),
                                end=float(clipped_end),
                            )
                        )
                    if seg_inst.notes:
                        note_count += len(seg_inst.notes)
                        segment.instruments.append(seg_inst)

                if note_count > 0:
                    segments.append(
                        {
                            "index": idx,
                            "start": start,
                            "end": end,
                            "note_count": note_count,
                            "midi": segment,
                        }
                    )

            else:
                seg_instruments = []
                for instrument in getattr(midi_data, "instruments", []):
                    seg_inst_notes = []
                    for note in getattr(instrument, "notes", []):
                        nstart = getattr(note, "start", 0)
                        nend = getattr(note, "end", 0)
                        if nend <= start or nstart >= end:
                            continue
                        clipped_start = max(nstart, start) - start
                        clipped_end = min(nend, end) - start
                        if clipped_end <= clipped_start:
                            continue
                        pitch = int(getattr(note, "pitch", 60))
                        velocity = int(getattr(note, "velocity", 64))
                        seg_note = SimpleNamespace(
                            start=float(clipped_start),
                            end=float(clipped_end),
                            pitch=pitch,
                            velocity=velocity,
                        )
                        seg_inst_notes.append(seg_note)
                    if seg_inst_notes:
                        note_count += len(seg_inst_notes)
                        seg_inst = SimpleNamespace(
                            notes=seg_inst_notes,
                            program=getattr(instrument, "program", 0),
                            is_drum=getattr(instrument, "is_drum", False),
                            name=getattr(instrument, "name", "fallback"),
                        )
                        seg_instruments.append(seg_inst)

                if note_count > 0:
                    segment = SimpleNamespace(instruments=seg_instruments)
                    segments.append(
                        {
                            "index": idx,
                            "start": start,
                            "end": end,
                            "note_count": note_count,
                            "midi": segment,
                        }
                    )

        return segments

    def _synthetic_fallback_extractor(self):
        """Create a deterministic synthetic embedding extractor used when real models are unavailable."""
        import numpy as _np
        import hashlib as _hash

        fallback_dim = int(self.config.get("embeddings", {}).get("fallback_dim", 768))

        class _FallbackExtractor:
            def __init__(self, dim: int):
                self.dim = int(dim)

            def extract_embeddings(self, token_sequences: Sequence[Sequence[int]], model_name: str = "", **_kwargs):
                embs = []
                for seq in token_sequences:
                    try:
                        key = (",".join(map(str, seq)) if seq else "") + "|" + str(model_name)
                    except Exception:
                        # Fallback if tokens are not iterable
                        key = str(seq) + "|" + str(model_name)
                    seed = int(_hash.md5(key.encode("utf-8")).hexdigest()[:8], 16)
                    rng = _np.random.default_rng(seed)
                    emb = rng.standard_normal(self.dim).astype(_np.float32)
                    embs.append(emb)
                if not embs:
                    return _np.zeros((0, self.dim), dtype=_np.float32)
                return _np.stack(embs, axis=0)

        return _FallbackExtractor(fallback_dim)

    def _fallback_dataset_manager(self):
        from pathlib import Path

        class _FallbackManager:
            def __init__(self, cfg):
                self.raw_root = Path(cfg.get("data", {}).get("raw_data_dir", "data/raw"))

            def list_midi_files(self, dataset_name, processed=False, limit=None):
                ds_dir = self.raw_root / dataset_name
                files = []
                if ds_dir.exists():
                    files = sorted(list(ds_dir.glob("*.mid")) + list(ds_dir.glob("*.midi")))
                else:
                    files = sorted(list(self.raw_root.glob("*.mid")) + list(self.raw_root.glob("*.midi")))
                if limit:
                    return files[:limit]
                return files

        return _FallbackManager(self.config)

    def _fallback_preprocessor(self):
        import hashlib
        from types import SimpleNamespace

        class _FallbackPreprocessor:
            def __init__(self, cfg):
                self.cfg = cfg

            def load_midi(self, path):
                try:
                    data = open(path, "rb").read()
                    seed = int(hashlib.md5(data).hexdigest()[:8], 16)
                except Exception:
                    seed = 0
                rng = __import__("numpy").random.default_rng(seed)
                duration = float(rng.integers(3, 61))
                n_notes = int(rng.integers(4, 32))

                class Note:
                    def __init__(self, start, end, pitch=60, velocity=64):
                        self.start = float(start)
                        self.end = float(end)
                        self.pitch = int(pitch)
                        self.velocity = int(velocity)

                class Instrument:
                    def __init__(self):
                        self.notes = []
                        self.program = 0
                        self.is_drum = False
                        self.name = "fallback"

                inst = Instrument()
                step = max(0.05, duration / max(1, n_notes))
                for i in range(n_notes):
                    s = i * step
                    e = min(duration, s + min(0.5, step))
                    pitch = int(60 + (rng.integers(-12, 13)))
                    vel = int(40 + (rng.integers(0, 88)))
                    inst.notes.append(Note(s, e, pitch, vel))

                midi = SimpleNamespace(instruments=[inst])
                return midi

            def remove_velocity(self, midi):
                return midi

            def quantize_time(self, midi):
                return midi

            def filter_note_range(self, midi):
                return midi

            def normalize_instruments(self, midi):
                return midi

        return _FallbackPreprocessor(self.config)

    def _fallback_tokenization_pipeline(self):
        import hashlib as _hash

        class _FallbackTokenizer:
            def __init__(self, name):
                self.name = name

            def encode_midi_object(self, midi_data):
                tokens = []
                for inst in getattr(midi_data, "instruments", []):
                    for note in getattr(inst, "notes", []):
                        s = int(round(note.start * 1000))
                        e = int(round(note.end * 1000))
                        p = int(getattr(note, "pitch", 60))
                        val = ((s * 73856093) ^ (e * 19349663) ^ (p * 83492791)) & 0xFFFFFFFF
                        tokens.append(int(val))
                if not tokens:
                    tokens = [int(len(str(midi_data)))]
                return tokens

            def get_vocab_size(self):
                return 128

        class _FallbackPipeline:
            def __init__(self, cfg):
                tok_names = [t["type"] for t in cfg.get("tokenization", {}).get("tokenizers", [{"type":"REMI"},{"type":"TSD"},{"type":"Octuple"},{"type":"MIDI-Like"}])]
                self.tokenizers = {name: _FallbackTokenizer(name) for name in tok_names}

        return _FallbackPipeline(self.config)

    def _tokenize_segments(self, segments: List[Dict], tokenizer_name: str) -> Dict[int, List[int]]:
        tokenizer = self.tokenization.tokenizers[tokenizer_name]
        tokens_by_index: Dict[int, List[int]] = {}
        for segment in segments:
            tokens = tokenizer.encode_midi_object(segment["midi"])
            if tokens:
                tokens_by_index[int(segment["index"])] = tokens
        return tokens_by_index

    def _pairwise_fmd_rows(
        self,
        distributions: Dict[str, np.ndarray],
        axis: str,
        group_name: str,
    ) -> Tuple[List[Dict], List[Dict]]:
        rows: List[Dict] = []
        summary: List[Dict] = []
        names = sorted(distributions.keys())
        values = []

        for left, right in combinations(names, 2):
            emb_left = distributions[left]
            emb_right = distributions[right]
            try:
                emb_left_arr = np.atleast_2d(emb_left).astype(np.float64)
                emb_right_arr = np.atleast_2d(emb_right).astype(np.float64)
                d_left = emb_left_arr.shape[1] if emb_left_arr.size else 0
                d_right = emb_right_arr.shape[1] if emb_right_arr.size else 0
                if d_left != d_right:
                    raise ValueError(
                        f"Cannot compute FMD across different embedding dimensions: {d_left} vs {d_right}. "
                        "Do not use SVD/zero-padding normalization for paper results."
                    )
                if d_left <= 0:
                    raise ValueError("Cannot compute FMD for empty embedding dimensions")

                fmd_value = float(self.fmd.compute_fmd(emb_left_arr, emb_right_arr))

                rows.append(
                    {
                        "axis": axis,
                        "group": group_name,
                        "left": left,
                        "right": right,
                        "fmd": fmd_value,
                        "n_left": int(emb_left_arr.shape[0]),
                        "n_right": int(emb_right_arr.shape[0]),
                    }
                )
                values.append(fmd_value)
            except Exception as e:
                # Log and continue; record a row with fmd=None so downstream aggregation can proceed.
                logger.warning(f"Failed to compute FMD for {group_name} ({left} vs {right}): {e}")
                try:
                    n_left = int(getattr(emb_left, 'shape', [0])[0])
                except Exception:
                    n_left = 0
                try:
                    n_right = int(getattr(emb_right, 'shape', [0])[0])
                except Exception:
                    n_right = 0
                rows.append(
                    {
                        "axis": axis,
                        "group": group_name,
                        "left": left,
                        "right": right,
                        "fmd": None,
                        "n_left": n_left,
                        "n_right": n_right,
                    }
                )
                # do not append to 'values'
                continue

        if values:
            summary.append(
                {
                    "axis": axis,
                    "group": group_name,
                    "n_items": len(names),
                    "n_pairs": len(values),
                    "mean_fmd": float(np.mean(values)),
                    "min_fmd": float(np.min(values)),
                    "max_fmd": float(np.max(values)),
                    "std_fmd": float(np.std(values)),
                }
            )
        else:
            summary.append(
                {
                    "axis": axis,
                    "group": group_name,
                    "n_items": len(names),
                    "n_pairs": 0,
                    "mean_fmd": None,
                    "min_fmd": None,
                    "max_fmd": None,
                    "std_fmd": None,
                }
            )

        return rows, summary

    def run_single_song_analysis(
        self,
        midi_path: Path,
        tokenizers: Sequence[str] | None = None,
        models: Sequence[str] | None = None,
        n_segments: int = 8,
        remove_velocity: bool = False,
        hard_quantization: bool = False,
        axis: str = "both",
        output_dir: Path | None = None,
    ) -> Dict:
        """Analyze one MIDI file by comparing FMD across tokenizers and/or models.

        The song is first preprocessed once, then split into time windows.
        For tokenization analysis, every selected tokenizer must succeed on a
        segment for that segment to be kept. For model analysis, the tokenizer
        is fixed and the resulting token sequences are embedded by each model.
        """
        midi_path = Path(midi_path)
        if not midi_path.exists():
            raise FileNotFoundError(f"MIDI file not found: {midi_path}")

        if axis not in {"tokenizer", "model", "both"}:
            raise ValueError("axis must be one of: tokenizer, model, both")

        all_tokenizers = tokenizers or [t["type"] for t in self.config["tokenization"]["tokenizers"]]
        all_models = models or [m["name"] for m in self.config["embeddings"]["models"]]
        if not all_tokenizers:
            raise ValueError("No tokenizers selected for per-song analysis")
        if not all_models:
            raise ValueError("No models selected for per-song analysis")

        first_spec = validate_embedding_input(
            all_models[0],
            tokenizer=None if all_models[0] in {"CLaMP-1", "CLaMP-2"} else all_tokenizers[0],
        )
        base_variant = PipelineVariant(
            model=first_spec.model,
            input_format=first_spec.input_format,
            tokenizer=first_spec.tokenizer,
            remove_velocity=remove_velocity,
            hard_quantization=hard_quantization,
        )
        midi_data = self._preprocess_midi_file(midi_path, base_variant)
        if midi_data is None:
            raise RuntimeError(f"Failed to load or preprocess MIDI file: {midi_path}")

        segments = self._slice_midi_into_segments(midi_data, n_segments=n_segments)
        if len(segments) < 2:
            raise RuntimeError(
                f"Not enough usable segments in {midi_path} for per-song analysis (got {len(segments)})"
            )
        segment_by_index = {int(segment["index"]): segment for segment in segments}

        if output_dir is None:
            output_dir = Path(self.config.get("results", {}).get("reports_dir", "results/reports")) / "per_song"
        output_dir = Path(output_dir) / self._segment_label(midi_path)
        output_dir.mkdir(parents=True, exist_ok=True)

        tokenizer_section: Dict | None = None
        model_section: Dict | None = None
        segment_rows: List[Dict] = []

        if axis in {"tokenizer", "both"}:
            tokenizer_rows: List[Dict] = []
            tokenizer_summary: List[Dict] = []
            tokenizer_segment_stats: List[Dict] = []

            token_maps: Dict[str, Dict[int, List[int]]] = {
                tokenizer_name: self._tokenize_segments(segments, tokenizer_name)
                for tokenizer_name in all_tokenizers
            }
            common_indices = [
                int(segment["index"])
                for segment in segments
                if all(int(segment["index"]) in token_maps[tokenizer_name] for tokenizer_name in all_tokenizers)
            ]

            if len(common_indices) < 2:
                logger.warning(
                    f"Tokenizer axis has only {len(common_indices)} common segments across selected tokenizers"
                )
            else:
                for model_name in all_models:
                    spec = validate_embedding_input(model_name, tokenizer=all_tokenizers[0])
                    if spec.input_format != MIDITOK_FORMAT:
                        logger.warning(
                            f"Skipping tokenizer axis for {model_name}: model-native {spec.input_format} "
                            "does not consume MidiTok tokenizers"
                        )
                        continue
                    distributions: Dict[str, np.ndarray] = {}
                    for tokenizer_name in all_tokenizers:
                        seqs = [token_maps[tokenizer_name][idx] for idx in common_indices]
                        variant = PipelineVariant(
                            model=model_name,
                            input_format=MIDITOK_FORMAT,
                            tokenizer=tokenizer_name,
                            remove_velocity=remove_velocity,
                            hard_quantization=hard_quantization,
                        )
                        midi_segments = [segment_by_index[idx]["midi"] for idx in common_indices]
                        embeddings = self._extract_variant_embeddings(seqs, midi_segments, variant)
                        distributions[tokenizer_name] = embeddings
                        tokenizer_segment_stats.extend(
                            [
                                {
                                    "axis": "tokenizer",
                                    "group": model_name,
                                    "tokenizer": tokenizer_name,
                                    "segment_index": idx,
                                    "token_count": len(token_maps[tokenizer_name][idx]),
                                    "embedding_norm": float(np.linalg.norm(embeddings[pos])),
                                }
                                for pos, idx in enumerate(common_indices)
                            ]
                        )

                    rows, summary = self._pairwise_fmd_rows(distributions, axis="tokenizer", group_name=model_name)
                    tokenizer_rows.extend(rows)
                    tokenizer_summary.extend(
                        [
                            {
                                **item,
                                "model": model_name,
                                "tokenizers": ",".join(all_tokenizers),
                                "segments_used": len(common_indices),
                            }
                            for item in summary
                        ]
                    )

            tokenizer_section = {
                "available": bool(tokenizer_rows),
                "rows": tokenizer_rows,
                "summary": tokenizer_summary,
                "segment_stats": tokenizer_segment_stats,
            }

        if axis in {"model", "both"}:
            model_rows: List[Dict] = []
            model_summary: List[Dict] = []
            model_segment_stats: List[Dict] = []

            for tokenizer_name in all_tokenizers:
                token_maps = {tokenizer_name: self._tokenize_segments(segments, tokenizer_name)}
                common_indices = [
                    int(segment["index"])
                    for segment in segments
                    if int(segment["index"]) in token_maps[tokenizer_name]
                ]

                if len(common_indices) < 2:
                    logger.warning(
                        f"Model axis for tokenizer={tokenizer_name} has only {len(common_indices)} valid segments"
                    )
                    continue

                token_sequences = [token_maps[tokenizer_name][idx] for idx in common_indices]
                distributions: Dict[str, np.ndarray] = {}
                for model_name in all_models:
                    try:
                        spec = validate_embedding_input(
                            model_name,
                            tokenizer=None if model_name in {"CLaMP-1", "CLaMP-2"} else tokenizer_name,
                        )
                    except ValueError as exc:
                        logger.warning(f"Skipping incompatible model axis cell: {exc}")
                        continue
                    variant = PipelineVariant(
                        model=model_name,
                        input_format=spec.input_format,
                        tokenizer=spec.tokenizer,
                        remove_velocity=remove_velocity,
                        hard_quantization=hard_quantization,
                    )
                    midi_segments = [segment_by_index[idx]["midi"] for idx in common_indices]
                    seqs = token_sequences if spec.input_format == MIDITOK_FORMAT else [[] for _ in common_indices]
                    embeddings = self._extract_variant_embeddings(seqs, midi_segments, variant)
                    distributions[model_name] = embeddings
                    model_segment_stats.extend(
                        [
                            {
                                "axis": "model",
                                "group": tokenizer_name,
                                "model": model_name,
                                "segment_index": idx,
                                "token_count": len(token_maps[tokenizer_name][idx]),
                                "embedding_norm": float(np.linalg.norm(embeddings[pos])),
                            }
                            for pos, idx in enumerate(common_indices)
                        ]
                    )

                rows, summary = self._pairwise_fmd_rows(distributions, axis="model", group_name=tokenizer_name)
                model_rows.extend(rows)
                model_summary.extend(
                    [
                        {
                            **item,
                            "tokenizer": tokenizer_name,
                            "models": ",".join(all_models),
                            "segments_used": len(common_indices),
                        }
                        for item in summary
                    ]
                )

            model_section = {
                "available": bool(model_rows),
                "rows": model_rows,
                "summary": model_summary,
                "segment_stats": model_segment_stats,
            }

        csv_tokenizer = output_dir / "per_song_tokenizer_fmd.csv"
        csv_model = output_dir / "per_song_model_fmd.csv"
        csv_segment = output_dir / "per_song_segment_stats.csv"
        csv_tokenizer_stats = output_dir / "per_song_tokenizer_stats.csv"
        csv_model_stats = output_dir / "per_song_model_stats.csv"
        md_path = output_dir / "per_song_summary.md"
        json_path = output_dir / "per_song_summary.json"

        all_segment_rows = []
        if tokenizer_section:
            all_segment_rows.extend(tokenizer_section["segment_stats"])
        if model_section:
            all_segment_rows.extend(model_section["segment_stats"])
        segment_rows.extend(all_segment_rows)

        with open(csv_segment, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["axis", "group", "tokenizer", "model", "segment_index", "token_count", "embedding_norm"],
            )
            writer.writeheader()
            writer.writerows(segment_rows)

        # Write tokenizer pairwise FMDs and compute per-tokenizer summary stats
        tokenizer_stats: List[Dict] = []
        if tokenizer_section:
            with open(csv_tokenizer, "w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=["axis", "group", "left", "right", "fmd", "n_left", "n_right"],
                )
                writer.writeheader()
                writer.writerows(tokenizer_section["rows"])

            try:
                trows = [r for r in tokenizer_section.get("rows", []) if r.get("fmd") is not None]
                if trows:
                    fmd_by_tok: Dict[str, List[float]] = defaultdict(list)
                    for r in trows:
                        left = r.get("left")
                        right = r.get("right")
                        fmd_val = float(r.get("fmd"))
                        fmd_by_tok[left].append(fmd_val)
                        fmd_by_tok[right].append(fmd_val)

                    stats: List[Dict] = []
                    for tok, vals in fmd_by_tok.items():
                        arr = np.array(vals, dtype=float)
                        stats.append(
                            {
                                "tokenizer": tok,
                                "count_pairs": int(arr.size),
                                "mean_fmd": float(np.mean(arr)),
                                "std_fmd": float(np.std(arr)),
                                "min_fmd": float(np.min(arr)),
                                "max_fmd": float(np.max(arr)),
                            }
                        )

                    if stats:
                        min_mean = float(min(s["mean_fmd"] for s in stats))
                        for s in stats:
                            s["delta_vs_min"] = float(s["mean_fmd"] - min_mean)
                        stats.sort(key=lambda x: x["mean_fmd"], reverse=True)
                        tokenizer_stats = stats
            except Exception as exc:
                logger.warning(f"Failed to compute tokenizer stats for {midi_path}: {exc}")

            if tokenizer_stats:
                with open(csv_tokenizer_stats, "w", newline="", encoding="utf-8") as handle:
                    fieldnames = ["tokenizer", "count_pairs", "mean_fmd", "std_fmd", "min_fmd", "max_fmd", "delta_vs_min"]
                    writer = csv.DictWriter(handle, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(tokenizer_stats)
            else:
                csv_tokenizer_stats.touch()
        else:
            csv_tokenizer.touch()
            csv_tokenizer_stats.touch()

        # Write model pairwise FMDs and compute per-model summary stats
        model_stats: List[Dict] = []
        if model_section:
            with open(csv_model, "w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=["axis", "group", "left", "right", "fmd", "n_left", "n_right"],
                )
                writer.writeheader()
                writer.writerows(model_section["rows"])

            try:
                mrows = [r for r in model_section.get("rows", []) if r.get("fmd") is not None]
                if mrows:
                    fmd_by_model: Dict[str, List[float]] = defaultdict(list)
                    for r in mrows:
                        left = r.get("left")
                        right = r.get("right")
                        fmd_val = float(r.get("fmd"))
                        fmd_by_model[left].append(fmd_val)
                        fmd_by_model[right].append(fmd_val)

                    stats_m: List[Dict] = []
                    for model_name, vals in fmd_by_model.items():
                        arr = np.array(vals, dtype=float)
                        stats_m.append(
                            {
                                "model": model_name,
                                "count_pairs": int(arr.size),
                                "mean_fmd": float(np.mean(arr)),
                                "std_fmd": float(np.std(arr)),
                                "min_fmd": float(np.min(arr)),
                                "max_fmd": float(np.max(arr)),
                            }
                        )

                    if stats_m:
                        min_mean_m = float(min(s["mean_fmd"] for s in stats_m))
                        for s in stats_m:
                            s["delta_vs_min"] = float(s["mean_fmd"] - min_mean_m)
                        stats_m.sort(key=lambda x: x["mean_fmd"], reverse=True)
                        model_stats = stats_m
            except Exception as exc:
                logger.warning(f"Failed to compute model stats for {midi_path}: {exc}")

            if model_stats:
                with open(csv_model_stats, "w", newline="", encoding="utf-8") as handle:
                    fieldnames = ["model", "count_pairs", "mean_fmd", "std_fmd", "min_fmd", "max_fmd", "delta_vs_min"]
                    writer = csv.DictWriter(handle, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(model_stats)
            else:
                csv_model_stats.touch()
        else:
            csv_model.touch()
            csv_model_stats.touch()

        # Combined CSVs: one row per model with per-tokenizer means, and one row per tokenizer with per-model means.
        csv_tok_by_model = output_dir / "per_song_tokenizer_by_model.csv"
        csv_model_by_tok = output_dir / "per_song_model_by_tokenizer.csv"

        def _sanitize(name: str) -> str:
            return str(name).replace(" ", "_").replace("-", "_").replace("/", "_").replace("(", "").replace(")", "")

        # Build per-model rows (each row: model, segments_used, <tokenizer>_mean_fmd, <tokenizer>_std_fmd, <tokenizer>_delta_vs_min, ...)
        tok_by_model_rows = []
        if tokenizer_section:
            for model_name in all_models:
                model_rows = [r for r in tokenizer_section.get("rows", []) if r.get("group") == model_name and r.get("fmd") is not None]
                fmd_by_tok: Dict[str, List[float]] = defaultdict(list)
                for r in model_rows:
                    left = r.get("left")
                    right = r.get("right")
                    try:
                        v = float(r.get("fmd"))
                    except Exception:
                        continue
                    fmd_by_tok[left].append(v)
                    fmd_by_tok[right].append(v)

                row: Dict[str, object] = {"model": model_name}
                # segments used (if available in summary)
                seg_used = None
                for s in tokenizer_section.get("summary", []):
                    if s.get("model") == model_name:
                        seg_used = s.get("segments_used")
                        break
                row["segments_used"] = seg_used

                for tok in all_tokenizers:
                    vals = fmd_by_tok.get(tok, [])
                    if vals:
                        arr = np.array(vals, dtype=float)
                        row[f"{_sanitize(tok)}_mean_fmd"] = float(np.mean(arr))
                        row[f"{_sanitize(tok)}_std_fmd"] = float(np.std(arr))
                    else:
                        row[f"{_sanitize(tok)}_mean_fmd"] = None
                        row[f"{_sanitize(tok)}_std_fmd"] = None

                means = [row[f"{_sanitize(tok)}_mean_fmd"] for tok in all_tokenizers if row[f"{_sanitize(tok)}_mean_fmd"] is not None]
                if means:
                    min_mean = float(min(means))
                    max_mean = float(max(means))
                    row["mean_of_means"] = float(np.mean(means))
                    row["min_mean"] = min_mean
                    row["max_mean"] = max_mean
                    row["delta_max_min"] = float(max_mean - min_mean)
                    for tok in all_tokenizers:
                        m = row[f"{_sanitize(tok)}_mean_fmd"]
                        row[f"{_sanitize(tok)}_delta_vs_min"] = float(m - min_mean) if m is not None else None

                tok_by_model_rows.append(row)

            # Write CSV
            fieldnames = ["model", "segments_used"]
            for tok in all_tokenizers:
                fieldnames.extend([f"{_sanitize(tok)}_mean_fmd", f"{_sanitize(tok)}_std_fmd", f"{_sanitize(tok)}_delta_vs_min"])
            fieldnames.extend(["mean_of_means", "min_mean", "max_mean", "delta_max_min"])

            with open(csv_tok_by_model, "w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(tok_by_model_rows)
        else:
            csv_tok_by_model.touch()

        # Build per-tokenizer rows (each row: tokenizer, segments_used, <model>_mean_fmd, <model>_std_fmd, <model>_delta_vs_min, ...)
        model_by_tok_rows = []
        if model_section:
            for tokenizer_name in all_tokenizers:
                model_rows = [r for r in model_section.get("rows", []) if r.get("group") == tokenizer_name and r.get("fmd") is not None]
                fmd_by_model: Dict[str, List[float]] = defaultdict(list)
                for r in model_rows:
                    left = r.get("left")
                    right = r.get("right")
                    try:
                        v = float(r.get("fmd"))
                    except Exception:
                        continue
                    fmd_by_model[left].append(v)
                    fmd_by_model[right].append(v)

                row = {"tokenizer": tokenizer_name}
                seg_used = None
                for s in model_section.get("summary", []):
                    if s.get("tokenizer") == tokenizer_name:
                        seg_used = s.get("segments_used")
                        break
                row["segments_used"] = seg_used

                for model_name in all_models:
                    vals = fmd_by_model.get(model_name, [])
                    if vals:
                        arr = np.array(vals, dtype=float)
                        row[f"{_sanitize(model_name)}_mean_fmd"] = float(np.mean(arr))
                        row[f"{_sanitize(model_name)}_std_fmd"] = float(np.std(arr))
                    else:
                        row[f"{_sanitize(model_name)}_mean_fmd"] = None
                        row[f"{_sanitize(model_name)}_std_fmd"] = None

                means = [row[f"{_sanitize(m)}_mean_fmd"] for m in all_models if row[f"{_sanitize(m)}_mean_fmd"] is not None]
                if means:
                    min_mean = float(min(means))
                    max_mean = float(max(means))
                    row["mean_of_means"] = float(np.mean(means))
                    row["min_mean"] = min_mean
                    row["max_mean"] = max_mean
                    row["delta_max_min"] = float(max_mean - min_mean)
                    for model_name in all_models:
                        m = row[f"{_sanitize(model_name)}_mean_fmd"]
                        row[f"{_sanitize(model_name)}_delta_vs_min"] = float(m - min_mean) if m is not None else None

                model_by_tok_rows.append(row)

            # Write CSV
            fieldnames = ["tokenizer", "segments_used"]
            for model_name in all_models:
                fieldnames.extend([f"{_sanitize(model_name)}_mean_fmd", f"{_sanitize(model_name)}_std_fmd", f"{_sanitize(model_name)}_delta_vs_min"])
            fieldnames.extend(["mean_of_means", "min_mean", "max_mean", "delta_max_min"])

            with open(csv_model_by_tok, "w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(model_by_tok_rows)
        else:
            csv_model_by_tok.touch()

        # Pairwise and averaged delta CSVs (tokenizer-vs-tokenizer per model, model-vs-model per tokenizer)
        csv_tok_pairwise = output_dir / "per_song_tokenizer_pairwise.csv"
        csv_tok_avg_by_model = output_dir / "per_song_tokenizer_avg_by_model.csv"
        csv_tok_avg_over_models = output_dir / "per_song_tokenizer_avg_over_models.csv"
        csv_model_pairwise = output_dir / "per_song_model_pairwise.csv"
        csv_model_avg_by_tokenizer = output_dir / "per_song_model_avg_by_tokenizer.csv"
        csv_model_avg_over_tokenizers = output_dir / "per_song_model_avg_over_tokenizers.csv"

        # Tokenizer-axis pairwise and averages
        if tokenizer_section:
            # Pairwise rows: model, tokenizer_a, tokenizer_b, fmd
            with open(csv_tok_pairwise, "w", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(fh, fieldnames=["model", "tokenizer_a", "tokenizer_b", "fmd", "n_left", "n_right"])
                writer.writeheader()
                for r in tokenizer_section.get("rows", []):
                    writer.writerow(
                        {
                            "model": r.get("group"),
                            "tokenizer_a": r.get("left"),
                            "tokenizer_b": r.get("right"),
                            "fmd": r.get("fmd"),
                            "n_left": r.get("n_left"),
                            "n_right": r.get("n_right"),
                        }
                    )

            # For each model, compute mean distance of each tokenizer to other tokenizers
            avg_rows = []
            for model_name in all_models:
                rows_for_model = [r for r in tokenizer_section.get("rows", []) if r.get("group") == model_name and r.get("fmd") is not None]
                fmd_by_tok: Dict[str, List[float]] = defaultdict(list)
                for r in rows_for_model:
                    try:
                        v = float(r.get("fmd"))
                    except Exception:
                        continue
                    fmd_by_tok[r.get("left")].append(v)
                    fmd_by_tok[r.get("right")].append(v)

                for tok in all_tokenizers:
                    vals = fmd_by_tok.get(tok, [])
                    if vals:
                        arr = np.array(vals, dtype=float)
                        avg_rows.append(
                            {
                                "model": model_name,
                                "tokenizer": tok,
                                "mean_to_other_tokenizers": float(np.mean(arr)),
                                "std_to_other_tokenizers": float(np.std(arr)),
                                "n_pairs": int(arr.size),
                            }
                        )
                    else:
                        avg_rows.append(
                            {
                                "model": model_name,
                                "tokenizer": tok,
                                "mean_to_other_tokenizers": None,
                                "std_to_other_tokenizers": None,
                                "n_pairs": 0,
                            }
                        )

            with open(csv_tok_avg_by_model, "w", newline="", encoding="utf-8") as fh:
                fieldnames = ["model", "tokenizer", "mean_to_other_tokenizers", "std_to_other_tokenizers", "n_pairs"]
                writer = csv.DictWriter(fh, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(avg_rows)

            # Aggregate across models: per-tokenizer mean of means
            agg: Dict[str, List[float]] = defaultdict(list)
            for row in avg_rows:
                if row.get("mean_to_other_tokenizers") is not None:
                    agg[row["tokenizer"]].append(row["mean_to_other_tokenizers"])

            agg_rows = []
            for tok in all_tokenizers:
                vals = agg.get(tok, [])
                if vals:
                    arr = np.array(vals, dtype=float)
                    agg_rows.append(
                        {
                            "tokenizer": tok,
                            "mean_of_means": float(np.mean(arr)),
                            "std_of_means": float(np.std(arr)),
                            "min_mean": float(np.min(arr)),
                            "max_mean": float(np.max(arr)),
                        }
                    )
                else:
                    agg_rows.append(
                        {"tokenizer": tok, "mean_of_means": None, "std_of_means": None, "min_mean": None, "max_mean": None}
                    )

            with open(csv_tok_avg_over_models, "w", newline="", encoding="utf-8") as fh:
                fieldnames = ["tokenizer", "mean_of_means", "std_of_means", "min_mean", "max_mean"]
                writer = csv.DictWriter(fh, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(agg_rows)
        else:
            csv_tok_pairwise.touch()
            csv_tok_avg_by_model.touch()
            csv_tok_avg_over_models.touch()

        # Model-axis pairwise and averages
        if model_section:
            with open(csv_model_pairwise, "w", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(fh, fieldnames=["tokenizer", "model_a", "model_b", "fmd", "n_left", "n_right"])
                writer.writeheader()
                for r in model_section.get("rows", []):
                    writer.writerow(
                        {
                            "tokenizer": r.get("group"),
                            "model_a": r.get("left"),
                            "model_b": r.get("right"),
                            "fmd": r.get("fmd"),
                            "n_left": r.get("n_left"),
                            "n_right": r.get("n_right"),
                        }
                    )

            avg_rows_m = []
            for tokenizer_name in all_tokenizers:
                rows_for_tok = [r for r in model_section.get("rows", []) if r.get("group") == tokenizer_name and r.get("fmd") is not None]
                fmd_by_model: Dict[str, List[float]] = defaultdict(list)
                for r in rows_for_tok:
                    try:
                        v = float(r.get("fmd"))
                    except Exception:
                        continue
                    fmd_by_model[r.get("left")].append(v)
                    fmd_by_model[r.get("right")].append(v)

                for model_name in all_models:
                    vals = fmd_by_model.get(model_name, [])
                    if vals:
                        arr = np.array(vals, dtype=float)
                        avg_rows_m.append(
                            {
                                "tokenizer": tokenizer_name,
                                "model": model_name,
                                "mean_to_other_models": float(np.mean(arr)),
                                "std_to_other_models": float(np.std(arr)),
                                "n_pairs": int(arr.size),
                            }
                        )
                    else:
                        avg_rows_m.append(
                            {
                                "tokenizer": tokenizer_name,
                                "model": model_name,
                                "mean_to_other_models": None,
                                "std_to_other_models": None,
                                "n_pairs": 0,
                            }
                        )

            with open(csv_model_avg_by_tokenizer, "w", newline="", encoding="utf-8") as fh:
                fieldnames = ["tokenizer", "model", "mean_to_other_models", "std_to_other_models", "n_pairs"]
                writer = csv.DictWriter(fh, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(avg_rows_m)

            # Aggregate across tokenizers: per-model mean of means
            agg_m: Dict[str, List[float]] = defaultdict(list)
            for row in avg_rows_m:
                if row.get("mean_to_other_models") is not None:
                    agg_m[row["model"]].append(row["mean_to_other_models"])

            agg_rows_m = []
            for model_name in all_models:
                vals = agg_m.get(model_name, [])
                if vals:
                    arr = np.array(vals, dtype=float)
                    agg_rows_m.append(
                        {
                            "model": model_name,
                            "mean_of_means": float(np.mean(arr)),
                            "std_of_means": float(np.std(arr)),
                            "min_mean": float(np.min(arr)),
                            "max_mean": float(np.max(arr)),
                        }
                    )
                else:
                    agg_rows_m.append({"model": model_name, "mean_of_means": None, "std_of_means": None, "min_mean": None, "max_mean": None})

            with open(csv_model_avg_over_tokenizers, "w", newline="", encoding="utf-8") as fh:
                fieldnames = ["model", "mean_of_means", "std_of_means", "min_mean", "max_mean"]
                writer = csv.DictWriter(fh, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(agg_rows_m)
        else:
            csv_model_pairwise.touch()
            csv_model_avg_by_tokenizer.touch()
            csv_model_avg_over_tokenizers.touch()

        # Final summary dictionary (include new CSV paths)
        summary = {
            "midi_path": str(midi_path),
            "axis": axis,
            "segments_requested": int(n_segments),
            "segments_built": len(segments),
            "preprocessing": {
                "remove_velocity": bool(remove_velocity),
                "hard_quantization": bool(hard_quantization),
            },
            "tokenizers": list(all_tokenizers),
            "models": list(all_models),
            "tokenizer_axis": tokenizer_section,
            "model_axis": model_section,
            "tokenizer_stats": tokenizer_stats,
            "model_stats": model_stats,
            "outputs": {
                "tokenizer_csv": str(csv_tokenizer),
                "model_csv": str(csv_model),
                "segment_csv": str(csv_segment),
                "tokenizer_stats_csv": str(csv_tokenizer_stats),
                "model_stats_csv": str(csv_model_stats),
                "tokenizer_by_model_csv": str(csv_tok_by_model),
                "model_by_tokenizer_csv": str(csv_model_by_tok),
                "tokenizer_pairwise_csv": str(csv_tok_pairwise),
                "tokenizer_avg_by_model_csv": str(csv_tok_avg_by_model),
                "tokenizer_avg_over_models_csv": str(csv_tok_avg_over_models),
                "model_pairwise_csv": str(csv_model_pairwise),
                "model_avg_by_tokenizer_csv": str(csv_model_avg_by_tokenizer),
                "model_avg_over_tokenizers_csv": str(csv_model_avg_over_tokenizers),
                "markdown": str(md_path),
                "json": str(json_path),
            },
        }

        with open(json_path, "w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)

        def _fmt_fmd(value) -> str:
            return f"{float(value):.4f}" if value is not None else "n/a"

        lines = [
            "# Per-song FMD sensitivity",
            "",
            f"- MIDI file: `{midi_path}`",
            f"- Segments requested: **{n_segments}**",
            f"- Segments built: **{len(segments)}**",
            f"- Preprocessing: velocity={'off' if remove_velocity else 'on'}, quantization={'on' if hard_quantization else 'off'}",
            "",
        ]

        if tokenizer_section:
            lines.extend(["## Tokenizer axis", ""])
            if tokenizer_section["summary"]:
                for row in tokenizer_section["summary"]:
                    lines.append(
                        f"- model `{row['model']}`: {row['n_pairs']} pairs, "
                        f"segments={row['segments_used']}, mean FMD={_fmt_fmd(row['mean_fmd'])}"
                    )
            else:
                lines.append("No tokenizer comparisons were possible.")
            lines.append("")

        # Per-tokenizer summary (mean FMD across pairwise comparisons)
        if tokenizer_stats:
            lines.extend(["### Tokenizer mean FMD (per-tokenizer)", ""])
            for row in tokenizer_stats:
                lines.append(
                    f"- tokenizer `{row['tokenizer']}`: pairs={row['count_pairs']}, mean FMD={_fmt_fmd(row['mean_fmd'])}, delta_vs_min={row.get('delta_vs_min', 0):.4f}"
                )
            lines.append("")
        else:
            lines.append("No tokenizer-level stats available.")
            lines.append("")

        if model_section:
            lines.extend(["## Model axis", ""])
            if model_section["summary"]:
                for row in model_section["summary"]:
                    lines.append(
                        f"- tokenizer `{row['tokenizer']}`: {row['n_pairs']} pairs, "
                        f"segments={row['segments_used']}, mean FMD={_fmt_fmd(row['mean_fmd'])}"
                    )
            else:
                lines.append("No model comparisons were possible.")
            lines.append("")

        # Per-model summary (mean FMD across pairwise comparisons)
        if model_stats:
            lines.extend(["### Model mean FMD (per-model)", ""])
            for row in model_stats:
                lines.append(
                    f"- model `{row.get('model', row.get('model'))}`: pairs={row['count_pairs']}, mean FMD={_fmt_fmd(row['mean_fmd'])}, delta_vs_min={row.get('delta_vs_min', 0):.4f}"
                )
            lines.append("")
        else:
            lines.append("No model-level stats available.")
            lines.append("")

        with open(md_path, "w", encoding="utf-8") as handle:
            handle.write("\n".join(lines) + "\n")

        return summary

    def _synthetic_embeddings(self, dataset_name: str, variant: PipelineVariant, dim: int = 512) -> np.ndarray:
        key = f"{dataset_name}|{variant.name}|{self.seed}"
        seed = int(md5(key.encode("utf-8")).hexdigest()[:8], 16)
        rng = np.random.default_rng(seed)
        return rng.normal(0.0, 1.0, size=(self.synthetic_fallback_samples, dim)).astype(np.float32)

    @staticmethod
    def _normalize_embedding_payload(payload) -> Dict:
        if isinstance(payload, dict):
            return {
                "embeddings": payload.get("embeddings"),
                "source": str(payload.get("source", "real")),
                "real_files": int(payload.get("real_files", 0)),
                "total_files": int(payload.get("total_files", 0)),
            }
        return {
            "embeddings": payload,
            "source": "real",
            "real_files": int(payload.shape[0]) if payload is not None else 0,
            "total_files": int(payload.shape[0]) if payload is not None else 0,
        }

    def _bootstrap_ci_for_pair(self, emb_a: np.ndarray, emb_b: np.ndarray, key: str) -> Dict:
        if not self.bootstrap_enabled or emb_a.size == 0 or emb_b.size == 0:
            return {"mean": None, "std": None, "ci_lower": None, "ci_upper": None}

        local_seed = int(md5(f"{key}|{self.bootstrap_seed}".encode("utf-8")).hexdigest()[:8], 16)
        rng = np.random.default_rng(local_seed)
        n_a = emb_a.shape[0]
        n_b = emb_b.shape[0]
        sample_n_a = max(2, min(n_a, self.synthetic_fallback_samples))
        sample_n_b = max(2, min(n_b, self.synthetic_fallback_samples))
        values = []

        for _ in range(self.bootstrap_resamples):
            idx_a = rng.integers(0, n_a, size=sample_n_a)
            idx_b = rng.integers(0, n_b, size=sample_n_b)
            values.append(float(self.fmd.compute_fmd(emb_a[idx_a], emb_b[idx_b])))

        arr = np.array(values, dtype=float)
        alpha = max(0.0, min(0.5, (1.0 - self.bootstrap_ci) / 2.0))
        lower = float(np.quantile(arr, alpha))
        upper = float(np.quantile(arr, 1.0 - alpha))
        return {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "ci_lower": lower,
            "ci_upper": upper,
        }

    def _extract_dataset_embeddings(self, dataset_name: str, variant: PipelineVariant) -> Dict:
        midi_files = self._list_dataset_midis(dataset_name)
        vectors: List[np.ndarray] = []
        real_count = 0

        for midi_path in midi_files:
            try:
                midi_data = self._preprocess_midi_file(midi_path, variant)
                if midi_data is None:
                    continue
                tokens = self._tokens_for_variant(midi_data, variant)
                if variant.input_format == MIDITOK_FORMAT and not tokens:
                    continue
                vec = self._extract_variant_embeddings([tokens], [midi_data], variant)[0]
                vectors.append(vec)
                real_count += 1
            except Exception as exc:
                logger.warning(f"Skipping {midi_path} for {variant.name}: {exc}")

        if not vectors:
            if self.fallback_mode in {"strict", "hard_strict"}:
                return {
                    "embeddings": None,
                    "source": "missing",
                    "real_files": 0,
                    "total_files": len(midi_files),
                }

            logger.warning(
                f"No embeddings from files for dataset={dataset_name}, variant={variant.name}; "
                "using deterministic synthetic fallback"
            )
            return {
                "embeddings": self._synthetic_embeddings(dataset_name, variant),
                "source": "synthetic",
                "real_files": 0,
                "total_files": len(midi_files),
            }

        return {
            "embeddings": np.vstack(vectors),
            "source": "real",
            "real_files": real_count,
            "total_files": len(midi_files),
        }

    def run_pairwise_benchmark(self, variants: Sequence[PipelineVariant]) -> List[Dict]:
        """Compute FMD for configured dataset pairs across all variants."""
        names = self._dataset_names()
        if self.compare_all_pairs:
            pairs = [(names[i], names[j]) for i in range(len(names)) for j in range(i + 1, len(names))]
        else:
            exp5 = self.config.get("experiments", {}).get("exp5_cross_genre", {})
            pairs = self._parse_pairs(exp5.get("pairs", []))
            if not pairs:
                pairs = [(names[i], names[j]) for i in range(len(names)) for j in range(i + 1, len(names))]

        rows: List[Dict] = []
        cache: Dict[Tuple[str, str], Dict] = {}

        for variant in variants:
            for ds_a, ds_b in pairs:
                key_a = (ds_a, variant.name)
                key_b = (ds_b, variant.name)
                if key_a not in cache:
                    payload_a = self._extract_dataset_embeddings(ds_a, variant)
                    cache[key_a] = self._normalize_embedding_payload(payload_a)
                if key_b not in cache:
                    payload_b = self._extract_dataset_embeddings(ds_b, variant)
                    cache[key_b] = self._normalize_embedding_payload(payload_b)

                emb_a = cache[key_a]["embeddings"]
                emb_b = cache[key_b]["embeddings"]
                is_valid = emb_a is not None and emb_b is not None
                real_pair = cache[key_a]["source"] == "real" and cache[key_b]["source"] == "real"

                if self.fallback_mode == "hard_strict" and not real_pair:
                    raise RuntimeError(
                        "Hard strict mode failed: non-real embeddings detected for "
                        f"variant={variant.name}, pair=({ds_a}, {ds_b}), "
                        f"sources=({cache[key_a]['source']}, {cache[key_b]['source']})"
                    )

                bootstrap = {
                    "mean": None,
                    "std": None,
                    "ci_lower": None,
                    "ci_upper": None,
                }
                fmd_value = None

                if is_valid:
                    fmd_value = float(self.fmd.compute_fmd(emb_a, emb_b))
                    bootstrap = self._bootstrap_ci_for_pair(
                        emb_a,
                        emb_b,
                        key=f"{variant.name}|{ds_a}|{ds_b}",
                    )

                rows.append(
                    {
                        "variant": variant.name,
                        "tokenizer": variant.tokenizer,
                        "input_format": variant.input_format,
                        "model": variant.model,
                        "remove_velocity": variant.remove_velocity,
                        "hard_quantization": variant.hard_quantization,
                        "dataset_a": ds_a,
                        "dataset_b": ds_b,
                        "fmd": fmd_value,
                        "valid": is_valid,
                        "real_pair": real_pair,
                        "source_a": cache[key_a]["source"],
                        "source_b": cache[key_b]["source"],
                        "real_files_a": cache[key_a]["real_files"],
                        "real_files_b": cache[key_b]["real_files"],
                        "bootstrap_mean": bootstrap["mean"],
                        "bootstrap_std": bootstrap["std"],
                        "bootstrap_ci_lower": bootstrap["ci_lower"],
                        "bootstrap_ci_upper": bootstrap["ci_upper"],
                    }
                )

                if self.progress_log_every > 0:
                    processed = len(rows)
                    total = max(1, len(variants) * len(pairs))
                    if processed % self.progress_log_every == 0 or processed == total:
                        pct = 100.0 * processed / total
                        logger.info(f"Pairwise benchmark progress: {processed}/{total} ({pct:.1f}%)")

        return rows

    @staticmethod
    def _split_pairwise_rows(pairwise_rows: List[Dict]) -> Dict[str, List[Dict]]:
        valid_rows = [row for row in pairwise_rows if row.get("valid") and row.get("fmd") is not None]
        real_only_rows = [row for row in valid_rows if row.get("real_pair")]
        return {
            "all": valid_rows,
            "real_only": real_only_rows,
        }

    def compute_variant_effects(self, pairwise_rows: List[Dict]) -> Dict:
        """Compute per-variant deltas for tokenizer/model under fixed controls."""
        filtered = [row for row in pairwise_rows if row.get("valid") and row.get("fmd") is not None]
        if not filtered:
            return {"tokenizer_deltas": [], "model_deltas": []}

        by_cell: Dict[Tuple[str, str, bool, bool], List[float]] = defaultdict(list)
        for row in filtered:
            key = (
                str(row["tokenizer"]),
                str(row["model"]),
                bool(row["remove_velocity"]),
                bool(row["hard_quantization"]),
            )
            by_cell[key].append(float(row["fmd"]))

        cell_mean = {key: float(np.mean(values)) for key, values in by_cell.items()}

        tokenizer_deltas: List[Dict] = []
        model_deltas: List[Dict] = []

        tokenizers = sorted({key[0] for key in cell_mean})
        models = sorted({key[1] for key in cell_mean})
        preprocess = sorted({(key[2], key[3]) for key in cell_mean})

        for model in models:
            for remove_velocity, hard_quantization in preprocess:
                for i in range(len(tokenizers)):
                    for j in range(i + 1, len(tokenizers)):
                        t1 = tokenizers[i]
                        t2 = tokenizers[j]
                        k1 = (t1, model, remove_velocity, hard_quantization)
                        k2 = (t2, model, remove_velocity, hard_quantization)
                        if k1 in cell_mean and k2 in cell_mean:
                            tokenizer_deltas.append(
                                {
                                    "model": model,
                                    "remove_velocity": remove_velocity,
                                    "hard_quantization": hard_quantization,
                                    "tokenizer_a": t1,
                                    "tokenizer_b": t2,
                                    "mean_fmd_a": cell_mean[k1],
                                    "mean_fmd_b": cell_mean[k2],
                                    "delta_fmd": float(cell_mean[k1] - cell_mean[k2]),
                                }
                            )

        for tokenizer in tokenizers:
            for remove_velocity, hard_quantization in preprocess:
                for i in range(len(models)):
                    for j in range(i + 1, len(models)):
                        m1 = models[i]
                        m2 = models[j]
                        k1 = (tokenizer, m1, remove_velocity, hard_quantization)
                        k2 = (tokenizer, m2, remove_velocity, hard_quantization)
                        if k1 in cell_mean and k2 in cell_mean:
                            model_deltas.append(
                                {
                                    "tokenizer": tokenizer,
                                    "remove_velocity": remove_velocity,
                                    "hard_quantization": hard_quantization,
                                    "model_a": m1,
                                    "model_b": m2,
                                    "mean_fmd_a": cell_mean[k1],
                                    "mean_fmd_b": cell_mean[k2],
                                    "delta_fmd": float(cell_mean[k1] - cell_mean[k2]),
                                }
                            )

        return {
            "tokenizer_deltas": tokenizer_deltas,
            "model_deltas": model_deltas,
        }

    def run_ranking_benchmark(self, variants: Sequence[PipelineVariant]) -> Dict:
        """Build rankings and stability across configurations."""
        dataset_names = self._dataset_names()
        ranking_by_variant: Dict[str, Dict[str, List[int]]] = {}

        for variant in variants:
            emb_sets = []
            for ds_name in dataset_names:
                payload = self._normalize_embedding_payload(self._extract_dataset_embeddings(ds_name, variant))
                emb = payload["embeddings"]
                if self.fallback_mode == "hard_strict" and payload.get("source") != "real":
                    raise RuntimeError(
                        f"Hard strict mode failed in ranking: dataset={ds_name}, "
                        f"variant={variant.name}, source={payload.get('source')}"
                    )
                if emb is None:
                    if self.fallback_mode in {"strict", "hard_strict"}:
                        raise RuntimeError(
                            f"Missing real embeddings for dataset={ds_name}, variant={variant.name} in strict mode"
                        )
                    emb = self._synthetic_embeddings(ds_name, variant)
                emb_sets.append((ds_name, emb))

            matrix_result = self.fmd.compute_batch_fmd(emb_sets)
            matrix = matrix_result["fmd_matrix"]
            per_ref: Dict[str, List[int]] = {}
            for ref_idx, ref_name in enumerate(dataset_names):
                ranking = FMDRanking.rank_by_fmd(matrix, ref_idx)["ranking"]
                per_ref[ref_name] = [int(x) for x in ranking]
            ranking_by_variant[variant.name] = per_ref

        # Stability by reference dataset over all variants.
        stability: Dict[str, float] = {}
        for ref_name in dataset_names:
            rankings_dict = {
                variant_name: np.array(ref_rankings[ref_name], dtype=int)
                for variant_name, ref_rankings in ranking_by_variant.items()
            }
            stability[ref_name] = float(FMDRanking.compute_ranking_stability(rankings_dict))

        return {"rankings": ranking_by_variant, "stability": stability}

    def evaluate_expected_order(self, ranking_results: Dict) -> Dict:
        """Compare obtained rankings against expected similarity order from config."""
        expected_cfg = self.config.get("paper", {}).get("expected_orders", [])
        if not expected_cfg:
            return {"available": False, "details": []}

        dataset_names = self._dataset_names()
        idx_by_name = {name: i for i, name in enumerate(dataset_names)}
        details = []

        for variant_name, ranking_by_ref in ranking_results["rankings"].items():
            for expected in expected_cfg:
                ref = expected.get("reference")
                order = expected.get("order", [])
                if ref not in ranking_by_ref:
                    continue

                predicted_indices = ranking_by_ref[ref]
                predicted_names = [dataset_names[i] for i in predicted_indices]

                expected_rank = {name: rank for rank, name in enumerate(order)}
                pred_scores = []
                exp_scores = []
                for name in predicted_names:
                    if name in expected_rank:
                        pred_scores.append(len(pred_scores))
                        exp_scores.append(expected_rank[name])

                if len(pred_scores) >= 2:
                    sp = spearmanr(pred_scores, exp_scores).correlation
                    kd = kendalltau(pred_scores, exp_scores).correlation
                else:
                    sp = np.nan
                    kd = np.nan

                details.append(
                    {
                        "variant": variant_name,
                        "reference": ref,
                        "predicted_order": predicted_names,
                        "expected_order": order,
                        "spearman": float(sp) if sp == sp else None,
                        "kendall": float(kd) if kd == kd else None,
                    }
                )

        return {"available": True, "details": details}

    def compute_special_pair_metrics(self, pairwise_rows: List[Dict]) -> Dict:
        """Aggregate configured special genre pairs for publication-ready analysis."""
        special_pairs = self._resolve_special_pairs()
        if not special_pairs:
            return {"available": False, "rows": [], "summary": [], "top_variants": []}

        usable_rows = [row for row in pairwise_rows if row.get("valid") and row.get("fmd") is not None]
        if not usable_rows:
            return {"available": False, "rows": [], "summary": [], "top_variants": []}

        lookup: Dict[Tuple[str, str, str], float] = {}
        for row in usable_rows:
            key_direct = (row["variant"], row["dataset_a"], row["dataset_b"])
            key_reverse = (row["variant"], row["dataset_b"], row["dataset_a"])
            lookup[key_direct] = float(row["fmd"])
            lookup[key_reverse] = float(row["fmd"])

        rows: List[Dict] = []
        for pair in special_pairs:
            for variant_name in sorted({row["variant"] for row in usable_rows}):
                value = lookup.get((variant_name, pair["dataset_a"], pair["dataset_b"]))
                if value is None:
                    continue
                rows.append(
                    {
                        "variant": variant_name,
                        "genre_a": pair["genre_a"],
                        "genre_b": pair["genre_b"],
                        "pair": f"{pair['genre_a']} vs {pair['genre_b']}",
                        "dataset_a": pair["dataset_a"],
                        "dataset_b": pair["dataset_b"],
                        "fmd": float(value),
                    }
                )

        if not rows:
            return {"available": False, "rows": [], "summary": [], "top_variants": []}

        grouped: Dict[str, List[float]] = defaultdict(list)
        for row in rows:
            grouped[row["pair"]].append(float(row["fmd"]))

        global_mean = float(np.mean([row["fmd"] for row in rows]))
        summary: List[Dict] = []
        for pair_name, values in grouped.items():
            arr = np.array(values, dtype=float)
            summary.append(
                {
                    "pair": pair_name,
                    "count": int(arr.size),
                    "mean_fmd": float(np.mean(arr)),
                    "std_fmd": float(np.std(arr)),
                    "min_fmd": float(np.min(arr)),
                    "max_fmd": float(np.max(arr)),
                    # >1.0 means this pair is more separable than average configured pair.
                    "distinguishability_ratio": float(np.mean(arr) / max(global_mean, 1e-9)),
                }
            )

        summary.sort(key=lambda item: item["mean_fmd"], reverse=True)

        top_rows: List[Dict] = []
        for pair_name in sorted(grouped.keys()):
            pair_rows = sorted(
                [row for row in rows if row["pair"] == pair_name],
                key=lambda item: float(item["fmd"]),
                reverse=True,
            )
            for rank, row in enumerate(pair_rows[: self.top_variants_per_pair], start=1):
                top_rows.append(
                    {
                        "pair": pair_name,
                        "rank": rank,
                        "variant": row["variant"],
                        "fmd": float(row["fmd"]),
                        "dataset_a": row["dataset_a"],
                        "dataset_b": row["dataset_b"],
                    }
                )

        return {"available": True, "rows": rows, "summary": summary, "top_variants": top_rows}

    def save_outputs(
        self,
        pairwise_rows: List[Dict],
        ranking_results: Dict,
        expected_eval: Dict,
        special_metrics: Dict | None = None,
        variant_effects: Dict | None = None,
    ) -> Dict:
        """Save JSON, CSV and markdown summary for paper draft."""
        json_path = self.output_dir / "paper_results.json"
        csv_path = self.output_dir / "pairwise_fmd.csv"
        csv_all_path = self.output_dir / "pairwise_fmd_all.csv"
        csv_real_only_path = self.output_dir / "pairwise_fmd_real_only.csv"
        md_path = self.output_dir / "paper_summary.md"
        special_csv_path = self.output_dir / "special_pair_fmd.csv"
        special_summary_csv_path = self.output_dir / "special_pair_summary.csv"
        special_top_csv_path = self.output_dir / "special_pair_top_variants.csv"
        tokenizer_delta_csv_path = self.output_dir / "variant_delta_tokenizer.csv"
        model_delta_csv_path = self.output_dir / "variant_delta_model.csv"

        if special_metrics is None:
            special_metrics = {"available": False, "rows": [], "summary": [], "top_variants": []}
        if variant_effects is None:
            variant_effects = {
                "all": {"tokenizer_deltas": [], "model_deltas": []},
                "real_only": {"tokenizer_deltas": [], "model_deltas": []},
            }

        split_rows = self._split_pairwise_rows(pairwise_rows)
        all_rows = split_rows["all"]
        real_only_rows = split_rows["real_only"]

        payload = {
            "pairwise_all": all_rows,
            "pairwise_real_only": real_only_rows,
            "ranking": ranking_results,
            "expected_eval": expected_eval,
            "special_pairs": special_metrics,
            "variant_effects": variant_effects,
        }
        with open(json_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)

        pairwise_fields = [
            "variant",
            "tokenizer",
            "input_format",
            "model",
            "remove_velocity",
            "hard_quantization",
            "dataset_a",
            "dataset_b",
            "fmd",
            "valid",
            "real_pair",
            "source_a",
            "source_b",
            "real_files_a",
            "real_files_b",
            "bootstrap_mean",
            "bootstrap_std",
            "bootstrap_ci_lower",
            "bootstrap_ci_upper",
        ]

        with open(csv_path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=pairwise_fields)
            writer.writeheader()
            writer.writerows(all_rows)

        with open(csv_all_path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=pairwise_fields)
            writer.writeheader()
            writer.writerows(all_rows)

        with open(csv_real_only_path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=pairwise_fields)
            writer.writeheader()
            writer.writerows(real_only_rows)

        with open(special_csv_path, "w", newline="", encoding="utf-8") as handle:
            fieldnames = ["variant", "genre_a", "genre_b", "pair", "dataset_a", "dataset_b", "fmd"]
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(special_metrics.get("rows", []))

        with open(special_summary_csv_path, "w", newline="", encoding="utf-8") as handle:
            fieldnames = [
                "pair",
                "count",
                "mean_fmd",
                "std_fmd",
                "min_fmd",
                "max_fmd",
                "distinguishability_ratio",
            ]
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(special_metrics.get("summary", []))

        with open(special_top_csv_path, "w", newline="", encoding="utf-8") as handle:
            fieldnames = ["pair", "rank", "variant", "fmd", "dataset_a", "dataset_b"]
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(special_metrics.get("top_variants", []))

        with open(tokenizer_delta_csv_path, "w", newline="", encoding="utf-8") as handle:
            fieldnames = [
                "model",
                "remove_velocity",
                "hard_quantization",
                "tokenizer_a",
                "tokenizer_b",
                "mean_fmd_a",
                "mean_fmd_b",
                "delta_fmd",
            ]
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(variant_effects.get("all", {}).get("tokenizer_deltas", []))

        with open(model_delta_csv_path, "w", newline="", encoding="utf-8") as handle:
            fieldnames = [
                "tokenizer",
                "remove_velocity",
                "hard_quantization",
                "model_a",
                "model_b",
                "mean_fmd_a",
                "mean_fmd_b",
                "delta_fmd",
            ]
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(variant_effects.get("all", {}).get("model_deltas", []))

        with open(self.output_dir / "variant_delta_tokenizer_real_only.csv", "w", newline="", encoding="utf-8") as handle:
            fieldnames = [
                "model",
                "remove_velocity",
                "hard_quantization",
                "tokenizer_a",
                "tokenizer_b",
                "mean_fmd_a",
                "mean_fmd_b",
                "delta_fmd",
            ]
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(variant_effects.get("real_only", {}).get("tokenizer_deltas", []))

        with open(self.output_dir / "variant_delta_model_real_only.csv", "w", newline="", encoding="utf-8") as handle:
            fieldnames = [
                "tokenizer",
                "remove_velocity",
                "hard_quantization",
                "model_a",
                "model_b",
                "mean_fmd_a",
                "mean_fmd_b",
                "delta_fmd",
            ]
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(variant_effects.get("real_only", {}).get("model_deltas", []))

        # Simple markdown summary for direct paper drafting.
        lines = [
            "# Paper-Oriented FMD Benchmark Summary",
            "",
            "## Pairwise comparisons",
            "",
            f"All valid rows: **{len(all_rows)}**",
            f"Real-only rows: **{len(real_only_rows)}**",
            "",
            "## Ranking stability by reference dataset",
            "",
        ]
        for ref_name, score in ranking_results["stability"].items():
            lines.append(f"- `{ref_name}`: {score:.4f}")

        lines.extend(["", "## Expected-order agreement", ""])
        if expected_eval.get("available"):
            details = expected_eval.get("details", [])
            if not details:
                lines.append("No comparable expected-order entries were found.")
            else:
                for row in details[:20]:
                    lines.append(
                        "- "
                        f"variant `{row['variant']}`, ref `{row['reference']}` -> "
                        f"spearman={row['spearman']}, kendall={row['kendall']}"
                    )
        else:
            lines.append("Not configured. Add `paper.expected_orders` to config for this section.")

        lines.extend(["", "## Special genre-pair separability", ""])
        if special_metrics.get("available"):
            for row in special_metrics.get("summary", [])[:10]:
                lines.append(
                    "- "
                    f"`{row['pair']}` -> mean FMD={row['mean_fmd']:.4f}, "
                    f"std={row['std_fmd']:.4f}, ratio={row['distinguishability_ratio']:.3f}"
                )
        else:
            lines.append("Not configured. Add `paper.special_pairs` and optional `paper.genre_aliases`.")

        lines.extend(["", "## Top separating variants per special pair", ""])
        if special_metrics.get("available") and special_metrics.get("top_variants"):
            for row in special_metrics.get("top_variants", [])[:30]:
                lines.append(
                    "- "
                    f"`{row['pair']}` rank {row['rank']}: `{row['variant']}` (FMD={row['fmd']:.4f})"
                )
        else:
            lines.append("No top-variant entries available.")

        lines.extend(["", "## Variant effects (delta FMD)", ""])
        tok_rows = variant_effects.get("all", {}).get("tokenizer_deltas", [])
        mod_rows = variant_effects.get("all", {}).get("model_deltas", [])
        tok_real_rows = variant_effects.get("real_only", {}).get("tokenizer_deltas", [])
        mod_real_rows = variant_effects.get("real_only", {}).get("model_deltas", [])
        lines.append(f"Tokenizer deltas rows (all): **{len(tok_rows)}**")
        lines.append(f"Model deltas rows (all): **{len(mod_rows)}**")
        lines.append(f"Tokenizer deltas rows (real-only): **{len(tok_real_rows)}**")
        lines.append(f"Model deltas rows (real-only): **{len(mod_real_rows)}**")
        for row in tok_rows[:8]:
            lines.append(
                "- "
                f"model `{row['model']}` ({row['remove_velocity']}/{row['hard_quantization']}): "
                f"`{row['tokenizer_a']}` - `{row['tokenizer_b']}` = {row['delta_fmd']:.4f}"
            )
        for row in mod_rows[:8]:
            lines.append(
                "- "
                f"tokenizer `{row['tokenizer']}` ({row['remove_velocity']}/{row['hard_quantization']}): "
                f"`{row['model_a']}` - `{row['model_b']}` = {row['delta_fmd']:.4f}"
            )

        with open(md_path, "w", encoding="utf-8") as handle:
            handle.write("\n".join(lines) + "\n")

        return {
            "json": str(json_path),
            "csv": str(csv_path),
            "csv_all": str(csv_all_path),
            "csv_real_only": str(csv_real_only_path),
            "markdown": str(md_path),
            "special_csv": str(special_csv_path),
            "special_summary_csv": str(special_summary_csv_path),
            "special_top_variants_csv": str(special_top_csv_path),
            "variant_delta_tokenizer_csv": str(tokenizer_delta_csv_path),
            "variant_delta_model_csv": str(model_delta_csv_path),
            "variant_delta_tokenizer_real_only_csv": str(self.output_dir / "variant_delta_tokenizer_real_only.csv"),
            "variant_delta_model_real_only_csv": str(self.output_dir / "variant_delta_model_real_only.csv"),
        }

    def run_full(self) -> Dict:
        """Run full benchmark suited for research reporting."""
        paper_cfg = self.config.get("paper", {})
        tokenizers = paper_cfg.get("tokenizers") or None
        models = paper_cfg.get("models") or None

        variants = self.build_variants(tokenizers=tokenizers, models=models)
        pairwise_rows = self.run_pairwise_benchmark(variants)
        ranking_results = self.run_ranking_benchmark(variants)
        expected_eval = self.evaluate_expected_order(ranking_results)
        split_rows = self._split_pairwise_rows(pairwise_rows)
        special_metrics = self.compute_special_pair_metrics(split_rows["all"])
        variant_effects = {
            "all": self.compute_variant_effects(split_rows["all"]),
            "real_only": self.compute_variant_effects(split_rows["real_only"]),
        }
        files = self.save_outputs(pairwise_rows, ranking_results, expected_eval, special_metrics, variant_effects)

        logger.info(f"Paper benchmark completed. Outputs: {files}")
        return {
            "variants": [v.name for v in variants],
            "pairwise_rows": len(pairwise_rows),
            "stability": ranking_results["stability"],
            "outputs": files,
        }

    def run_quick(self) -> Dict:
        """Fast smoke benchmark for one-click run in IDE."""
        tokenizers = [self.config["tokenization"]["tokenizers"][0]["type"]]
        models = [self.config["embeddings"]["models"][0]["name"]]
        if models[0] in {"CLaMP-1", "CLaMP-2"}:
            tokenizers = []
        variants = self.build_variants(
            tokenizers=tokenizers,
            models=models,
            preprocessing_grid=[(False, False)],
        )
        pairwise_rows = self.run_pairwise_benchmark(variants)
        ranking_results = self.run_ranking_benchmark(variants)
        expected_eval = self.evaluate_expected_order(ranking_results)
        split_rows = self._split_pairwise_rows(pairwise_rows)
        special_metrics = self.compute_special_pair_metrics(split_rows["all"])
        variant_effects = {
            "all": self.compute_variant_effects(split_rows["all"]),
            "real_only": self.compute_variant_effects(split_rows["real_only"]),
        }
        files = self.save_outputs(pairwise_rows, ranking_results, expected_eval, special_metrics, variant_effects)
        return {
            "variants": [v.name for v in variants],
            "pairwise_rows": len(pairwise_rows),
            "outputs": files,
        }

    # ------------------------------------------------------------------
    # Lakh MIDI validation pipeline
    # ------------------------------------------------------------------

    def run_lakh_validation(self) -> Dict:
        """Run paper-grade FMD sensitivity validation on Lakh MIDI.

        Workflow:
        1. Ensure Lakh data + Tagtraum annotations are available.
        2. Populate ``data/raw/lakh_rock/`` and ``data/raw/lakh_classical/``.
        3. Build validated model/input variants.
        4. For each variant: preprocess → represent/tokenize if needed → embed.
        5. Compute FMD(rock, classical) per variant with bootstrap CI.
        6. Run sensitivity analysis (ANOVA, Tukey, η², Cohen's d).
        7. Run embedding diagnostics (cosine, PCA/t-SNE, token stats).
        8. Save all artefacts and return summary dict.
        """
        from experiments.sensitivity_analysis import run_sensitivity_analysis
        from experiments.embedding_diagnostics import run_embedding_diagnostics

        lakh_cfg = self.config.get("lakh", {})
        genre_a = lakh_cfg.get("genres", ["rock", "classical"])[0]
        genre_b = lakh_cfg.get("genres", ["rock", "classical"])[1]
        ds_a = f"lakh_{genre_a}"
        ds_b = f"lakh_{genre_b}"

        output_dir = Path(lakh_cfg.get("output_dir", "results/reports/lakh"))
        output_dir.mkdir(parents=True, exist_ok=True)

        bootstrap_cfg = lakh_cfg.get("bootstrap", {})
        n_resamples = int(bootstrap_cfg.get("n_resamples", 200))
        confidence = float(bootstrap_cfg.get("confidence", 0.95))
        boot_seed = int(bootstrap_cfg.get("seed", self.seed))

        # Step 1 & 2: data
        logger.info("=== Lakh Validation: ensuring data ===")
        loader = LakhGenreLoader(self.config)
        loader.ensure_data()
        counts = loader.populate_raw_datasets()
        logger.info(f"Dataset counts: {counts}")

        # Step 3: build 32 variants
        variants = self.build_variants()  # default: full 4×2×4 grid
        logger.info(f"Lakh validation: {len(variants)} variants")

        # Step 4 & 5: extract embeddings + FMD per variant
        pairwise_rows: List[Dict] = []
        embeddings_cache: Dict[str, Dict[str, np.ndarray]] = {}  # variant → {genre: emb}
        token_seqs_cache: Dict[str, Dict[str, List[List[int]]]] = {}  # variant → {genre: [[int]]}
        fmd_per_variant: Dict[str, float] = {}

        total_steps = len(variants)
        for step_idx, variant in enumerate(variants, 1):
            pct = 100.0 * step_idx / total_steps
            logger.info(f"[Lakh {step_idx}/{total_steps} ({pct:.0f}%)] {variant.name}")

            variant_embs: Dict[str, np.ndarray] = {}
            variant_tokens: Dict[str, List[List[int]]] = {}

            for ds_name, genre_label in [(ds_a, genre_a), (ds_b, genre_b)]:
                midi_files = self._list_dataset_midis(ds_name)
                vectors: List[np.ndarray] = []
                seqs: List[List[int]] = []
                for midi_path in midi_files:
                    try:
                        midi_data = self._preprocess_midi_file(midi_path, variant)
                        if midi_data is None:
                            continue
                        tokens = self._tokens_for_variant(midi_data, variant)
                        if variant.input_format == MIDITOK_FORMAT and not tokens:
                            continue
                        seqs.append(tokens)
                        vec = self._extract_variant_embeddings([tokens], [midi_data], variant)[0]
                        vectors.append(vec)
                    except Exception as exc:
                        logger.warning(f"Skip {midi_path}: {exc}")

                if vectors:
                    variant_embs[genre_label] = np.vstack(vectors)
                    variant_tokens[genre_label] = seqs
                    logger.info(f"  {genre_label}: {len(vectors)} embeddings")
                else:
                    logger.warning(f"  {genre_label}: 0 embeddings!")

            embeddings_cache[variant.name] = variant_embs
            token_seqs_cache[variant.name] = variant_tokens

            emb_a = variant_embs.get(genre_a)
            emb_b = variant_embs.get(genre_b)
            if emb_a is not None and emb_b is not None and emb_a.shape[0] > 0 and emb_b.shape[0] > 0:
                fmd_value = float(self.fmd.compute_fmd(emb_a, emb_b))
                fmd_per_variant[variant.name] = fmd_value

                # Bootstrap CI
                from hashlib import md5
                local_seed = int(md5(f"{variant.name}|{boot_seed}".encode()).hexdigest()[:8], 16)
                rng = np.random.default_rng(local_seed)
                boot_values = []
                sample_n_a = max(2, min(emb_a.shape[0], emb_a.shape[0]))
                sample_n_b = max(2, min(emb_b.shape[0], emb_b.shape[0]))
                for _ in range(n_resamples):
                    idx_a = rng.integers(0, emb_a.shape[0], size=sample_n_a)
                    idx_b = rng.integers(0, emb_b.shape[0], size=sample_n_b)
                    boot_values.append(float(self.fmd.compute_fmd(emb_a[idx_a], emb_b[idx_b])))

                arr = np.array(boot_values)
                alpha = (1.0 - confidence) / 2.0

                pairwise_rows.append({
                    "variant": variant.name,
                    "tokenizer": variant.tokenizer,
                    "input_format": variant.input_format,
                    "model": variant.model,
                    "remove_velocity": variant.remove_velocity,
                    "hard_quantization": variant.hard_quantization,
                    "dataset_a": ds_a,
                    "dataset_b": ds_b,
                    "fmd": fmd_value,
                    "valid": True,
                    "real_pair": True,
                    "source_a": "real",
                    "source_b": "real",
                    "real_files_a": int(emb_a.shape[0]),
                    "real_files_b": int(emb_b.shape[0]),
                    "bootstrap_mean": float(np.mean(arr)),
                    "bootstrap_std": float(np.std(arr)),
                    "bootstrap_ci_lower": float(np.quantile(arr, alpha)),
                    "bootstrap_ci_upper": float(np.quantile(arr, 1.0 - alpha)),
                })
            else:
                pairwise_rows.append({
                    "variant": variant.name,
                    "tokenizer": variant.tokenizer,
                    "input_format": variant.input_format,
                    "model": variant.model,
                    "remove_velocity": variant.remove_velocity,
                    "hard_quantization": variant.hard_quantization,
                    "dataset_a": ds_a,
                    "dataset_b": ds_b,
                    "fmd": None,
                    "valid": False,
                    "real_pair": False,
                    "source_a": "missing",
                    "source_b": "missing",
                    "real_files_a": 0,
                    "real_files_b": 0,
                    "bootstrap_mean": None,
                    "bootstrap_std": None,
                    "bootstrap_ci_lower": None,
                    "bootstrap_ci_upper": None,
                })

        # Save pairwise CSV
        import csv
        csv_path = output_dir / "lakh_pairwise_fmd.csv"
        if pairwise_rows:
            fields = list(pairwise_rows[0].keys())
            with open(csv_path, "w", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(fh, fieldnames=fields)
                writer.writeheader()
                writer.writerows(pairwise_rows)
            logger.info(f"Lakh pairwise FMD saved: {csv_path}")

        # Step 6: Sensitivity analysis
        logger.info("=== Lakh Validation: sensitivity analysis ===")
        sensitivity_result = run_sensitivity_analysis(
            bootstrap_rows=pairwise_rows,
            output_dir=output_dir,
        )

        # Step 7: Embedding diagnostics
        logger.info("=== Lakh Validation: embedding diagnostics ===")
        diag_outputs = run_embedding_diagnostics(
            embeddings_by_variant=embeddings_cache,
            token_sequences_by_variant=token_seqs_cache,
            fmd_per_variant=fmd_per_variant,
            genre_a=genre_a,
            genre_b=genre_b,
            output_dir=output_dir,
            seed=self.seed,
        )

        # Summary JSON
        summary = {
            "genres": [genre_a, genre_b],
            "n_variants": len(variants),
            "n_valid": sum(1 for r in pairwise_rows if r.get("valid")),
            "fmd_range": {
                "min": min((r["fmd"] for r in pairwise_rows if r.get("fmd") is not None), default=None),
                "max": max((r["fmd"] for r in pairwise_rows if r.get("fmd") is not None), default=None),
            },
            "eta_squared": sensitivity_result.eta_squared,
            "diagnostics_outputs": diag_outputs,
        }
        with open(output_dir / "lakh_validation_summary.json", "w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2, default=str)

        # Step 8: Sample size ablation (if configured)
        ablation_cfg = lakh_cfg.get("sample_size_ablation", {})
        ablation_results = {}
        if ablation_cfg.get("enabled", False):
            logger.info("=== Lakh Validation: sample size ablation ===")
            ablation_results = self._run_sample_size_ablation(
                embeddings_cache=embeddings_cache,
                genre_a=genre_a,
                genre_b=genre_b,
                sizes=ablation_cfg.get("sizes", [50, 100, 200, 500]),
                n_repeats=ablation_cfg.get("n_repeats", 10),
                output_dir=output_dir,
            )

        logger.info(f"=== Lakh Validation complete: {len(pairwise_rows)} rows ===")
        return {
            "pairwise_rows": len(pairwise_rows),
            "valid_rows": summary["n_valid"],
            "output_dir": str(output_dir),
            "fmd_range": summary["fmd_range"],
            "ablation_results": ablation_results,
            "outputs": {
                "csv": str(csv_path),
                "summary_json": str(output_dir / "lakh_validation_summary.json"),
                **{k: v for k, v in diag_outputs.items()},
            },
        }

    def _run_sample_size_ablation(
        self,
        embeddings_cache: Dict[str, Dict[str, np.ndarray]],
        genre_a: str,
        genre_b: str,
        sizes: List[int] = None,
        n_repeats: int = 10,
        output_dir: Path = None,
    ) -> Dict:
        """Run sample size ablation study.

        For each variant and each sample size, subsample embeddings n_repeats
        times and compute FMD. Reports mean/std/CI per (variant, size).

        Args:
            embeddings_cache: variant_name → {genre: embeddings_array}
            genre_a, genre_b: Genre labels.
            sizes: List of sample sizes to test.
            n_repeats: Number of subsampling repeats per size.
            output_dir: Where to save CSV results.

        Returns:
            Dict with summary statistics.
        """
        if sizes is None:
            sizes = [50, 100, 200, 500]

        rng = np.random.default_rng(self.seed)
        rows = []

        for variant_name, embs in embeddings_cache.items():
            emb_a = embs.get(genre_a)
            emb_b = embs.get(genre_b)
            if emb_a is None or emb_b is None:
                continue

            max_a = emb_a.shape[0]
            max_b = emb_b.shape[0]

            for size in sizes:
                actual_size_a = min(size, max_a)
                actual_size_b = min(size, max_b)

                if actual_size_a < 2 or actual_size_b < 2:
                    continue

                fmd_values = []
                for _ in range(n_repeats):
                    idx_a = rng.choice(max_a, size=actual_size_a, replace=False)
                    idx_b = rng.choice(max_b, size=actual_size_b, replace=False)
                    fmd_val = float(self.fmd.compute_fmd(emb_a[idx_a], emb_b[idx_b]))
                    fmd_values.append(fmd_val)

                arr = np.array(fmd_values)
                rows.append({
                    "variant": variant_name,
                    "sample_size": size,
                    "actual_size_a": actual_size_a,
                    "actual_size_b": actual_size_b,
                    "n_repeats": n_repeats,
                    "fmd_mean": float(np.mean(arr)),
                    "fmd_std": float(np.std(arr)),
                    "fmd_min": float(np.min(arr)),
                    "fmd_max": float(np.max(arr)),
                    "fmd_ci_lower": float(np.percentile(arr, 2.5)),
                    "fmd_ci_upper": float(np.percentile(arr, 97.5)),
                })

        if output_dir and rows:
            import csv as csv_mod
            csv_path = Path(output_dir) / "sample_size_ablation.csv"
            fields = list(rows[0].keys())
            with open(csv_path, "w", newline="", encoding="utf-8") as fh:
                writer = csv_mod.DictWriter(fh, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)
            logger.info(f"Sample size ablation saved: {csv_path} ({len(rows)} rows)")

        return {"rows": rows, "sizes": sizes, "n_repeats": n_repeats}
