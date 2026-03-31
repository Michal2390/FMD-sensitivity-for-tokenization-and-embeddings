"""Tests for paper-oriented experiment pipeline."""

from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import experiments.paper_pipeline as pipeline_mod


class _FakeDatasetManager:
    def __init__(self, config):
        self.config = config

    def list_midi_files(self, dataset_name, processed=False, limit=None):
        return []


class _FakePreprocessor:
    def __init__(self, config):
        self.config = config


class _FakeTokenizerPipeline:
    def __init__(self, config):
        self.config = config
        self.tokenizers = {}


class _FakeEmbeddings:
    def __init__(self, config):
        self.config = config


class _FakeFMD:
    def __init__(self, config):
        self.config = config

    def compute_fmd(self, emb1, emb2):
        mean1 = np.mean(emb1, axis=0)
        mean2 = np.mean(emb2, axis=0)
        return float(np.linalg.norm(mean1 - mean2))

    def compute_batch_fmd(self, embeddings_list):
        n = len(embeddings_list)
        matrix = np.zeros((n, n), dtype=float)
        names = [name for name, _ in embeddings_list]
        for i in range(n):
            for j in range(i, n):
                dist = self.compute_fmd(embeddings_list[i][1], embeddings_list[j][1])
                matrix[i, j] = dist
                matrix[j, i] = dist
        return {"fmd_matrix": matrix, "names": names}


@pytest.fixture
def light_config(tmp_path: Path):
    return {
        "data": {
            "raw_data_dir": str(tmp_path / "raw"),
            "processed_data_dir": str(tmp_path / "processed"),
            "embeddings_dir": str(tmp_path / "embeddings"),
            "datasets": [{"name": "maestro"}, {"name": "midicaps"}, {"name": "pop909"}],
        },
        "tokenization": {
            "tokenizers": [
                {"type": "REMI", "params": {}},
                {"type": "Octuple", "params": {}},
            ]
        },
        "embeddings": {
            "models": [{"name": "CLaMP-1"}, {"name": "CLaMP-2"}],
        },
        "results": {"reports_dir": str(tmp_path / "reports")},
        "paper": {
            "seed": 42,
            "max_files_per_dataset": 2,
            "synthetic_fallback_samples": 4,
            "top_variants_per_pair": 2,
            "genre_aliases": {"jazz": "maestro", "rock": "midicaps"},
            "special_pairs": [("jazz", "rock")],
            "expected_orders": [
                {"reference": "maestro", "order": ["maestro", "midicaps", "pop909"]}
            ],
        },
        "experiments": {"exp5_cross_genre": {"pairs": [("maestro", "midicaps")]}}
    }


@pytest.fixture
def runner(monkeypatch, light_config):
    monkeypatch.setattr(pipeline_mod, "DatasetManager", _FakeDatasetManager)
    monkeypatch.setattr(pipeline_mod, "MIDIPreprocessor", _FakePreprocessor)
    monkeypatch.setattr(pipeline_mod, "TokenizationPipeline", _FakeTokenizerPipeline)
    monkeypatch.setattr(pipeline_mod, "EmbeddingExtractor", _FakeEmbeddings)
    monkeypatch.setattr(pipeline_mod, "FrechetMusicDistance", _FakeFMD)

    runner = pipeline_mod.PaperExperimentRunner(light_config)

    # Deterministic synthetic embeddings: dataset-specific offset.
    offsets = {"maestro": 0.0, "midicaps": 1.0, "pop909": 2.0}

    def _fake_extract(dataset_name, variant):
        base = offsets.get(dataset_name, 0.0)
        rng = np.random.default_rng(abs(hash((dataset_name, variant.name))) % (2**32))
        return (rng.normal(size=(6, 8)) + base).astype(np.float32)

    runner._extract_dataset_embeddings = _fake_extract  # type: ignore[attr-defined]
    return runner


def test_build_variants_count(runner):
    variants = runner.build_variants()
    assert len(variants) == 2 * 2 * 4


def test_parse_pairs_handles_mixed_formats():
    raw = [("a", "b"), "('c', 'd')", "(e, f)"]
    pairs = pipeline_mod.PaperExperimentRunner._parse_pairs(raw)
    assert pairs == [("a", "b"), ("c", "d"), ("e", "f")]


def test_run_pairwise_and_outputs(runner):
    variants = runner.build_variants(tokenizers=["REMI"], models=["CLaMP-1"], preprocessing_grid=[(False, False)])
    rows = runner.run_pairwise_benchmark(variants)
    assert len(rows) == 1
    assert rows[0]["dataset_a"] == "maestro"
    assert rows[0]["dataset_b"] == "midicaps"

    ranking = runner.run_ranking_benchmark(variants)
    assert "stability" in ranking
    assert "maestro" in ranking["stability"]

    expected = runner.evaluate_expected_order(ranking)
    special = runner.compute_special_pair_metrics(rows)
    assert special["available"] is True
    assert len(special["rows"]) == 1
    assert len(special["top_variants"]) == 1
    assert special["top_variants"][0]["rank"] == 1

    files = runner.save_outputs(rows, ranking, expected, special)
    assert Path(files["json"]).exists()
    assert Path(files["csv"]).exists()
    assert Path(files["markdown"]).exists()
    assert Path(files["special_csv"]).exists()
    assert Path(files["special_summary_csv"]).exists()
    assert Path(files["special_top_variants_csv"]).exists()


def test_quick_mode_runs(runner):
    result = runner.run_quick()
    assert result["pairwise_rows"] >= 1
    assert "outputs" in result


def test_pairwise_all_combinations(monkeypatch, light_config):
    monkeypatch.setattr(pipeline_mod, "DatasetManager", _FakeDatasetManager)
    monkeypatch.setattr(pipeline_mod, "MIDIPreprocessor", _FakePreprocessor)
    monkeypatch.setattr(pipeline_mod, "TokenizationPipeline", _FakeTokenizerPipeline)
    monkeypatch.setattr(pipeline_mod, "EmbeddingExtractor", _FakeEmbeddings)
    monkeypatch.setattr(pipeline_mod, "FrechetMusicDistance", _FakeFMD)

    light_config["paper"]["compare_all_pairs"] = True
    local_runner = pipeline_mod.PaperExperimentRunner(light_config)
    local_runner._extract_dataset_embeddings = lambda dataset_name, variant: np.zeros((5, 8), dtype=np.float32)

    variants = local_runner.build_variants(tokenizers=["REMI"], models=["CLaMP-1"], preprocessing_grid=[(False, False)])
    rows = local_runner.run_pairwise_benchmark(variants)
    # 3 datasets => C(3,2) = 3 pairs.
    assert len(rows) == 3


def test_strict_mode_marks_invalid_pair(monkeypatch, light_config):
    monkeypatch.setattr(pipeline_mod, "DatasetManager", _FakeDatasetManager)
    monkeypatch.setattr(pipeline_mod, "MIDIPreprocessor", _FakePreprocessor)
    monkeypatch.setattr(pipeline_mod, "TokenizationPipeline", _FakeTokenizerPipeline)
    monkeypatch.setattr(pipeline_mod, "EmbeddingExtractor", _FakeEmbeddings)
    monkeypatch.setattr(pipeline_mod, "FrechetMusicDistance", _FakeFMD)

    light_config["paper"]["fallback_mode"] = "strict"
    local_runner = pipeline_mod.PaperExperimentRunner(light_config)

    def _fake_extract(dataset_name, variant):
        if dataset_name == "midicaps":
            return {"embeddings": None, "source": "missing", "real_files": 0, "total_files": 0}
        return np.zeros((5, 8), dtype=np.float32)

    local_runner._extract_dataset_embeddings = _fake_extract  # type: ignore[attr-defined]
    variants = local_runner.build_variants(tokenizers=["REMI"], models=["CLaMP-1"], preprocessing_grid=[(False, False)])
    rows = local_runner.run_pairwise_benchmark(variants)
    assert len(rows) == 1
    assert rows[0]["valid"] is False
    assert rows[0]["fmd"] is None


def test_hard_strict_raises_immediately(monkeypatch, light_config):
    monkeypatch.setattr(pipeline_mod, "DatasetManager", _FakeDatasetManager)
    monkeypatch.setattr(pipeline_mod, "MIDIPreprocessor", _FakePreprocessor)
    monkeypatch.setattr(pipeline_mod, "TokenizationPipeline", _FakeTokenizerPipeline)
    monkeypatch.setattr(pipeline_mod, "EmbeddingExtractor", _FakeEmbeddings)
    monkeypatch.setattr(pipeline_mod, "FrechetMusicDistance", _FakeFMD)

    light_config["paper"]["fallback_mode"] = "hard_strict"
    local_runner = pipeline_mod.PaperExperimentRunner(light_config)

    def _fake_extract(dataset_name, variant):
        if dataset_name == "midicaps":
            return {"embeddings": None, "source": "missing", "real_files": 0, "total_files": 0}
        return np.zeros((5, 8), dtype=np.float32)

    local_runner._extract_dataset_embeddings = _fake_extract  # type: ignore[attr-defined]
    variants = local_runner.build_variants(tokenizers=["REMI"], models=["CLaMP-1"], preprocessing_grid=[(False, False)])
    with pytest.raises(RuntimeError):
        local_runner.run_pairwise_benchmark(variants)


def test_split_rows_and_effects(runner):
    variants = runner.build_variants(tokenizers=["REMI", "Octuple"], models=["CLaMP-1"], preprocessing_grid=[(False, False)])
    rows = runner.run_pairwise_benchmark(variants)
    split = runner._split_pairwise_rows(rows)
    assert len(split["all"]) == len(rows)
    assert isinstance(split["real_only"], list)

    effects = runner.compute_variant_effects(split["all"])
    assert "tokenizer_deltas" in effects
    assert "model_deltas" in effects


