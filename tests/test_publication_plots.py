"""Tests for publication plot generation."""

from pathlib import Path
import sys
import json

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from experiments.publication_plots import generate_publication_plots


def test_generate_publication_plots_creates_expected_files(tmp_path: Path):
    report_dir = tmp_path / "reports" / "paper"
    report_dir.mkdir(parents=True, exist_ok=True)

    pairwise = pd.DataFrame(
        [
            {"variant": "v1", "dataset_a": "jazz", "dataset_b": "rock", "fmd": 1.2},
            {"variant": "v1", "dataset_a": "classical", "dataset_b": "pop", "fmd": 1.6},
            {"variant": "v2", "dataset_a": "jazz", "dataset_b": "rock", "fmd": 1.1},
            {"variant": "v2", "dataset_a": "classical", "dataset_b": "pop", "fmd": 1.7},
        ]
    )
    pairwise.to_csv(report_dir / "pairwise_fmd.csv", index=False)

    special_rows = pd.DataFrame(
        [
            {
                "variant": "v1",
                "genre_a": "jazz",
                "genre_b": "rock",
                "pair": "jazz vs rock",
                "dataset_a": "jazz",
                "dataset_b": "rock",
                "fmd": 1.2,
            },
            {
                "variant": "v2",
                "genre_a": "jazz",
                "genre_b": "rock",
                "pair": "jazz vs rock",
                "dataset_a": "jazz",
                "dataset_b": "rock",
                "fmd": 1.1,
            },
        ]
    )
    special_rows.to_csv(report_dir / "special_pair_fmd.csv", index=False)

    pd.DataFrame(
        [
            {
                "pair": "jazz vs rock",
                "count": 2,
                "mean_fmd": 1.15,
                "std_fmd": 0.05,
                "min_fmd": 1.1,
                "max_fmd": 1.2,
                "distinguishability_ratio": 1.1,
            }
        ]
    ).to_csv(report_dir / "special_pair_summary.csv", index=False)

    pd.DataFrame(
        [
            {
                "pair": "jazz vs rock",
                "rank": 1,
                "variant": "v1",
                "fmd": 1.2,
                "dataset_a": "jazz",
                "dataset_b": "rock",
            },
            {
                "pair": "jazz vs rock",
                "rank": 2,
                "variant": "v2",
                "fmd": 1.1,
                "dataset_a": "jazz",
                "dataset_b": "rock",
            },
        ]
    ).to_csv(report_dir / "special_pair_top_variants.csv", index=False)

    payload = {
        "ranking": {"stability": {"jazz": 0.92, "rock": 0.88}},
        "expected_eval": {
            "details": [
                {
                    "variant": "v1",
                    "reference": "jazz",
                    "spearman": 0.6,
                    "kendall": 0.4,
                }
            ]
        },
    }
    with open(report_dir / "paper_results.json", "w", encoding="utf-8") as handle:
        json.dump(payload, handle)

    config = {
        "results": {
            "reports_dir": str(tmp_path / "reports"),
            "plots_dir": str(tmp_path / "plots"),
        }
    }

    outputs = generate_publication_plots(config)
    assert outputs
    assert "top_variants_per_pair" in outputs
    for path in outputs.values():
        assert Path(path).exists()


