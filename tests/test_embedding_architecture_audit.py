"""Tests for embedding architecture audit and shared-space calibration."""

from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from experiments.embedding_architecture_audit import (  # noqa: E402
	SharedSpaceCalibrator,
	is_symbolic_shared_space_candidate,
)


def test_shared_space_calibrator_reduces_anchor_mismatch():
	rng = np.random.default_rng(123)
	latent = rng.normal(size=(96, 12))

	emb_a = latent + 0.01 * rng.normal(size=latent.shape)

	q, _ = np.linalg.qr(rng.normal(size=(12, 12)))
	emb_b = 3.5 * (latent @ q) + 2.0

	calibrator = SharedSpaceCalibrator(reference_model="model_a", max_dim=12)
	info = calibrator.fit({"model_a": emb_a, "model_b": emb_b})

	aligned_a = calibrator.transform("model_a", emb_a)
	aligned_b = calibrator.transform("model_b", emb_b)

	raw_gap = float(np.mean(np.linalg.norm(emb_a - emb_b, axis=1)))
	aligned_gap = float(np.mean(np.linalg.norm(aligned_a - aligned_b, axis=1)))

	assert info["reference_model"] == "model_a"
	assert info["target_dim"] <= 12
	assert aligned_a.shape == aligned_b.shape
	assert aligned_gap < raw_gap * 0.35


def test_shared_space_calibrator_generalizes_to_holdout_data():
	rng = np.random.default_rng(321)
	latent_train = rng.normal(size=(72, 10))
	latent_test = rng.normal(size=(48, 10))

	q, _ = np.linalg.qr(rng.normal(size=(10, 10)))
	train_a = latent_train + 0.02 * rng.normal(size=latent_train.shape)
	train_b = 2.8 * (latent_train @ q) + 1.5

	test_a = latent_test + 0.02 * rng.normal(size=latent_test.shape)
	test_b = 2.8 * (latent_test @ q) + 1.5

	calibrator = SharedSpaceCalibrator(reference_model="model_a", max_dim=10)
	calibrator.fit({"model_a": train_a, "model_b": train_b})

	aligned_test_a = calibrator.transform("model_a", test_a)
	aligned_test_b = calibrator.transform("model_b", test_b)

	raw_gap = float(np.mean(np.linalg.norm(test_a - test_b, axis=1)))
	aligned_gap = float(np.mean(np.linalg.norm(aligned_test_a - aligned_test_b, axis=1)))

	assert aligned_test_a.shape == aligned_test_b.shape
	assert aligned_gap < raw_gap * 0.45


def test_symbolic_shared_space_candidates_exclude_audio_family():
	assert is_symbolic_shared_space_candidate(
		"MusicBERT",
		{"format_type": "token_text"},
		{"status": "real"},
	)
	assert not is_symbolic_shared_space_candidate(
		"MERT",
		{"format_type": "audio"},
		{"status": "real"},
	)


