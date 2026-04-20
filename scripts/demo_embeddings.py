"""Demo script for embedding extraction with embedding models."""

import sys
from pathlib import Path
import numpy as np
import argparse
from loguru import logger

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from utils.config import load_config, setup_logging
from embeddings.extractor import (
    EmbeddingFactory,
    EmbeddingExtractor,
    EmbeddingAnalyzer,
)


def demo_models():
    """Demonstrate available embedding models."""
    print("\n" + "=" * 80)
    print("DEMO 1: Available Embedding Models")
    print("=" * 80)
    
    config = load_config("configs/config.yaml")
    
    available_models = EmbeddingFactory.get_available_models()
    print(f"\nAvailable models ({len(available_models)}):")
    
    for model_name in available_models:
        model = EmbeddingFactory.create_model(config, model_name)
        print(f"  ✓ {model_name}")
        print(f"    - Format type: {model.format_type}")
        print(f"    - Embedding dimension: {model.get_embedding_dim()}")
    
    print("\n" + "=" * 80)


def demo_single_encoding():
    """Demonstrate encoding a single sequence."""
    print("\n" + "=" * 80)
    print("DEMO 2: Single Sequence Encoding")
    print("=" * 80)
    
    config = load_config("configs/config.yaml")
    
    # Sample token sequence (simulated MIDI tokens)
    tokens = [1, 2, 3, 4, 5, 10, 20, 30, 40, 50]
    print(f"\nSample token sequence: {tokens}")
    print(f"Number of tokens: {len(tokens)}")
    
    # Encode with both models
    for model_name in ["MusicBERT", "MusicBERT-large"]:
        print(f"\n--- Encoding with {model_name} ---")
        model = EmbeddingFactory.create_model(config, model_name)
        embedding = model.encode(tokens)
        
        print(f"  Shape: {embedding.shape}")
        print(f"  Dtype: {embedding.dtype}")
        print(f"  Mean: {embedding.mean():.6f}")
        print(f"  Std: {embedding.std():.6f}")
        print(f"  Min: {embedding.min():.6f}")
        print(f"  Max: {embedding.max():.6f}")
    
    print("\n" + "=" * 80)


def demo_batch_encoding():
    """Demonstrate batch encoding of multiple sequences."""
    print("\n" + "=" * 80)
    print("DEMO 3: Batch Encoding")
    print("=" * 80)
    
    config = load_config("configs/config.yaml")
    
    # Create sample token sequences
    num_sequences = 5
    token_sequences = [
        [i + j for j in range(10)]
        for i in range(num_sequences)
    ]
    
    print(f"\nNumber of sequences: {num_sequences}")
    print(f"Tokens per sequence: 10")
    print(f"Sample sequences:")
    for i, seq in enumerate(token_sequences[:3]):
        print(f"  {i}: {seq}")
    
    # Batch encode
    model = EmbeddingFactory.create_model(config, "MusicBERT-large")
    embeddings = model.encode_batch(token_sequences)
    
    print(f"\n--- Batch Encoding Results ---")
    print(f"Output shape: {embeddings.shape}")
    print(f"Dtype: {embeddings.dtype}")
    
    # Compute statistics
    stats = EmbeddingAnalyzer.compute_statistics(embeddings)
    print(f"\nStatistics:")
    print(f"  Mean across samples: {stats['mean'].mean():.6f}")
    print(f"  Std across samples: {stats['std'].mean():.6f}")
    
    print("\n" + "=" * 80)


def demo_extractor():
    """Demonstrate full embedding extraction pipeline."""
    print("\n" + "=" * 80)
    print("DEMO 4: Full Embedding Extraction Pipeline")
    print("=" * 80)
    
    config = load_config("configs/config.yaml")
    
    # Create extractor
    extractor = EmbeddingExtractor(config)
    print(f"\n[OK] EmbeddingExtractor initialized")
    print(f"  Cache enabled: {extractor.use_cache}")
    print(f"  Cache directory: {extractor.cache_dir}")
    
    # Create sample data
    token_sequences = [
        list(range(1, 11)),
        list(range(10, 20)),
        list(range(20, 30)),
        list(range(30, 40)),
    ]
    
    print(f"\nProcessing {len(token_sequences)} token sequences...")
    
    # Extract embeddings with both models
    for model_name in ["MusicBERT", "MusicBERT-large"]:
        print(f"\n--- Extracting with {model_name} ---")
        
        embeddings = extractor.extract_embeddings(token_sequences, model_name)
        
        print(f"  Shape: {embeddings.shape}")
        print(f"  Memory usage: {embeddings.nbytes / 1024 / 1024:.2f} MB")
        
        # Compute and display statistics
        stats = EmbeddingAnalyzer.compute_statistics(embeddings)
        print(f"\n  Statistics:")
        print(f"    Samples: {stats['num_samples']}")
        print(f"    Embedding dim: {stats['embedding_dim']}")
        print(f"    Mean magnitude: {np.linalg.norm(stats['mean']):.6f}")
    
    print("\n" + "=" * 80)


def demo_caching():
    """Demonstrate embedding caching."""
    print("\n" + "=" * 80)
    print("DEMO 5: Embedding Caching")
    print("=" * 80)
    
    config = load_config("configs/config.yaml")
    config["embeddings"]["cache_embeddings"] = True
    
    extractor = EmbeddingExtractor(config)
    print(f"\n[OK] Caching enabled: {extractor.use_cache}")
    
    # First extraction
    token_sequence = [[1, 2, 3, 4, 5]]
    print(f"\nFirst extraction of tokens: {token_sequence[0]}")
    emb1 = extractor.extract_embeddings(token_sequence, "MusicBERT-large")
    print(f"  [OK] Extracted (shape: {emb1.shape})")
    
    # Second extraction (should use cache)
    print(f"\nSecond extraction of same tokens (should use cache)...")
    emb2 = extractor.extract_embeddings(token_sequence, "MusicBERT-large")
    print(f"  [OK] Extracted from cache (shape: {emb2.shape})")
    
    # Verify they're identical
    if np.allclose(emb1, emb2):
        print(f"  [OK] Embeddings are identical (as expected from cache)")
    else:
        print(f"  [ERROR] Embeddings differ (unexpected!)")
    
    # Different sequence should not use cache
    print(f"\nThird extraction of different tokens...")
    token_sequence2 = [[10, 20, 30, 40, 50]]
    emb3 = extractor.extract_embeddings(token_sequence2, "MusicBERT-large")
    print(f"  [OK] Extracted (shape: {emb3.shape})")
    
    if not np.allclose(emb1, emb3):
        print(f"  [OK] Embeddings differ (as expected from different tokens)")
    else:
        print(f"  [ERROR] Embeddings are identical (unexpected!)")
    
    print(f"\nCache directory contents:")
    cache_files = list(extractor.cache_dir.glob("*.npy"))
    print(f"  - .npy files: {len(cache_files)}")
    print(f"  - Total cache size: {sum(f.stat().st_size for f in cache_files) / 1024:.2f} KB")
    
    print("\n" + "=" * 80)


def demo_similarity():
    """Demonstrate computing similarities between embeddings."""
    print("\n" + "=" * 80)
    print("DEMO 6: Embedding Similarity Analysis")
    print("=" * 80)
    
    config = load_config("configs/config.yaml")
    extractor = EmbeddingExtractor(config)
    
    # Create similar and different sequences
    similar_seq1 = list(range(1, 11))
    similar_seq2 = list(range(1, 11))
    different_seq = list(range(100, 110))
    
    print(f"\nSequence 1 (similar to 2): {similar_seq1}")
    print(f"Sequence 2 (similar to 1): {similar_seq2}")
    print(f"Sequence 3 (different): {different_seq}")
    
    # Extract embeddings
    embeddings = extractor.extract_embeddings(
        [similar_seq1, similar_seq2, different_seq],
        "MusicBERT-large"
    )
    
    # Compute pairwise distances
    print(f"\n--- Euclidean Distances ---")
    distances_euclidean = EmbeddingAnalyzer.compute_pairwise_distances(
        embeddings, embeddings, metric="euclidean"
    )
    
    print(f"  Distance 1-2: {distances_euclidean[0, 1]:.6f}")
    print(f"  Distance 1-3: {distances_euclidean[0, 2]:.6f}")
    print(f"  Distance 2-3: {distances_euclidean[1, 2]:.6f}")
    
    print(f"\n--- Cosine Distances ---")
    distances_cosine = EmbeddingAnalyzer.compute_pairwise_distances(
        embeddings, embeddings, metric="cosine"
    )
    
    print(f"  Distance 1-2: {distances_cosine[0, 1]:.6f}")
    print(f"  Distance 1-3: {distances_cosine[0, 2]:.6f}")
    print(f"  Distance 2-3: {distances_cosine[1, 2]:.6f}")
    
    print("\n" + "=" * 80)


def main():
    """Run all demos."""
    parser = argparse.ArgumentParser(
        description="Demo script for embedding extraction"
    )
    parser.add_argument(
        "--demo",
        type=str,
        choices=["all", "models", "single", "batch", "extractor", "cache", "similarity"],
        default="all",
        help="Which demo to run"
    )
    parser.add_argument("--skip-cache", action="store_true", help="Skip caching demo")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging("INFO", "logs/experiment.log")
    
    print("\n" + "=" * 80)
    print("EMBEDDING EXTRACTION - DEMO SCRIPT")
    print("Week 3: Embedding Models Integration")
    print("=" * 80)
    
    # Run selected demos
    demos = {
        "models": demo_models,
        "single": demo_single_encoding,
        "batch": demo_batch_encoding,
        "extractor": demo_extractor,
        "cache": demo_caching,
        "similarity": demo_similarity,
    }
    
    if args.demo == "all":
        if not args.skip_cache:
            demo_models()
            demo_single_encoding()
            demo_batch_encoding()
            demo_extractor()
            demo_caching()
            demo_similarity()
        else:
            demo_models()
            demo_single_encoding()
            demo_batch_encoding()
            demo_extractor()
            demo_similarity()
    else:
        demos[args.demo]()
    
    print("\n[OK] Demo completed successfully!\n")


if __name__ == "__main__":
    main()





