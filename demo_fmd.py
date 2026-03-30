"""Demo script for Frechet Music Distance (FMD) calculation."""

import sys
from pathlib import Path
import numpy as np
import argparse
from loguru import logger

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from utils.config import load_config, setup_logging
from embeddings.extractor import EmbeddingExtractor, EmbeddingAnalyzer
from metrics.fmd import FrechetMusicDistance, FMDRanking, FMDComparator


def demo_basic_fmd():
    """Demonstrate basic FMD calculation."""
    print("\n" + "=" * 80)
    print("DEMO 1: Basic FMD Calculation")
    print("=" * 80)
    
    config = load_config("configs/config.yaml")
    fmd_calc = FrechetMusicDistance(config)
    
    # Create sample embedding distributions
    print("\nCreating two embedding distributions...")
    np.random.seed(42)
    embeddings1 = np.random.randn(100, 64)
    embeddings2 = np.random.randn(100, 64) + 1  # Shifted distribution
    
    print(f"  Distribution 1: shape {embeddings1.shape}, mean={embeddings1.mean():.4f}")
    print(f"  Distribution 2: shape {embeddings2.shape}, mean={embeddings2.mean():.4f}")
    
    # Compute FMD
    fmd = fmd_calc.compute_fmd(embeddings1, embeddings2)
    print(f"\nFMD between distributions: {fmd:.6f}")
    
    # Compute FMD with identical distributions
    fmd_identical = fmd_calc.compute_fmd(embeddings1, embeddings1)
    print(f"FMD with identical distribution: {fmd_identical:.6f} (should be ~0)")
    
    print("\n" + "=" * 80)


def demo_fmd_matrix():
    """Demonstrate FMD matrix computation."""
    print("\n" + "=" * 80)
    print("DEMO 2: FMD Matrix (Pairwise Distances)")
    print("=" * 80)
    
    config = load_config("configs/config.yaml")
    fmd_calc = FrechetMusicDistance(config)
    
    # Create multiple embedding sets
    print("\nCreating 5 embedding distributions...")
    np.random.seed(42)
    embeddings_list = [
        ("Classical_1", np.random.randn(50, 64)),
        ("Classical_2", np.random.randn(50, 64) + 0.1),  # Similar to Classical_1
        ("Pop_1", np.random.randn(50, 64) + 3),  # Different from Classical
        ("Pop_2", np.random.randn(50, 64) + 3.1),  # Similar to Pop_1
        ("Jazz_1", np.random.randn(50, 64) + 1.5),  # Different from both
    ]
    
    # Compute batch FMD
    result = fmd_calc.compute_batch_fmd(embeddings_list)
    
    print(f"\nFMD Matrix Statistics:")
    print(f"  Mean FMD: {result['mean_fmd']:.6f}")
    print(f"  Median FMD: {result['median_fmd']:.6f}")
    print(f"  Std FMD: {result['std_fmd']:.6f}")
    print(f"  Min FMD: {result['min_fmd']:.6f}")
    print(f"  Max FMD: {result['max_fmd']:.6f}")
    
    # Display matrix as formatted table
    print(f"\nFMD Matrix:")
    print("         ", end="")
    for name in result["names"]:
        print(f"{name:12}", end=" ")
    print()
    
    for i, name in enumerate(result["names"]):
        print(f"{name:8}", end=" ")
        for j in range(len(result["names"])):
            fmd_val = result["fmd_matrix"][i, j]
            print(f"{fmd_val:12.4f}", end=" ")
        print()
    
    print("\n" + "=" * 80)


def demo_fmd_ranking():
    """Demonstrate FMD-based ranking."""
    print("\n" + "=" * 80)
    print("DEMO 3: FMD-Based Ranking")
    print("=" * 80)
    
    config = load_config("configs/config.yaml")
    fmd_calc = FrechetMusicDistance(config)
    
    # Create embeddings
    np.random.seed(42)
    embeddings_list = [
        ("Jazz_Reference", np.random.randn(100, 64)),
        ("Jazz_Similar_1", np.random.randn(100, 64) + 0.2),
        ("Jazz_Similar_2", np.random.randn(100, 64) + 0.3),
        ("Pop", np.random.randn(100, 64) + 2),
        ("Classical", np.random.randn(100, 64) + 4),
    ]
    
    # Compute batch FMD
    result = fmd_calc.compute_batch_fmd(embeddings_list)
    fmd_matrix = result["fmd_matrix"]
    names = result["names"]
    
    # Rank by reference (Jazz_Reference)
    reference_idx = 0
    ranking = FMDRanking.rank_by_fmd(fmd_matrix, reference_idx)
    
    print(f"\nRanking from '{names[reference_idx]}' (reference):")
    print(f"  Rank  | Dataset            | Distance")
    print(f"  ------|--------------------|-----------")
    for rank, idx in enumerate(ranking["ranking"]):
        dist = ranking["distances"][rank]
        marker = " [REF]" if idx == reference_idx else ""
        print(f"  {rank+1:4}  | {names[idx]:18} | {dist:9.6f}{marker}")
    
    print("\n" + "=" * 80)


def demo_embedding_fmd_integration():
    """Demonstrate FMD with real embedding extraction."""
    print("\n" + "=" * 80)
    print("DEMO 4: Embedding Extraction + FMD Integration")
    print("=" * 80)
    
    config = load_config("configs/config.yaml")
    config["embeddings"]["cache_embeddings"] = False
    
    # Extract embeddings from token sequences
    print("\nExtracting embeddings...")
    token_sequences_list = [
        ("Song_A_v1", list(range(1, 11))),
        ("Song_A_v2", list(range(1, 11))),  # Similar to v1
        ("Song_B_v1", list(range(100, 110))),
        ("Song_B_v2", list(range(100, 110))),  # Similar to v1
    ]
    
    extractor = EmbeddingExtractor(config)
    embeddings_list = []
    
    for name, tokens in token_sequences_list:
        emb = extractor.extract_embeddings([tokens], "CLaMP-2")[0]
        embeddings_list.append((name, emb))
        print(f"  [{len(embeddings_list)}] Extracted embedding for {name}")
    
    # Compute FMD
    print("\nComputing pairwise FMD...")
    fmd_calc = FrechetMusicDistance(config)
    result = fmd_calc.compute_batch_fmd(embeddings_list)
    
    print(f"\nFMD Results:")
    print(f"  Mean distance: {result['mean_fmd']:.6f}")
    print(f"  Std deviation: {result['std_fmd']:.6f}")
    
    # Analyze similarity within groups
    print(f"\nIntra-group distances (should be similar):")
    print(f"  Song_A_v1 vs Song_A_v2: {result['fmd_matrix'][0, 1]:.6f}")
    print(f"  Song_B_v1 vs Song_B_v2: {result['fmd_matrix'][2, 3]:.6f}")
    
    print(f"\nInter-group distances (should be different):")
    print(f"  Song_A_v1 vs Song_B_v1: {result['fmd_matrix'][0, 2]:.6f}")
    print(f"  Song_A_v1 vs Song_B_v2: {result['fmd_matrix'][0, 3]:.6f}")
    
    print("\n" + "=" * 80)


def demo_fmd_stability():
    """Demonstrate ranking stability across configurations."""
    print("\n" + "=" * 80)
    print("DEMO 5: Ranking Stability Analysis")
    print("=" * 80)
    
    config = load_config("configs/config.yaml")
    fmd_calc = FrechetMusicDistance(config)
    
    np.random.seed(42)
    
    # Create different configurations
    print("\nSimulating 3 tokenization configurations...")
    
    embeddings_base = [
        ("Dataset_A", np.random.randn(100, 64)),
        ("Dataset_B", np.random.randn(100, 64) + 1),
        ("Dataset_C", np.random.randn(100, 64) + 2),
    ]
    
    rankings = {}
    fmd_matrices = {}
    
    for config_name in ["REMI", "TSD", "Octuple"]:
        # Slightly perturb embeddings for each config
        embeddings_perturbed = [
            (name, emb + np.random.randn(*emb.shape) * 0.1)
            for name, emb in embeddings_base
        ]
        
        result = fmd_calc.compute_batch_fmd(embeddings_perturbed)
        matrix = result["fmd_matrix"]
        
        # Get ranking from first dataset
        ranking = FMDRanking.rank_by_fmd(matrix, reference_idx=0)
        rankings[config_name] = ranking["ranking"]
        fmd_matrices[config_name] = matrix
        
        print(f"  [{config_name}] FMD matrix computed")
    
    # Compute stability
    stability = FMDRanking.compute_ranking_stability(rankings)
    print(f"\nRanking Stability Score: {stability:.4f}")
    print(f"  Interpretation: {'High consistency' if stability > 0.7 else 'Low consistency'} across configs")
    
    print(f"\nRankings per configuration:")
    for config_name, ranking in rankings.items():
        dataset_names = ["Dataset_A", "Dataset_B", "Dataset_C"]
        ranked_names = [dataset_names[i] for i in ranking]
        print(f"  {config_name:10}: {' > '.join(ranked_names)}")
    
    print("\n" + "=" * 80)


def demo_tokenization_sensitivity():
    """Demonstrate FMD sensitivity to tokenization."""
    print("\n" + "=" * 80)
    print("DEMO 6: FMD Sensitivity to Tokenization")
    print("=" * 80)
    
    config = load_config("configs/config.yaml")
    config["embeddings"]["cache_embeddings"] = False
    
    print("\nSimulating different tokenization impacts on FMD...")
    
    # Base datasets
    datasets = {
        "Pop": list(range(1, 21)),
        "Classical": list(range(100, 120)),
        "Jazz": list(range(200, 220)),
    }
    
    results_per_tokenizer = {}
    extractor = EmbeddingExtractor(config)
    
    for tokenizer_name in ["REMI", "TSD", "Octuple"]:
        print(f"\n[{tokenizer_name}]:")
        
        # Extract embeddings using same model
        embeddings_list = []
        for dataset_name, tokens in datasets.items():
            # In real scenario, different tokenizers would produce different token sequences
            # Here we just create different embeddings to simulate
            base_emb = extractor.extract_embeddings([tokens], "CLaMP-2")[0]
            # Add tokenizer-specific perturbation
            perturbation = {
                "REMI": np.random.randn(base_emb.shape[0]) * 0.05,
                "TSD": np.random.randn(base_emb.shape[0]) * 0.08,
                "Octuple": np.random.randn(base_emb.shape[0]) * 0.10,
            }
            perturbed_emb = base_emb + perturbation.get(tokenizer_name, 0)
            embeddings_list.append((dataset_name, perturbed_emb))
        
        # Compute FMD
        fmd_calc = FrechetMusicDistance(config)
        result = fmd_calc.compute_batch_fmd(embeddings_list)
        results_per_tokenizer[tokenizer_name] = result
        
        print(f"  Mean FMD: {result['mean_fmd']:.6f}")
        print(f"  Std FMD:  {result['std_fmd']:.6f}")
        print(f"  Pop vs Classical: {result['fmd_matrix'][0, 1]:.6f}")
        print(f"  Pop vs Jazz:      {result['fmd_matrix'][0, 2]:.6f}")
    
    # Compare sensitivity
    print(f"\nFMD Stability Across Tokenizers:")
    mean_fmds = [results_per_tokenizer[tok]["mean_fmd"] for tok in ["REMI", "TSD", "Octuple"]]
    print(f"  Mean FMDs: {[f'{m:.6f}' for m in mean_fmds]}")
    print(f"  Variation: {np.std(mean_fmds):.6f} (lower = more stable)")
    
    print("\n" + "=" * 80)


def main():
    """Run all demos."""
    parser = argparse.ArgumentParser(
        description="Demo script for Frechet Music Distance (FMD) calculation"
    )
    parser.add_argument(
        "--demo",
        type=str,
        choices=["all", "basic", "matrix", "ranking", "integration", "stability", "tokenization"],
        default="all",
        help="Which demo to run"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging("INFO", "logs/experiment.log")
    
    print("\n" + "=" * 80)
    print("FRECHET MUSIC DISTANCE (FMD) - DEMO SCRIPT")
    print("Week 4: FMD Calculation and Analysis")
    print("=" * 80)
    
    # Run selected demos
    demos = {
        "basic": demo_basic_fmd,
        "matrix": demo_fmd_matrix,
        "ranking": demo_fmd_ranking,
        "integration": demo_embedding_fmd_integration,
        "stability": demo_fmd_stability,
        "tokenization": demo_tokenization_sensitivity,
    }
    
    if args.demo == "all":
        demo_basic_fmd()
        demo_fmd_matrix()
        demo_fmd_ranking()
        demo_embedding_fmd_integration()
        demo_fmd_stability()
        demo_tokenization_sensitivity()
    else:
        demos[args.demo]()
    
    print("\n[OK] Demo completed successfully!\n")


if __name__ == "__main__":
    main()

