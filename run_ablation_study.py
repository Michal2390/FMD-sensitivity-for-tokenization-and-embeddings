"""Ablation study experiments for Week 5."""

import sys
from pathlib import Path
import numpy as np
import json
from typing import Dict, List, Tuple
from loguru import logger

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from utils.config import load_config, setup_logging
from preprocessing.processor import MIDIPreprocessor, PreprocessingPipeline
from tokenization.tokenizer import TokenizationPipeline
from embeddings.extractor import EmbeddingExtractor
from metrics.fmd import FrechetMusicDistance, FMDRanking


class AblationStudy:
    """Conduct ablation study experiments to measure FMD sensitivity."""

    def __init__(self, config: Dict):
        """
        Initialize ablation study.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.preprocessor = MIDIPreprocessor(config)
        self.tokenizer_pipeline = TokenizationPipeline(config)
        self.embedding_extractor = EmbeddingExtractor(config)
        self.fmd_calc = FrechetMusicDistance(config)
        
        self.results_dir = Path("results/ablation_study")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("AblationStudy initialized")

    def create_variants(self, midi_data, remove_velocity=False, hard_quantize=False) -> Dict:
        """
        Create MIDI variants by applying preprocessing modifications.

        Args:
            midi_data: Original MIDI data
            remove_velocity: Whether to remove velocity
            hard_quantize: Whether to apply hard quantization

        Returns:
            Dictionary of variants
        """
        variants = {
            "original": midi_data
        }

        # Variant 1: Remove velocity only
        if remove_velocity:
            midi_no_vel = self.preprocessor.remove_velocity(midi_data)
            variants["no_velocity"] = midi_no_vel

        # Variant 2: Hard quantize only
        if hard_quantize:
            midi_quantized = self.preprocessor.quantize_time(midi_data)
            variants["quantized"] = midi_quantized

        # Variant 3: Both modifications
        if remove_velocity and hard_quantize:
            midi_combined = self.preprocessor.remove_velocity(midi_data)
            midi_combined = self.preprocessor.quantize_time(midi_combined)
            variants["no_velocity_quantized"] = midi_combined

        return variants

    def extract_embeddings_for_variants(
        self, variants: Dict, tokenizer_name: str, model_name: str
    ) -> Dict:
        """
        Extract embeddings for all MIDI variants.

        Args:
            variants: Dictionary of MIDI variants
            tokenizer_name: Name of tokenizer to use
            model_name: Name of embedding model to use

        Returns:
            Dictionary of embeddings for each variant
        """
        embeddings_dict = {}

        for variant_name, midi_data in variants.items():
            try:
                # Tokenize
                tokens = self.tokenizer_pipeline.tokenizers[tokenizer_name].encode_midi_object(midi_data)
                
                # Extract embedding
                embedding = self.embedding_extractor.extract_embeddings([tokens], model_name)[0]
                embeddings_dict[variant_name] = embedding

                logger.info(f"Extracted embedding for {variant_name}: shape {embedding.shape}")

            except Exception as e:
                logger.error(f"Failed to extract embedding for {variant_name}: {e}")
                embeddings_dict[variant_name] = None

        return embeddings_dict

    def compute_fmd_impact(self, embeddings_dict: Dict) -> Dict:
        """
        Compute FMD impact of preprocessing modifications.

        Args:
            embeddings_dict: Dictionary of embeddings for variants

        Returns:
            Dictionary with FMD differences
        """
        results = {}

        # Get original embedding as reference
        original_emb = embeddings_dict.get("original")
        if original_emb is None:
            logger.error("Original embedding not found")
            return results

        for variant_name, variant_emb in embeddings_dict.items():
            if variant_name == "original" or variant_emb is None:
                continue

            # Compute FMD between original and variant
            fmd = self.fmd_calc.compute_fmd(
                original_emb.reshape(1, -1),
                variant_emb.reshape(1, -1)
            )

            results[variant_name] = {
                "fmd_distance": fmd,
                "fmd_percent_change": abs(fmd) / (np.linalg.norm(original_emb) + 1e-8) * 100
            }

            logger.info(f"{variant_name}: FMD={fmd:.6f}, % change={results[variant_name]['fmd_percent_change']:.2f}%")

        return results

    def run_ablation_experiment(
        self, midi_files: List[Path], tokenizer_name: str, model_name: str
    ) -> Dict:
        """
        Run complete ablation study on multiple MIDI files.

        Args:
            midi_files: List of MIDI file paths
            tokenizer_name: Name of tokenizer
            model_name: Name of embedding model

        Returns:
            Complete ablation study results
        """
        logger.info(f"Running ablation study on {len(midi_files)} files")
        logger.info(f"Tokenizer: {tokenizer_name}, Model: {model_name}")

        results_per_file = {}
        aggregate_results = {
            "no_velocity": [],
            "quantized": [],
            "no_velocity_quantized": []
        }

        for i, midi_file in enumerate(midi_files, 1):
            logger.info(f"Processing {i}/{len(midi_files)}: {midi_file.name}")

            try:
                # Load MIDI
                midi_data = self.preprocessor.load_midi(midi_file)
                if midi_data is None:
                    continue

                # Create variants
                variants = self.create_variants(
                    midi_data,
                    remove_velocity=True,
                    hard_quantize=True
                )

                # Extract embeddings
                embeddings = self.extract_embeddings_for_variants(
                    variants, tokenizer_name, model_name
                )

                # Compute FMD impact
                fmd_results = self.compute_fmd_impact(embeddings)

                results_per_file[midi_file.name] = fmd_results

                # Aggregate
                for variant, result in fmd_results.items():
                    if variant in aggregate_results:
                        aggregate_results[variant].append(result["fmd_distance"])

            except Exception as e:
                logger.error(f"Error processing {midi_file}: {e}")

        # Compute aggregate statistics
        summary = {
            "total_files": len(midi_files),
            "tokenizer": tokenizer_name,
            "model": model_name,
            "per_file_results": results_per_file,
            "aggregate_statistics": {}
        }

        for variant, distances in aggregate_results.items():
            if distances:
                summary["aggregate_statistics"][variant] = {
                    "mean_fmd": float(np.mean(distances)),
                    "std_fmd": float(np.std(distances)),
                    "min_fmd": float(np.min(distances)),
                    "max_fmd": float(np.max(distances)),
                    "samples": len(distances)
                }

        return summary

    def compare_tokenization_sensitivity(
        self, midi_files: List[Path], model_name: str
    ) -> Dict:
        """
        Compare FMD sensitivity across different tokenizers.

        Args:
            midi_files: List of MIDI files
            model_name: Name of embedding model

        Returns:
            Comparison results across tokenizers
        """
        tokenizers = ["REMI", "TSD", "Octuple", "MIDI-Like"]
        results = {}

        for tokenizer in tokenizers:
            logger.info(f"Testing tokenizer: {tokenizer}")
            result = self.run_ablation_experiment(midi_files, tokenizer, model_name)
            results[tokenizer] = result

        return results

    def compare_model_sensitivity(
        self, midi_files: List[Path], tokenizer_name: str
    ) -> Dict:
        """
        Compare FMD sensitivity across different embedding models.

        Args:
            midi_files: List of MIDI files
            tokenizer_name: Name of tokenizer

        Returns:
            Comparison results across models
        """
        models = ["CLaMP-1", "CLaMP-2"]
        results = {}

        for model in models:
            logger.info(f"Testing model: {model}")
            result = self.run_ablation_experiment(midi_files, tokenizer_name, model)
            results[model] = result

        return results

    def save_results(self, results: Dict, filename: str):
        """Save results to JSON file."""
        output_file = self.results_dir / filename
        with open(output_file, "w") as f:
            # Convert numpy types to native Python types
            json_results = json.loads(json.dumps(results, default=str))
            json.dump(json_results, f, indent=2)
        logger.info(f"Results saved to {output_file}")

    def generate_report(self, results: Dict) -> str:
        """Generate human-readable report from results."""
        report = []
        report.append("\n" + "=" * 80)
        report.append("ABLATION STUDY REPORT")
        report.append("=" * 80)

        if "aggregate_statistics" in results:
            # Single experiment report
            report.append(f"\nTokenizer: {results['tokenizer']}")
            report.append(f"Model: {results['model']}")
            report.append(f"Files analyzed: {results['total_files']}")

            report.append("\nImpact on FMD:")
            for variant, stats in results["aggregate_statistics"].items():
                report.append(f"\n  {variant}:")
                report.append(f"    Mean FMD change: {stats['mean_fmd']:.6f}")
                report.append(f"    Std deviation: {stats['std_fmd']:.6f}")
                report.append(f"    Range: [{stats['min_fmd']:.6f}, {stats['max_fmd']:.6f}]")

        else:
            # Comparison report
            report.append("\nComparison across configurations:")
            for config_name, config_results in results.items():
                report.append(f"\n{config_name}:")
                if "aggregate_statistics" in config_results:
                    for variant, stats in config_results["aggregate_statistics"].items():
                        report.append(f"  {variant}: mean_fmd={stats['mean_fmd']:.6f}")

        report.append("\n" + "=" * 80)
        return "\n".join(report)


def main():
    """Main entry point for ablation study."""
    setup_logging("INFO", "logs/ablation_study.log")
    config = load_config("configs/config.yaml")

    logger.info("Starting Ablation Study - Week 5")

    # Create ablation study instance
    ablation = AblationStudy(config)

    # For demo, create synthetic MIDI data
    logger.info("Creating demo MIDI files for testing")
    midi_dir = Path("data/raw")
    
    if not list(midi_dir.glob("*.mid")):
        logger.warning("No MIDI files found in data/raw/. Creating demo files...")
        # Create dummy MIDI files for testing
        import pretty_midi
        
        for i in range(3):
            midi = pretty_midi.PrettyMIDI()
            midi.instruments.append(pretty_midi.Instrument(program=0))
            
            # Add some notes
            for j in range(5):
                note = pretty_midi.Note(
                    velocity=100,
                    pitch=60 + j,
                    start=j,
                    end=j + 1
                )
                midi.instruments[0].notes.append(note)
            
            midi.write(midi_dir / f"demo_{i}.mid")
            logger.info(f"Created demo file: demo_{i}.mid")

    # Get MIDI files
    midi_files = list(midi_dir.glob("*.mid"))[:3]  # Use first 3 files for demo
    
    if not midi_files:
        logger.error("No MIDI files available for ablation study")
        return

    logger.info(f"Using {len(midi_files)} MIDI files for ablation study")

    # Run Experiment 3: Velocity removal ablation
    logger.info("\n=== EXPERIMENT 3: Velocity Removal Impact ===")
    ablation_results_velocity = ablation.run_ablation_experiment(
        midi_files, "REMI", "CLaMP-2"
    )
    ablation.save_results(ablation_results_velocity, "ablation_velocity.json")
    logger.info(ablation.generate_report(ablation_results_velocity))

    # Run Experiment 4: Hard quantization ablation
    logger.info("\n=== EXPERIMENT 4: Hard Quantization Impact ===")
    ablation_results_quantization = ablation.run_ablation_experiment(
        midi_files, "REMI", "CLaMP-2"
    )
    ablation.save_results(ablation_results_quantization, "ablation_quantization.json")
    logger.info(ablation.generate_report(ablation_results_quantization))

    # Run tokenization sensitivity comparison
    logger.info("\n=== TOKENIZATION SENSITIVITY COMPARISON ===")
    tokenization_comparison = ablation.compare_tokenization_sensitivity(
        midi_files, "CLaMP-2"
    )
    ablation.save_results(tokenization_comparison, "tokenization_sensitivity.json")
    logger.info(ablation.generate_report(tokenization_comparison))

    # Run model sensitivity comparison
    logger.info("\n=== MODEL SENSITIVITY COMPARISON ===")
    model_comparison = ablation.compare_model_sensitivity(
        midi_files, "REMI"
    )
    ablation.save_results(model_comparison, "model_sensitivity.json")
    logger.info(ablation.generate_report(model_comparison))

    logger.info("\nAblation study complete!")


if __name__ == "__main__":
    main()

