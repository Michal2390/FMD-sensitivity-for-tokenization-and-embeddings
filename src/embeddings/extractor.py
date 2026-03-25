"""Embedding extraction module using CLaMP models."""

from pathlib import Path
from typing import Dict, List
from loguru import logger
import numpy as np
from abc import ABC, abstractmethod


class EmbeddingModel(ABC):
    """Base class for embedding models."""

    def __init__(self, config: Dict, model_name: str):
        """
        Initialize embedding model.

        Args:
            config: Configuration dictionary
            model_name: Name of the model
        """
        self.config = config
        self.model_name = model_name
        self.device = config["embeddings"].get("device", "cpu")
        logger.info(f"Initialized {model_name} embedding model")

    @abstractmethod
    def encode(self, tokens: List[int]) -> np.ndarray:
        """
        Encode tokens to embeddings.

        Args:
            tokens: List of token IDs

        Returns:
            Embedding vector (1D numpy array)
        """
        pass

    @abstractmethod
    def encode_batch(self, token_sequences: List[List[int]]) -> np.ndarray:
        """
        Encode batch of token sequences to embeddings.

        Args:
            token_sequences: List of token sequences

        Returns:
            Embedding matrix (N x D)
        """
        pass

    def get_embedding_dim(self) -> int:
        """Get dimensionality of embeddings."""
        raise NotImplementedError


class CLaMP1Model(EmbeddingModel):
    """CLaMP 1 model - ABC text format based."""

    def __init__(self, config: Dict):
        """Initialize CLaMP 1 model."""
        super().__init__(config, "CLaMP-1")
        self.format_type = "abc"
        self.embedding_dim = 512  # Placeholder

    def encode(self, tokens: List[int]) -> np.ndarray:
        """Encode tokens to embeddings using CLaMP 1."""
        logger.debug(f"Encoding {len(tokens)} tokens with CLaMP-1")
        # Placeholder: Will load and use actual CLaMP 1 model
        return np.random.randn(self.embedding_dim).astype(np.float32)

    def encode_batch(self, token_sequences: List[List[int]]) -> np.ndarray:
        """Encode batch of token sequences."""
        embeddings = np.array([self.encode(seq) for seq in token_sequences])
        return embeddings

    def get_embedding_dim(self) -> int:
        """Get embedding dimensionality."""
        return self.embedding_dim


class CLaMP2Model(EmbeddingModel):
    """CLaMP 2 model - Direct MIDI structure based."""

    def __init__(self, config: Dict):
        """Initialize CLaMP 2 model."""
        super().__init__(config, "CLaMP-2")
        self.format_type = "midi"
        self.embedding_dim = 512  # Placeholder

    def encode(self, tokens: List[int]) -> np.ndarray:
        """Encode tokens to embeddings using CLaMP 2."""
        logger.debug(f"Encoding {len(tokens)} tokens with CLaMP-2")
        # Placeholder: Will load and use actual CLaMP 2 model
        return np.random.randn(self.embedding_dim).astype(np.float32)

    def encode_batch(self, token_sequences: List[List[int]]) -> np.ndarray:
        """Encode batch of token sequences."""
        embeddings = np.array([self.encode(seq) for seq in token_sequences])
        return embeddings

    def get_embedding_dim(self) -> int:
        """Get embedding dimensionality."""
        return self.embedding_dim


class EmbeddingFactory:
    """Factory for creating embedding models."""

    _models = {
        "CLaMP-1": CLaMP1Model,
        "CLaMP-2": CLaMP2Model,
    }

    @classmethod
    def create_model(cls, config: Dict, model_name: str) -> EmbeddingModel:
        """
        Create an embedding model instance.

        Args:
            config: Configuration dictionary
            model_name: Name of the model

        Returns:
            Embedding model instance
        """
        if model_name not in cls._models:
            raise ValueError(f"Unknown model: {model_name}")

        return cls._models[model_name](config)

    @classmethod
    def get_available_models(cls) -> List[str]:
        """Get list of available models."""
        return list(cls._models.keys())


class EmbeddingExtractor:
    """Extract embeddings from tokenized MIDI data."""

    def __init__(self, config: Dict):
        """
        Initialize embedding extractor.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.models = {
            model_name: EmbeddingFactory.create_model(config, model_name)
            for model_name in EmbeddingFactory.get_available_models()
        }
        self.cache = {} if config["embeddings"].get("cache_embeddings", True) else None
        logger.info("EmbeddingExtractor initialized")

    def extract_embeddings(self, token_sequences: List[List[int]], model_name: str) -> np.ndarray:
        """
        Extract embeddings from token sequences.

        Args:
            token_sequences: List of token sequences
            model_name: Name of the model to use

        Returns:
            Embedding matrix (N x D)
        """
        if model_name not in self.models:
            raise ValueError(f"Unknown model: {model_name}")

        model = self.models[model_name]

        # Check cache if enabled
        if self.cache is not None:
            cache_key = (model_name, tuple(map(tuple, token_sequences)))
            if cache_key in self.cache:
                logger.debug(f"Cache hit for {model_name}")
                return self.cache[cache_key]

        # Extract embeddings
        batch_size = self.config["embeddings"].get("batch_size", 32)
        embeddings_list = []

        for i in range(0, len(token_sequences), batch_size):
            batch = token_sequences[i : i + batch_size]
            batch_embeddings = model.encode_batch(batch)
            embeddings_list.append(batch_embeddings)

        embeddings = np.vstack(embeddings_list)

        # Cache if enabled
        if self.cache is not None:
            cache_key = (model_name, tuple(map(tuple, token_sequences)))
            self.cache[cache_key] = embeddings

        return embeddings

    def extract_dataset_embeddings(
        self, token_files: List[Path], output_dir: Path, model_name: str
    ) -> Dict:
        """
        Extract embeddings for multiple tokenized files.

        Args:
            token_files: List of paths to tokenized files
            output_dir: Directory to save embeddings
            model_name: Name of the model to use

        Returns:
            Dictionary with extraction statistics
        """
        stats = {
            "model": model_name,
            "total": len(token_files),
            "successful": 0,
            "failed": 0,
            "embedding_dim": self.models[model_name].get_embedding_dim(),
        }

        output_dir.mkdir(parents=True, exist_ok=True)

        for i, token_file in enumerate(token_files, 1):
            logger.info(f"Extracting embeddings {i}/{len(token_files)}: {token_file.name}")

            try:
                # Load tokens
                with open(token_file, "r") as f:
                    token_str = f.read().strip()
                    tokens = list(map(int, token_str.split()))

                # Extract embeddings
                embedding = self.extract_embeddings([tokens], model_name)[0]

                # Save embeddings
                output_file = output_dir / (token_file.stem + "_embedding.npy")
                np.save(output_file, embedding)

                stats["successful"] += 1
            except Exception as e:
                logger.error(f"Error extracting embeddings for {token_file}: {e}")
                stats["failed"] += 1

        logger.info(
            f"Embedding extraction complete: {stats['successful']} successful, {stats['failed']} failed"
        )
        return stats


class EmbeddingAnalyzer:
    """Analyze embedding statistics."""

    @staticmethod
    def compute_statistics(embeddings: np.ndarray) -> Dict:
        """
        Compute statistics about embeddings.

        Args:
            embeddings: Embedding matrix (N x D)

        Returns:
            Dictionary with statistics
        """
        stats = {
            "mean": np.mean(embeddings, axis=0),
            "std": np.std(embeddings, axis=0),
            "cov": np.cov(embeddings.T),
            "num_samples": embeddings.shape[0],
            "embedding_dim": embeddings.shape[1] if embeddings.ndim > 1 else 1,
        }
        return stats

    @staticmethod
    def compute_pairwise_distances(
        embeddings1: np.ndarray, embeddings2: np.ndarray, metric: str = "euclidean"
    ) -> np.ndarray:
        """
        Compute pairwise distances between two sets of embeddings.

        Args:
            embeddings1: First set of embeddings (N x D)
            embeddings2: Second set of embeddings (M x D)
            metric: Distance metric ('euclidean', 'cosine')

        Returns:
            Distance matrix (N x M)
        """
        if metric == "euclidean":
            # Euclidean distance
            distances = np.linalg.norm(
                embeddings1[:, np.newaxis, :] - embeddings2[np.newaxis, :, :], axis=2
            )
        elif metric == "cosine":
            # Cosine distance
            embeddings1_norm = embeddings1 / (
                np.linalg.norm(embeddings1, axis=1, keepdims=True) + 1e-8
            )
            embeddings2_norm = embeddings2 / (
                np.linalg.norm(embeddings2, axis=1, keepdims=True) + 1e-8
            )
            distances = 1 - np.dot(embeddings1_norm, embeddings2_norm.T)
        else:
            raise ValueError(f"Unknown metric: {metric}")

        return distances
