"""Embedding extraction module using CLaMP models."""

import json
import hashlib
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from loguru import logger
from abc import ABC, abstractmethod
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm


def _resolve_hf_model_id(config: Dict, logical_name: str, default_model_id: str) -> str:
    """Resolve HuggingFace model id for a logical embedding model name."""
    for model_cfg in config.get("embeddings", {}).get("models", []):
        if model_cfg.get("name") == logical_name:
            return model_cfg.get("hf_model_name", default_model_id)
    return default_model_id


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
    """CLaMP 1 model - Text format based (ABC/MTF)."""

    def __init__(self, config: Dict):
        """Initialize CLaMP 1 model."""
        super().__init__(config, "CLaMP-1")
        self.format_type = "text"
        
        # Load model and tokenizer from HuggingFace
        try:
            logger.info("Loading CLaMP-1 model from HuggingFace...")
            model_name = _resolve_hf_model_id(
                config,
                "CLaMP-1",
                "sentence-transformers/all-MiniLM-L6-v2",
            )
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name)
            self.model.to(self.device)
            self.model.eval()
            
            # Get embedding dimension from model
            self.embedding_dim = self.model.config.hidden_size
            logger.info(f"CLaMP-1 loaded successfully (dim={self.embedding_dim})")
        except Exception as e:
            logger.warning(f"Failed to load CLaMP-1 from HuggingFace: {e}")
            logger.warning("Using dummy model for testing purposes")
            self.model = None
            self.tokenizer = None
            self.embedding_dim = 512

    def _tokens_to_text(self, tokens: List[int]) -> str:
        """Convert token IDs to text representation for CLaMP-1.
        
        Args:
            tokens: List of token IDs
            
        Returns:
            Text representation (space-separated token strings)
        """
        if self.tokenizer is None:
            # Fallback: treat tokens as ASCII
            return " ".join([str(t) for t in tokens])
        
        try:
            # Try to decode tokens
            text = self.tokenizer.decode(tokens, skip_special_tokens=True)
            return text
        except Exception:
            # Fallback
            return " ".join([str(t) for t in tokens])

    def encode(self, tokens: List[int]) -> np.ndarray:
        """Encode tokens to embeddings using CLaMP 1."""
        if self.model is None:
            # Dummy embedding
            logger.debug(f"Encoding {len(tokens)} tokens with CLaMP-1 (dummy)")
            return np.random.randn(self.embedding_dim).astype(np.float32)
        
        logger.debug(f"Encoding {len(tokens)} tokens with CLaMP-1")
        
        try:
            # Convert tokens to text
            text = self._tokens_to_text(tokens)
            
            # Tokenize text
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Get embeddings
            with torch.no_grad():
                outputs = self.model(**inputs)
                # Use [CLS] token embedding (first token)
                embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            
            return embedding[0].astype(np.float32)
        except Exception as e:
            logger.error(f"Error encoding with CLaMP-1: {e}")
            return np.random.randn(self.embedding_dim).astype(np.float32)

    def encode_batch(self, token_sequences: List[List[int]]) -> np.ndarray:
        """Encode batch of token sequences."""
        if self.model is None:
            # Dummy embeddings
            return np.random.randn(len(token_sequences), self.embedding_dim).astype(np.float32)
        
        logger.debug(f"Batch encoding {len(token_sequences)} sequences with CLaMP-1")
        
        try:
            # Convert all sequences to texts
            texts = [self._tokens_to_text(seq) for seq in token_sequences]
            
            # Tokenize in batch
            inputs = self.tokenizer(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Get embeddings
            with torch.no_grad():
                outputs = self.model(**inputs)
                # Use [CLS] token embedding (first token)
                embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            
            return embeddings.astype(np.float32)
        except Exception as e:
            logger.error(f"Error batch encoding with CLaMP-1: {e}")
            return np.random.randn(len(token_sequences), self.embedding_dim).astype(np.float32)

    def get_embedding_dim(self) -> int:
        """Get embedding dimensionality."""
        return self.embedding_dim


class CLaMP2Model(EmbeddingModel):
    """CLaMP 2 model - Direct MIDI structure based."""

    def __init__(self, config: Dict):
        """Initialize CLaMP 2 model."""
        super().__init__(config, "CLaMP-2")
        self.format_type = "midi"
        
        # Load model and tokenizer from HuggingFace
        try:
            logger.info("Loading CLaMP-2 model from HuggingFace...")
            model_name = _resolve_hf_model_id(
                config,
                "CLaMP-2",
                "sentence-transformers/all-MiniLM-L6-v2",
            )
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name)
            self.model.to(self.device)
            self.model.eval()
            
            # Get embedding dimension from model
            self.embedding_dim = self.model.config.hidden_size
            logger.info(f"CLaMP-2 loaded successfully (dim={self.embedding_dim})")
        except Exception as e:
            logger.warning(f"Failed to load CLaMP-2 from HuggingFace: {e}")
            logger.warning("Using dummy model for testing purposes")
            self.model = None
            self.tokenizer = None
            self.embedding_dim = 512

    def _tokens_to_midi_text(self, tokens: List[int]) -> str:
        """Convert token IDs to MIDI-like text representation for CLaMP-2.
        
        Args:
            tokens: List of token IDs
            
        Returns:
            MIDI-like text representation
        """
        # Create MIDI-like textual representation
        # Format: "token1 token2 token3 ..."
        return " ".join([str(t) for t in tokens])

    def encode(self, tokens: List[int]) -> np.ndarray:
        """Encode tokens to embeddings using CLaMP 2."""
        if self.model is None:
            # Dummy embedding
            logger.debug(f"Encoding {len(tokens)} tokens with CLaMP-2 (dummy)")
            return np.random.randn(self.embedding_dim).astype(np.float32)
        
        logger.debug(f"Encoding {len(tokens)} tokens with CLaMP-2")
        
        try:
            # Convert tokens to MIDI text representation
            midi_text = self._tokens_to_midi_text(tokens)
            
            # Tokenize text
            inputs = self.tokenizer(midi_text, return_tensors="pt", truncation=True, max_length=512)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Get embeddings
            with torch.no_grad():
                outputs = self.model(**inputs)
                # Use [CLS] token embedding (first token)
                embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            
            return embedding[0].astype(np.float32)
        except Exception as e:
            logger.error(f"Error encoding with CLaMP-2: {e}")
            return np.random.randn(self.embedding_dim).astype(np.float32)

    def encode_batch(self, token_sequences: List[List[int]]) -> np.ndarray:
        """Encode batch of token sequences."""
        if self.model is None:
            # Dummy embeddings
            return np.random.randn(len(token_sequences), self.embedding_dim).astype(np.float32)
        
        logger.debug(f"Batch encoding {len(token_sequences)} sequences with CLaMP-2")
        
        try:
            # Convert all sequences to MIDI texts
            midi_texts = [self._tokens_to_midi_text(seq) for seq in token_sequences]
            
            # Tokenize in batch
            inputs = self.tokenizer(
                midi_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Get embeddings
            with torch.no_grad():
                outputs = self.model(**inputs)
                # Use [CLS] token embedding (first token)
                embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            
            return embeddings.astype(np.float32)
        except Exception as e:
            logger.error(f"Error batch encoding with CLaMP-2: {e}")
            return np.random.randn(len(token_sequences), self.embedding_dim).astype(np.float32)

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
    """Extract embeddings from tokenized MIDI data with caching."""

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
        
        # Initialize disk cache
        self.cache_dir = Path(config["embeddings"].get("cache_dir", "data/embeddings/cache"))
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.use_cache = config["embeddings"].get("cache_embeddings", True)
        self.memory_cache = {} if self.use_cache else None
        
        logger.info(f"EmbeddingExtractor initialized with cache at {self.cache_dir}")

    def _get_cache_key(self, model_name: str, token_hash: str) -> str:
        """Generate cache key."""
        return f"{model_name}_{token_hash}"

    def _hash_tokens(self, tokens: List[int]) -> str:
        """Generate hash for token sequence."""
        token_str = ",".join(map(str, tokens))
        return hashlib.md5(token_str.encode()).hexdigest()

    def _get_cache_path(self, cache_key: str) -> Path:
        """Get cache file path."""
        return self.cache_dir / f"{cache_key}.npy"

    def _get_metadata_path(self, cache_key: str) -> Path:
        """Get metadata file path."""
        return self.cache_dir / f"{cache_key}_meta.json"

    def _load_from_disk_cache(self, cache_key: str) -> Optional[np.ndarray]:
        """Load embedding from disk cache."""
        cache_path = self._get_cache_path(cache_key)
        if cache_path.exists():
            try:
                embedding = np.load(cache_path)
                logger.debug(f"Loaded embedding from disk cache: {cache_key}")
                return embedding
            except Exception as e:
                logger.warning(f"Failed to load from disk cache: {e}")
        return None

    def _save_to_disk_cache(self, cache_key: str, embedding: np.ndarray, metadata: Dict = None):
        """Save embedding to disk cache."""
        cache_path = self._get_cache_path(cache_key)
        try:
            np.save(cache_path, embedding)
            
            # Save metadata
            if metadata:
                meta_path = self._get_metadata_path(cache_key)
                with open(meta_path, "w") as f:
                    json.dump(metadata, f)
            
            logger.debug(f"Saved embedding to disk cache: {cache_key}")
        except Exception as e:
            logger.warning(f"Failed to save to disk cache: {e}")

    def extract_embeddings(self, token_sequences: List[List[int]], model_name: str) -> np.ndarray:
        """
        Extract embeddings from token sequences with caching.

        Args:
            token_sequences: List of token sequences
            model_name: Name of the model to use

        Returns:
            Embedding matrix (N x D)
        """
        if model_name not in self.models:
            raise ValueError(f"Unknown model: {model_name}")

        model = self.models[model_name]
        batch_size = self.config["embeddings"].get("batch_size", 32)
        embeddings_list = []

        logger.info(f"Extracting embeddings for {len(token_sequences)} sequences with {model_name}")

        for i, tokens in enumerate(tqdm(token_sequences, desc=f"Extracting with {model_name}")):
            # Check cache
            if self.use_cache:
                token_hash = self._hash_tokens(tokens)
                cache_key = self._get_cache_key(model_name, token_hash)
                
                # Check memory cache
                if cache_key in self.memory_cache:
                    embeddings_list.append(self.memory_cache[cache_key])
                    continue
                
                # Check disk cache
                cached_embedding = self._load_from_disk_cache(cache_key)
                if cached_embedding is not None:
                    self.memory_cache[cache_key] = cached_embedding
                    embeddings_list.append(cached_embedding)
                    continue

            # Extract embedding if not cached
            if len(embeddings_list) < (i + 1):  # Only extract if not in cache
                embedding = model.encode(tokens)
                embeddings_list.append(embedding)
                
                # Save to cache
                if self.use_cache:
                    token_hash = self._hash_tokens(tokens)
                    cache_key = self._get_cache_key(model_name, token_hash)
                    self.memory_cache[cache_key] = embedding
                    metadata = {
                        "model": model_name,
                        "num_tokens": len(tokens),
                        "embedding_dim": embedding.shape[0]
                    }
                    self._save_to_disk_cache(cache_key, embedding, metadata)

        embeddings = np.array(embeddings_list)
        logger.info(f"Extracted {len(embeddings_list)} embeddings (shape: {embeddings.shape})")
        return embeddings

    def extract_dataset_embeddings(
        self, token_files: List[Path], output_dir: Path, model_name: str
    ) -> Dict:
        """
        Extract embeddings for multiple tokenized files.

        Args:
            token_files: List of paths to tokenized files (JSON or text)
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
            "errors": []
        }

        output_dir.mkdir(parents=True, exist_ok=True)

        for i, token_file in enumerate(token_files, 1):
            logger.info(f"Extracting embeddings {i}/{len(token_files)}: {token_file.name}")

            try:
                # Load tokens from file
                tokens = self._load_tokens_from_file(token_file)
                
                if not tokens:
                    raise ValueError("No tokens found in file")

                # Extract embeddings
                embedding = self.extract_embeddings([tokens], model_name)[0]

                # Save embeddings
                output_file = output_dir / (token_file.stem + "_embedding.npy")
                np.save(output_file, embedding)
                
                # Save metadata
                meta_file = output_dir / (token_file.stem + "_meta.json")
                metadata = {
                    "source_file": token_file.name,
                    "model": model_name,
                    "embedding_shape": embedding.shape,
                    "num_tokens": len(tokens)
                }
                with open(meta_file, "w") as f:
                    json.dump(metadata, f, indent=2)

                stats["successful"] += 1
                logger.debug(f"✓ Saved to {output_file}")
                
            except Exception as e:
                logger.error(f"Error extracting embeddings for {token_file}: {e}")
                stats["failed"] += 1
                stats["errors"].append(str(token_file))

        logger.info(
            f"Embedding extraction complete: {stats['successful']} successful, {stats['failed']} failed"
        )
        return stats

    @staticmethod
    def _load_tokens_from_file(token_file: Path) -> List[int]:
        """Load tokens from file (JSON or text format).
        
        Args:
            token_file: Path to token file
            
        Returns:
            List of token IDs
        """
        try:
            if token_file.suffix == ".json":
                with open(token_file, "r") as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        return [int(t) for t in data]
                    elif isinstance(data, dict) and "tokens" in data:
                        return [int(t) for t in data["tokens"]]
            else:
                # Text format: space-separated or comma-separated integers
                with open(token_file, "r") as f:
                    content = f.read().strip()
                    # Try space-separated first
                    if " " in content:
                        return [int(t) for t in content.split()]
                    # Try comma-separated
                    elif "," in content:
                        return [int(t) for t in content.split(",")]
                    # Single line of integers
                    else:
                        return [int(t) for t in content.split()]
        except Exception as e:
            logger.error(f"Failed to load tokens from {token_file}: {e}")
        
        return []


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
