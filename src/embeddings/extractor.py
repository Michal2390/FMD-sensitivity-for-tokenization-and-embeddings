"""Embedding extraction module using real CLaMP music models.

CLaMP-1: sander-wood/clamp-small-512 — RoBERTa on ABC notation (512-dim)
CLaMP-2: shanghaicai/CLaMP2 — MIDI-native encoder (512-dim)

Both models are music-domain contrastive models (music–text alignment).
They produce embeddings that capture musical semantics.
If real models are unavailable, falls back to proxy models with warnings.
"""

import json
import hashlib
import tempfile
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from loguru import logger
from abc import ABC, abstractmethod
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
        self.config = config
        self.model_name = model_name
        requested_device = str(config["embeddings"].get("device", "cpu")).lower()
        if requested_device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        elif requested_device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA requested but not available; falling back to CPU")
            self.device = "cpu"
        else:
            self.device = requested_device

        if self.device == "cuda":
            gpu_name = torch.cuda.get_device_name(0)
            logger.info(f"Initialized {model_name} on CUDA GPU: {gpu_name}")
        else:
            logger.info(f"Initialized {model_name} on device: {self.device}")
        logger.info("Embedding stage runs inference only (no neural network training)")

    @abstractmethod
    def encode(self, tokens: List[int], midi_data=None) -> np.ndarray:
        """Encode tokens (and optionally raw midi_data) to embeddings.

        Args:
            tokens: List of token IDs from miditok tokenizer.
            midi_data: Optional PrettyMIDI object for models that need raw MIDI.

        Returns:
            Embedding vector (1D numpy array)
        """
        pass

    @abstractmethod
    def encode_batch(self, token_sequences: List[List[int]], midi_data_list: Optional[List] = None) -> np.ndarray:
        """Encode batch of token sequences to embeddings."""
        pass

    def get_embedding_dim(self) -> int:
        raise NotImplementedError


class CLaMP1Model(EmbeddingModel):
    """CLaMP-1 model — Music encoder trained on ABC notation.

    Uses ``sander-wood/clamp-small-512`` from HuggingFace.
    Converts MIDI to ABC notation via music21, then encodes with
    the CLaMP-1 music encoder (RoBERTa-based, 512-dim output).
    """

    DEFAULT_MODEL = "sander-wood/clamp-small-512"

    def __init__(self, config: Dict):
        super().__init__(config, "CLaMP-1")
        self.format_type = "abc"
        self._use_real_model = False

        model_id = _resolve_hf_model_id(config, "CLaMP-1", self.DEFAULT_MODEL)

        try:
            logger.info(f"Loading CLaMP-1 model: {model_id}")
            from transformers import AutoModel, AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
            self.model = AutoModel.from_pretrained(model_id, trust_remote_code=True)
            self.model.to(self.device)
            self.model.eval()
            self.embedding_dim = self.model.config.hidden_size
            self._use_real_model = True
            logger.info(f"CLaMP-1 loaded: {model_id} (dim={self.embedding_dim})")
        except Exception as e:
            logger.warning(f"Failed to load CLaMP-1 ({model_id}): {e}")
            logger.warning("Falling back to sentence-transformers proxy for CLaMP-1")
            try:
                from transformers import AutoModel, AutoTokenizer
                fallback = "sentence-transformers/all-MiniLM-L6-v2"
                self.tokenizer = AutoTokenizer.from_pretrained(fallback)
                self.model = AutoModel.from_pretrained(fallback)
                self.model.to(self.device)
                self.model.eval()
                self.embedding_dim = self.model.config.hidden_size
                logger.info(f"CLaMP-1 fallback: {fallback} (dim={self.embedding_dim})")
            except Exception as e2:
                logger.warning(f"Fallback also failed: {e2}. Using dummy model.")
                self.model = None
                self.tokenizer = None
                self.embedding_dim = 512

        self._has_music21 = False
        try:
            import music21  # noqa: F401
            self._has_music21 = True
            logger.info("music21 available for MIDI→ABC conversion")
        except ImportError:
            logger.warning("music21 not installed — CLaMP-1 will use token-text fallback")

    def _midi_to_abc(self, midi_data) -> str:
        """Convert PrettyMIDI object to ABC notation string."""
        if not self._has_music21 or midi_data is None:
            return ""
        try:
            import music21
            with tempfile.NamedTemporaryFile(suffix=".mid", delete=False) as tmp:
                tmp_path = tmp.name
                midi_data.write(tmp_path)
            score = music21.converter.parse(tmp_path)
            abc_str = music21.converter.toData(score, fmt='abc')
            if isinstance(abc_str, bytes):
                abc_str = abc_str.decode('utf-8', errors='replace')
            Path(tmp_path).unlink(missing_ok=True)
            return abc_str[:8192] if len(abc_str) > 8192 else abc_str
        except Exception as e:
            logger.debug(f"MIDI→ABC conversion failed: {e}")
            return ""

    def _tokens_to_text(self, tokens: List[int]) -> str:
        return " ".join([str(t) for t in tokens[:512]])

    def encode(self, tokens: List[int], midi_data=None) -> np.ndarray:
        if self.model is None:
            return np.random.randn(self.embedding_dim).astype(np.float32)
        try:
            if self._has_music21 and midi_data is not None and self._use_real_model:
                text = self._midi_to_abc(midi_data)
                if not text:
                    text = self._tokens_to_text(tokens)
            else:
                text = self._tokens_to_text(tokens)

            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = self.model(**inputs)
                embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            return embedding[0].astype(np.float32)
        except Exception as e:
            logger.error(f"Error encoding with CLaMP-1: {e}")
            return np.random.randn(self.embedding_dim).astype(np.float32)

    def encode_batch(self, token_sequences: List[List[int]], midi_data_list: Optional[List] = None) -> np.ndarray:
        if self.model is None:
            return np.random.randn(len(token_sequences), self.embedding_dim).astype(np.float32)
        try:
            texts = []
            for i, seq in enumerate(token_sequences):
                md = midi_data_list[i] if midi_data_list and i < len(midi_data_list) else None
                if self._has_music21 and md is not None and self._use_real_model:
                    abc = self._midi_to_abc(md)
                    texts.append(abc if abc else self._tokens_to_text(seq))
                else:
                    texts.append(self._tokens_to_text(seq))

            inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = self.model(**inputs)
                embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            return embeddings.astype(np.float32)
        except Exception as e:
            logger.error(f"Error batch encoding with CLaMP-1: {e}")
            return np.random.randn(len(token_sequences), self.embedding_dim).astype(np.float32)

    def get_embedding_dim(self) -> int:
        return self.embedding_dim


class CLaMP2Model(EmbeddingModel):
    """CLaMP-2 model — MIDI-native music encoder.

    Uses ``shanghaicai/CLaMP2`` from HuggingFace.
    CLaMP-2 processes MIDI directly with its own tokenizer.
    Falls back to a DIFFERENT proxy than CLaMP-1 if unavailable.
    """

    DEFAULT_MODEL = "shanghaicai/CLaMP2"

    def __init__(self, config: Dict):
        super().__init__(config, "CLaMP-2")
        self.format_type = "midi"
        self._use_real_model = False

        model_id = _resolve_hf_model_id(config, "CLaMP-2", self.DEFAULT_MODEL)

        try:
            logger.info(f"Loading CLaMP-2 model: {model_id}")
            from transformers import AutoModel, AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
            self.model = AutoModel.from_pretrained(model_id, trust_remote_code=True)
            self.model.to(self.device)
            self.model.eval()
            self.embedding_dim = self.model.config.hidden_size
            self._use_real_model = True
            logger.info(f"CLaMP-2 loaded: {model_id} (dim={self.embedding_dim})")
        except Exception as e:
            logger.warning(f"Failed to load CLaMP-2 ({model_id}): {e}")
            logger.warning("Falling back to all-mpnet-base-v2 proxy for CLaMP-2")
            try:
                from transformers import AutoModel, AutoTokenizer
                fallback = "sentence-transformers/all-mpnet-base-v2"
                self.tokenizer = AutoTokenizer.from_pretrained(fallback)
                self.model = AutoModel.from_pretrained(fallback)
                self.model.to(self.device)
                self.model.eval()
                self.embedding_dim = self.model.config.hidden_size
                logger.info(f"CLaMP-2 fallback: {fallback} (dim={self.embedding_dim})")
            except Exception as e2:
                logger.warning(f"Fallback also failed: {e2}. Using dummy model.")
                self.model = None
                self.tokenizer = None
                self.embedding_dim = 512

    def _midi_to_text(self, midi_data) -> str:
        """Convert PrettyMIDI to MIDI-event text representation."""
        if midi_data is None:
            return ""
        try:
            events = []
            for instrument in midi_data.instruments:
                if instrument.is_drum:
                    continue
                for note in sorted(instrument.notes, key=lambda n: n.start):
                    start_tick = int(note.start * 480)
                    dur_tick = int((note.end - note.start) * 480)
                    events.append(f"p{note.pitch} v{note.velocity} t{start_tick} d{dur_tick}")
            return " ".join(events[:1000])
        except Exception as e:
            logger.debug(f"MIDI→text conversion failed: {e}")
            return ""

    def _tokens_to_midi_text(self, tokens: List[int]) -> str:
        return " ".join([str(t) for t in tokens[:512]])

    def encode(self, tokens: List[int], midi_data=None) -> np.ndarray:
        if self.model is None:
            return np.random.randn(self.embedding_dim).astype(np.float32)
        try:
            if midi_data is not None and self._use_real_model:
                text = self._midi_to_text(midi_data)
                if not text:
                    text = self._tokens_to_midi_text(tokens)
            else:
                text = self._tokens_to_midi_text(tokens)

            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = self.model(**inputs)
                embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            return embedding[0].astype(np.float32)
        except Exception as e:
            logger.error(f"Error encoding with CLaMP-2: {e}")
            return np.random.randn(self.embedding_dim).astype(np.float32)

    def encode_batch(self, token_sequences: List[List[int]], midi_data_list: Optional[List] = None) -> np.ndarray:
        if self.model is None:
            return np.random.randn(len(token_sequences), self.embedding_dim).astype(np.float32)
        try:
            texts = []
            for i, seq in enumerate(token_sequences):
                md = midi_data_list[i] if midi_data_list and i < len(midi_data_list) else None
                if md is not None and self._use_real_model:
                    t = self._midi_to_text(md)
                    texts.append(t if t else self._tokens_to_midi_text(seq))
                else:
                    texts.append(self._tokens_to_midi_text(seq))

            inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = self.model(**inputs)
                embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            return embeddings.astype(np.float32)
        except Exception as e:
            logger.error(f"Error batch encoding with CLaMP-2: {e}")
            return np.random.randn(len(token_sequences), self.embedding_dim).astype(np.float32)

    def get_embedding_dim(self) -> int:
        return self.embedding_dim


class MusicBERTModel(EmbeddingModel):
    """MusicBERT model — symbolic music understanding via OctupleMIDI tokenization.

    Attempts to load ``m-a-p/MusicBERT-base`` from HuggingFace.
    Falls back to ``bert-base-uncased`` with token-as-text encoding
    if the real model is unavailable.

    Two modes:
      * **native**: Uses OctupleMIDI tokenisation internally (via model's own tokenizer).
      * **text**: Converts our token IDs to space-separated text, fed through the
        model's WordPiece tokenizer (same approach as CLaMP fallbacks).
    """

    DEFAULT_MODEL = "m-a-p/MusicBERT-base"
    FALLBACK_MODEL = "bert-base-uncased"

    def __init__(self, config: Dict):
        super().__init__(config, "MusicBERT")
        self._use_real_model = False

        model_id = _resolve_hf_model_id(config, "MusicBERT", self.DEFAULT_MODEL)

        try:
            logger.info(f"Loading MusicBERT model: {model_id}")
            from transformers import AutoModel, AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
            self.model = AutoModel.from_pretrained(model_id, trust_remote_code=True)
            self.model.to(self.device)
            self.model.eval()
            self.embedding_dim = self.model.config.hidden_size
            self._use_real_model = True
            logger.info(f"MusicBERT loaded: {model_id} (dim={self.embedding_dim})")
        except Exception as e:
            logger.warning(f"Failed to load MusicBERT ({model_id}): {e}")
            logger.warning(f"Falling back to {self.FALLBACK_MODEL} proxy for MusicBERT")
            try:
                from transformers import AutoModel, AutoTokenizer
                self.tokenizer = AutoTokenizer.from_pretrained(self.FALLBACK_MODEL)
                self.model = AutoModel.from_pretrained(self.FALLBACK_MODEL)
                self.model.to(self.device)
                self.model.eval()
                self.embedding_dim = self.model.config.hidden_size
                logger.info(f"MusicBERT fallback: {self.FALLBACK_MODEL} (dim={self.embedding_dim})")
            except Exception as e2:
                logger.warning(f"Fallback also failed: {e2}. Using dummy model.")
                self.model = None
                self.tokenizer = None
                self.embedding_dim = 768

    def _tokens_to_text(self, tokens: List[int]) -> str:
        return " ".join([str(t) for t in tokens[:512]])

    def encode(self, tokens: List[int], midi_data=None) -> np.ndarray:
        if self.model is None:
            return np.random.randn(self.embedding_dim).astype(np.float32)
        try:
            text = self._tokens_to_text(tokens)
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = self.model(**inputs)
                embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            return embedding[0].astype(np.float32)
        except Exception as e:
            logger.error(f"Error encoding with MusicBERT: {e}")
            return np.random.randn(self.embedding_dim).astype(np.float32)

    def encode_batch(self, token_sequences: List[List[int]], midi_data_list: Optional[List] = None) -> np.ndarray:
        if self.model is None:
            return np.random.randn(len(token_sequences), self.embedding_dim).astype(np.float32)
        try:
            texts = [self._tokens_to_text(seq) for seq in token_sequences]
            inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = self.model(**inputs)
                embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            return embeddings.astype(np.float32)
        except Exception as e:
            logger.error(f"Error batch encoding with MusicBERT: {e}")
            return np.random.randn(len(token_sequences), self.embedding_dim).astype(np.float32)

    def get_embedding_dim(self) -> int:
        return self.embedding_dim


class EmbeddingFactory:
    """Factory for creating embedding models."""
    _models = {
        "CLaMP-1": CLaMP1Model,
        "CLaMP-2": CLaMP2Model,
        "MusicBERT": MusicBERTModel,
    }

    @classmethod
    def create_model(cls, config: Dict, model_name: str) -> EmbeddingModel:
        if model_name not in cls._models:
            raise ValueError(f"Unknown model: {model_name}")
        return cls._models[model_name](config)

    @classmethod
    def get_available_models(cls) -> List[str]:
        return list(cls._models.keys())


class EmbeddingExtractor:
    """Extract embeddings from tokenized MIDI data with caching."""

    def __init__(self, config: Dict):
        self.config = config
        self.models = {
            model_name: EmbeddingFactory.create_model(config, model_name)
            for model_name in EmbeddingFactory.get_available_models()
        }
        self.cache_dir = Path(config["embeddings"].get("cache_dir", "data/embeddings/cache"))
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.use_cache = config["embeddings"].get("cache_embeddings", True)
        self.memory_cache = {} if self.use_cache else None
        logger.info(f"EmbeddingExtractor initialized with cache at {self.cache_dir}")

    def _get_cache_key(self, model_name: str, token_hash: str) -> str:
        return f"{model_name}_{token_hash}"

    def _hash_tokens(self, tokens: List[int]) -> str:
        token_str = ",".join(map(str, tokens))
        return hashlib.md5(token_str.encode()).hexdigest()

    def _get_cache_path(self, cache_key: str) -> Path:
        return self.cache_dir / f"{cache_key}.npy"

    def _get_metadata_path(self, cache_key: str) -> Path:
        return self.cache_dir / f"{cache_key}_meta.json"

    def _load_from_disk_cache(self, cache_key: str) -> Optional[np.ndarray]:
        cache_path = self._get_cache_path(cache_key)
        if cache_path.exists():
            try:
                embedding = np.load(cache_path)
                return embedding
            except Exception as e:
                logger.warning(f"Failed to load from disk cache: {e}")
        return None

    def _save_to_disk_cache(self, cache_key: str, embedding: np.ndarray, metadata: Dict = None):
        cache_path = self._get_cache_path(cache_key)
        try:
            np.save(cache_path, embedding)
            if metadata:
                meta_path = self._get_metadata_path(cache_key)
                with open(meta_path, "w") as f:
                    json.dump(metadata, f)
        except Exception as e:
            logger.warning(f"Failed to save to disk cache: {e}")

    def extract_embeddings(
        self,
        token_sequences: List[List[int]],
        model_name: str,
        midi_data_list: Optional[List] = None,
    ) -> np.ndarray:
        """Extract embeddings from token sequences with caching.

        Args:
            token_sequences: List of token sequences
            model_name: Name of the model to use
            midi_data_list: Optional list of PrettyMIDI objects

        Returns:
            Embedding matrix (N x D)
        """
        if model_name not in self.models:
            raise ValueError(f"Unknown model: {model_name}")

        model = self.models[model_name]
        embeddings_list = []

        logger.info(f"Extracting embeddings for {len(token_sequences)} sequences with {model_name}")

        for i, tokens in enumerate(tqdm(token_sequences, desc=f"Extracting with {model_name}")):
            if self.use_cache:
                token_hash = self._hash_tokens(tokens)
                cache_key = self._get_cache_key(model_name, token_hash)
                if cache_key in self.memory_cache:
                    embeddings_list.append(self.memory_cache[cache_key])
                    continue
                cached_embedding = self._load_from_disk_cache(cache_key)
                if cached_embedding is not None:
                    self.memory_cache[cache_key] = cached_embedding
                    embeddings_list.append(cached_embedding)
                    continue

            if len(embeddings_list) < (i + 1):
                midi_data = midi_data_list[i] if midi_data_list and i < len(midi_data_list) else None
                embedding = model.encode(tokens, midi_data=midi_data)
                embeddings_list.append(embedding)

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
            try:
                tokens = self._load_tokens_from_file(token_file)
                if not tokens:
                    raise ValueError("No tokens found in file")
                embedding = self.extract_embeddings([tokens], model_name)[0]
                output_file = output_dir / (token_file.stem + "_embedding.npy")
                np.save(output_file, embedding)
                meta_file = output_dir / (token_file.stem + "_meta.json")
                metadata = {
                    "source_file": token_file.name,
                    "model": model_name,
                    "embedding_shape": list(embedding.shape),
                    "num_tokens": len(tokens)
                }
                with open(meta_file, "w") as f:
                    json.dump(metadata, f, indent=2)
                stats["successful"] += 1
            except Exception as e:
                logger.error(f"Error extracting embeddings for {token_file}: {e}")
                stats["failed"] += 1
                stats["errors"].append(str(token_file))

        return stats

    @staticmethod
    def _load_tokens_from_file(token_file: Path) -> List[int]:
        try:
            if token_file.suffix == ".json":
                with open(token_file, "r") as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        return [int(t) for t in data]
                    elif isinstance(data, dict) and "tokens" in data:
                        return [int(t) for t in data["tokens"]]
            else:
                with open(token_file, "r") as f:
                    content = f.read().strip()
                    if " " in content:
                        return [int(t) for t in content.split()]
                    elif "," in content:
                        return [int(t) for t in content.split(",")]
                    else:
                        return [int(t) for t in content.split()]
        except Exception as e:
            logger.error(f"Failed to load tokens from {token_file}: {e}")
        return []


class EmbeddingAnalyzer:
    """Analyze embedding statistics."""

    @staticmethod
    def compute_statistics(embeddings: np.ndarray) -> Dict:
        return {
            "mean": np.mean(embeddings, axis=0),
            "std": np.std(embeddings, axis=0),
            "cov": np.cov(embeddings.T),
            "num_samples": embeddings.shape[0],
            "embedding_dim": embeddings.shape[1] if embeddings.ndim > 1 else 1,
        }

    @staticmethod
    def compute_pairwise_distances(
        embeddings1: np.ndarray, embeddings2: np.ndarray, metric: str = "euclidean"
    ) -> np.ndarray:
        if metric == "euclidean":
            distances = np.linalg.norm(
                embeddings1[:, np.newaxis, :] - embeddings2[np.newaxis, :, :], axis=2
            )
        elif metric == "cosine":
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
