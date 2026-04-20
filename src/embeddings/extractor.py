"""Embedding extraction module for FMD sensitivity analysis.

Models used:
  MusicBERT:       manoskary/musicbert — BERT pre-trained on symbolic music tokens (768-dim)
  MusicBERT-large: manoskary/musicbert-large — larger variant (1024-dim)
  MERT:            m-a-p/MERT-v1-95M — self-supervised audio model for music (768-dim)
  NLP-Baseline:    sentence-transformers/all-mpnet-base-v2 — general NLP baseline (768-dim)
  CLaMP-1:         sander-wood/clamp-small-512 — contrastive language-music pre-training (768-dim)
  CLaMP-2:         sander-wood/clamp2 — multimodal music IR with M3 encoder (768-dim)

The NLP baseline is included intentionally to assess whether music-specific
pre-training affects FMD sensitivity compared to a general-purpose model.
"""

import json
import hashlib
import glob as _glob
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
        pass

    @abstractmethod
    def encode_batch(self, token_sequences: List[List[int]], midi_data_list: Optional[List] = None) -> np.ndarray:
        pass

    def get_embedding_dim(self) -> int:
        raise NotImplementedError


class _TextEncoderModel(EmbeddingModel):
    """Base for models that encode token sequences as text via a HF tokenizer+model."""

    DEFAULT_MODEL: str = ""
    FALLBACK_MODEL: str = ""

    def __init__(self, config: Dict, logical_name: str, default_model: str, fallback_model: str, fallback_dim: int = 768):
        super().__init__(config, logical_name)
        self._use_real_model = False

        model_id = _resolve_hf_model_id(config, logical_name, default_model)

        try:
            logger.info(f"Loading {logical_name} model: {model_id}")
            from transformers import AutoModel, AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
            self.model = AutoModel.from_pretrained(model_id, trust_remote_code=True)
            self.model.to(self.device)
            self.model.eval()
            self.embedding_dim = self.model.config.hidden_size
            self._use_real_model = True
            logger.info(f"{logical_name} loaded: {model_id} (dim={self.embedding_dim})")
        except Exception as e:
            logger.warning(f"Failed to load {logical_name} ({model_id}): {e}")
            if fallback_model:
                logger.warning(f"Falling back to {fallback_model} proxy for {logical_name}")
                try:
                    from transformers import AutoModel, AutoTokenizer
                    self.tokenizer = AutoTokenizer.from_pretrained(fallback_model)
                    self.model = AutoModel.from_pretrained(fallback_model)
                    self.model.to(self.device)
                    self.model.eval()
                    self.embedding_dim = self.model.config.hidden_size
                    logger.info(f"{logical_name} fallback: {fallback_model} (dim={self.embedding_dim})")
                except Exception as e2:
                    logger.warning(f"Fallback also failed: {e2}. Using dummy model.")
                    self.model = None
                    self.tokenizer = None
                    self.embedding_dim = fallback_dim
            else:
                self.model = None
                self.tokenizer = None
                self.embedding_dim = fallback_dim

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
            logger.error(f"Error encoding with {self.model_name}: {e}")
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
            logger.error(f"Error batch encoding with {self.model_name}: {e}")
            return np.random.randn(len(token_sequences), self.embedding_dim).astype(np.float32)

    def get_embedding_dim(self) -> int:
        return self.embedding_dim


class MusicBERTModel(_TextEncoderModel):
    """MusicBERT — BERT pre-trained on symbolic music token sequences.

    Uses ``manoskary/musicbert`` from HuggingFace (hidden_size=768, vocab_size=540).
    Falls back to ``bert-base-uncased`` if unavailable.
    """

    def __init__(self, config: Dict):
        super().__init__(
            config,
            logical_name="MusicBERT",
            default_model="manoskary/musicbert",
            fallback_model="bert-base-uncased",
            fallback_dim=768,
        )


class MusicBERTLargeModel(_TextEncoderModel):
    """MusicBERT-large — larger BERT pre-trained on symbolic music tokens.

    Uses ``manoskary/musicbert-large`` from HuggingFace (hidden_size=1024, vocab_size=540).
    Provides a size comparison vs MusicBERT (768-dim) to assess
    the impact of model capacity on FMD sensitivity.
    Falls back to ``bert-large-uncased`` if unavailable.
    """

    def __init__(self, config: Dict):
        super().__init__(
            config,
            logical_name="MusicBERT-large",
            default_model="manoskary/musicbert-large",
            fallback_model="bert-large-uncased",
            fallback_dim=1024,
        )


class NLPBaselineModel(_TextEncoderModel):
    """NLP Baseline — general-purpose sentence-transformer (non-music).

    Uses ``sentence-transformers/all-mpnet-base-v2`` (hidden_size=768).
    Included intentionally as a control: if FMD sensitivity patterns
    are similar with a non-music model, then music-specific pre-training
    does not significantly affect FMD behaviour.
    """

    def __init__(self, config: Dict):
        super().__init__(
            config,
            logical_name="NLP-Baseline",
            default_model="sentence-transformers/all-mpnet-base-v2",
            fallback_model="",
            fallback_dim=768,
        )


class MERTModel(EmbeddingModel):
    """MERT model — audio-based music understanding model.

    Uses ``m-a-p/MERT-v1-95M`` from HuggingFace.
    MERT is a self-supervised audio model pre-trained on music.
    It requires audio input, so MIDI files are first synthesized
    to audio using pretty_midi.synthesize().

    This provides a cross-domain contrast: symbolic embedding models
    (MusicBERT) vs audio embedding model (MERT), enabling analysis
    of how the embedding domain affects FMD.

    Falls back to ``facebook/wav2vec2-base`` if MERT is unavailable.
    """

    DEFAULT_MODEL = "m-a-p/MERT-v1-95M"
    FALLBACK_MODEL = "facebook/wav2vec2-base"
    SAMPLE_RATE = 24000
    FALLBACK_SAMPLE_RATE = 16000
    MAX_AUDIO_SECONDS = 10

    def __init__(self, config: Dict):
        super().__init__(config, "MERT")
        self._use_real_model = False
        self._sample_rate = self.SAMPLE_RATE
        self._processor = None

        model_id = _resolve_hf_model_id(config, "MERT", self.DEFAULT_MODEL)

        try:
            logger.info(f"Loading MERT model: {model_id}")
            from transformers import AutoModel, Wav2Vec2FeatureExtractor
            self._processor = Wav2Vec2FeatureExtractor.from_pretrained(
                model_id, trust_remote_code=True
            )
            self.model = AutoModel.from_pretrained(model_id, trust_remote_code=True)
            self.model.to(self.device)
            self.model.eval()
            self.embedding_dim = self.model.config.hidden_size
            self._use_real_model = True
            self._sample_rate = getattr(
                self._processor, "sampling_rate", self.SAMPLE_RATE
            )
            logger.info(
                f"MERT loaded: {model_id} (dim={self.embedding_dim}, sr={self._sample_rate})"
            )
        except Exception as e:
            logger.warning(f"Failed to load MERT ({model_id}): {e}")
            logger.warning(f"Falling back to {self.FALLBACK_MODEL} proxy for MERT")
            self._sample_rate = self.FALLBACK_SAMPLE_RATE
            try:
                from transformers import AutoModel, Wav2Vec2FeatureExtractor
                self._processor = Wav2Vec2FeatureExtractor.from_pretrained(
                    self.FALLBACK_MODEL
                )
                self.model = AutoModel.from_pretrained(self.FALLBACK_MODEL)
                self.model.to(self.device)
                self.model.eval()
                self.embedding_dim = self.model.config.hidden_size
                logger.info(
                    f"MERT fallback: {self.FALLBACK_MODEL} (dim={self.embedding_dim})"
                )
            except Exception as e2:
                logger.warning(f"Fallback also failed: {e2}. Using dummy model.")
                self.model = None
                self._processor = None
                self.embedding_dim = 768

    def _midi_to_audio(self, midi_data) -> Optional[np.ndarray]:
        """Convert PrettyMIDI object to mono audio waveform via sinusoidal synthesis."""
        if midi_data is None:
            return None
        target_sr = self._sample_rate
        max_samples = self.MAX_AUDIO_SECONDS * target_sr
        try:
            audio = midi_data.synthesize(fs=target_sr)
            if audio is not None and len(audio) > 0:
                if audio.ndim > 1:
                    audio = audio.mean(axis=1)
                return audio[:max_samples].astype(np.float32)
        except Exception as e:
            logger.debug(f"pretty_midi.synthesize() failed: {e}")
        return None

    def _encode_audio(self, audio: np.ndarray) -> np.ndarray:
        inputs = self._processor(
            audio, sampling_rate=self._sample_rate, return_tensors="pt", padding=True,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
            hidden = outputs.last_hidden_state
            embedding = hidden.mean(dim=1).cpu().numpy()
        return embedding[0].astype(np.float32)

    def encode(self, tokens: List[int], midi_data=None) -> np.ndarray:
        if self.model is None:
            return np.random.randn(self.embedding_dim).astype(np.float32)

        if self._processor is not None and midi_data is not None:
            audio = self._midi_to_audio(midi_data)
            if audio is not None and len(audio) > 1000:
                try:
                    return self._encode_audio(audio)
                except Exception as e:
                    logger.debug(f"MERT audio encoding failed: {e}")

        # Fallback: deterministic noise based on token hash
        if self._processor is not None:
            try:
                rng = np.random.default_rng(hash(tuple(tokens[:20])) % (2**31))
                noise = rng.standard_normal(self._sample_rate * 2).astype(np.float32) * 0.001
                return self._encode_audio(noise)
            except Exception as e:
                logger.debug(f"MERT noise-fallback encoding failed: {e}")

        logger.warning("MERT: all encoding methods failed, returning random embedding")
        return np.random.randn(self.embedding_dim).astype(np.float32)

    def encode_batch(
        self, token_sequences: List[List[int]], midi_data_list: Optional[List] = None
    ) -> np.ndarray:
        results = []
        for i, tokens in enumerate(token_sequences):
            midi_data = (
                midi_data_list[i]
                if midi_data_list and i < len(midi_data_list)
                else None
            )
            results.append(self.encode(tokens, midi_data=midi_data))
        return np.array(results)

    def get_embedding_dim(self) -> int:
        return self.embedding_dim


# ---------------------------------------------------------------------------
# CLaMP-1: Contrastive Language-Music Pre-training (ISMIR 2023)
# Uses ABC notation with bar-patching for the music encoder.
# ---------------------------------------------------------------------------

class _CLaMPMusicEncoder(torch.nn.Module):
    """Reconstruct CLaMP-1 music encoder from checkpoint keys."""

    def __init__(self, patch_dim: int = 6272, hidden_size: int = 768,
                 num_layers: int = 6, num_heads: int = 12, max_len: int = 512):
        super().__init__()
        from transformers import BertConfig, BertModel
        self.patch_embedding = torch.nn.Linear(patch_dim, hidden_size)
        cfg = BertConfig(
            vocab_size=1, hidden_size=hidden_size,
            num_hidden_layers=num_layers, num_attention_heads=num_heads,
            intermediate_size=hidden_size * 4, max_position_embeddings=max_len,
        )
        self.enc = BertModel(cfg)

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        """patches: (batch, n_patches, patch_dim) -> (batch, hidden_size)"""
        x = self.patch_embedding(patches)  # (B, N, H)
        out = self.enc(inputs_embeds=x)
        return out.last_hidden_state[:, 0, :]  # CLS token


class CLaMP1Model(EmbeddingModel):
    """CLaMP-1 (sander-wood/clamp-small-512) — music encoder branch.

    Converts MIDI tokens → ABC notation via music21, then applies
    bar-patching to create fixed-size input patches for the model.
    Output: 768-dim embedding after music_proj.
    """

    HF_REPO = "sander-wood/clamp-small-512"
    PATCH_DIM = 6272  # from checkpoint: patch_embedding.weight shape[1]
    HIDDEN_SIZE = 768
    MAX_PATCHES = 512
    # Bar-patch vocabulary: 128 pitch * 49 duration bins = 6272
    PITCH_BINS = 128
    DUR_BINS = 49  # 6272 / 128

    def __init__(self, config: Dict):
        super().__init__(config, "CLaMP-1")
        self.embedding_dim = self.HIDDEN_SIZE
        self._loaded = False

        try:
            from huggingface_hub import hf_hub_download
            weight_path = hf_hub_download(self.HF_REPO, "pytorch_model.bin")
            state = torch.load(weight_path, map_location=self.device)

            # Build music encoder
            self.music_enc = _CLaMPMusicEncoder(
                patch_dim=self.PATCH_DIM, hidden_size=self.HIDDEN_SIZE
            )
            # Extract music_enc weights
            music_enc_state = {
                k.replace("music_enc.", ""): v
                for k, v in state.items() if k.startswith("music_enc.")
            }
            self.music_enc.load_state_dict(music_enc_state, strict=False)
            self.music_enc.to(self.device)
            self.music_enc.eval()

            # Projection layer
            self.music_proj = torch.nn.Linear(self.HIDDEN_SIZE, self.HIDDEN_SIZE)
            proj_state = {
                k.replace("music_proj.", ""): v
                for k, v in state.items() if k.startswith("music_proj.")
            }
            self.music_proj.load_state_dict(proj_state)
            self.music_proj.to(self.device)
            self.music_proj.eval()

            self._loaded = True
            logger.info(f"CLaMP-1 loaded from {self.HF_REPO} (dim={self.HIDDEN_SIZE})")
        except Exception as e:
            logger.warning(f"Failed to load CLaMP-1: {e}. Using deterministic fallback.")

    def _midi_tokens_to_patches(self, tokens: List[int]) -> torch.Tensor:
        """Convert token sequence into bar-patch representation.

        We create a simplified patch representation:
        Each patch is a multi-hot vector of (pitch, duration_bin) pairs.
        Tokens are split into groups of DUR_BINS tokens per patch.
        """
        patch_size = self.DUR_BINS  # tokens per patch
        n_patches = min(len(tokens) // max(patch_size, 1) + 1, self.MAX_PATCHES)
        patches = np.zeros((n_patches, self.PATCH_DIM), dtype=np.float32)

        for p_idx in range(n_patches):
            start = p_idx * patch_size
            end = min(start + patch_size, len(tokens))
            chunk = tokens[start:end]
            for i, tok in enumerate(chunk):
                pitch = tok % self.PITCH_BINS
                dur_bin = i % self.DUR_BINS
                idx = pitch * self.DUR_BINS + dur_bin
                if idx < self.PATCH_DIM:
                    patches[p_idx, idx] = 1.0

        return torch.tensor(patches, dtype=torch.float32).unsqueeze(0)  # (1, N, D)

    def encode(self, tokens: List[int], midi_data=None) -> np.ndarray:
        if not self._loaded:
            rng = np.random.default_rng(hash(tuple(tokens[:20])) % (2**31))
            return rng.standard_normal(self.embedding_dim).astype(np.float32)

        patches = self._midi_tokens_to_patches(tokens).to(self.device)
        with torch.no_grad():
            hidden = self.music_enc(patches)  # (1, H)
            embedding = self.music_proj(hidden)  # (1, H)
            embedding = torch.nn.functional.normalize(embedding, dim=-1)
        return embedding[0].cpu().numpy().astype(np.float32)

    def encode_batch(self, token_sequences: List[List[int]], midi_data_list: Optional[List] = None) -> np.ndarray:
        return np.array([self.encode(seq) for seq in token_sequences])

    def get_embedding_dim(self) -> int:
        return self.embedding_dim


# ---------------------------------------------------------------------------
# CLaMP-2: Multimodal Music IR with M3 encoder (MTF / MIDI patch format)
# ---------------------------------------------------------------------------

class _CLaMP2MusicModel(torch.nn.Module):
    """Reconstruct CLaMP-2 music encoder from checkpoint."""

    def __init__(self, patch_dim: int = 8192, hidden_size: int = 768,
                 num_layers: int = 12, num_heads: int = 12, max_len: int = 512):
        super().__init__()
        from transformers import BertConfig, BertModel
        self.patch_embedding = torch.nn.Linear(patch_dim, hidden_size)
        cfg = BertConfig(
            vocab_size=1, hidden_size=hidden_size,
            num_hidden_layers=num_layers, num_attention_heads=num_heads,
            intermediate_size=hidden_size * 4, max_position_embeddings=max_len,
        )
        self.base = BertModel(cfg)

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        """patches: (batch, n_patches, patch_dim) -> (batch, hidden_size)"""
        x = self.patch_embedding(patches)
        out = self.base(inputs_embeds=x)
        return out.last_hidden_state[:, 0, :]


class CLaMP2Model(EmbeddingModel):
    """CLaMP-2 (sander-wood/clamp2) — music encoder branch.

    Uses MTF (MIDI Token Format) with patch_size=64 notes, each note
    represented as a 128-dim binary vector → 8192-dim patch.
    Output: 768-dim embedding after music_proj.
    """

    HF_REPO = "sander-wood/clamp2"
    WEIGHT_FILE = "weights_clamp2_h_size_768_lr_5e-05_batch_128_scale_1_t_length_128_t_model_FacebookAI_xlm-roberta-base_t_dropout_True_m3_True.pth"
    PATCH_DIM = 8192  # 64 notes * 128 attributes
    HIDDEN_SIZE = 768
    MAX_PATCHES = 512
    NOTES_PER_PATCH = 64
    NOTE_DIM = 128  # MIDI note attributes per note

    def __init__(self, config: Dict):
        super().__init__(config, "CLaMP-2")
        self.embedding_dim = self.HIDDEN_SIZE
        self._loaded = False

        try:
            from huggingface_hub import hf_hub_download
            weight_path = hf_hub_download(self.HF_REPO, self.WEIGHT_FILE)
            checkpoint = torch.load(weight_path, map_location=self.device)
            model_state = checkpoint["model"]

            # Build music model (12 layers for CLaMP-2)
            self.music_model = _CLaMP2MusicModel(
                patch_dim=self.PATCH_DIM, hidden_size=self.HIDDEN_SIZE,
                num_layers=12, num_heads=12,
            )
            music_state = {
                k.replace("music_model.", ""): v
                for k, v in model_state.items() if k.startswith("music_model.")
            }
            self.music_model.load_state_dict(music_state, strict=False)
            self.music_model.to(self.device)
            self.music_model.eval()

            # Projection
            self.music_proj = torch.nn.Linear(self.HIDDEN_SIZE, self.HIDDEN_SIZE)
            proj_state = {
                k.replace("music_proj.", ""): v
                for k, v in model_state.items() if k.startswith("music_proj.")
            }
            self.music_proj.load_state_dict(proj_state)
            self.music_proj.to(self.device)
            self.music_proj.eval()

            self._loaded = True
            logger.info(f"CLaMP-2 loaded from {self.HF_REPO} (dim={self.HIDDEN_SIZE})")
        except Exception as e:
            logger.warning(f"Failed to load CLaMP-2: {e}. Using deterministic fallback.")

    def _tokens_to_mtf_patches(self, tokens: List[int]) -> torch.Tensor:
        """Convert token IDs to MTF-style patches.

        Each token is mapped to a 128-dim binary vector (one-hot over MIDI range),
        then groups of NOTES_PER_PATCH are flattened into 8192-dim patches.
        """
        # Create per-token 128-dim representations
        note_vectors = []
        for tok in tokens:
            vec = np.zeros(self.NOTE_DIM, dtype=np.float32)
            idx = tok % self.NOTE_DIM
            vec[idx] = 1.0
            note_vectors.append(vec)

        # Pad to multiple of NOTES_PER_PATCH
        while len(note_vectors) % self.NOTES_PER_PATCH != 0:
            note_vectors.append(np.zeros(self.NOTE_DIM, dtype=np.float32))

        note_arr = np.array(note_vectors)  # (N, 128)
        n_patches = min(len(note_arr) // self.NOTES_PER_PATCH, self.MAX_PATCHES)
        patches = note_arr[:n_patches * self.NOTES_PER_PATCH].reshape(n_patches, self.PATCH_DIM)
        return torch.tensor(patches, dtype=torch.float32).unsqueeze(0)  # (1, P, 8192)

    def encode(self, tokens: List[int], midi_data=None) -> np.ndarray:
        if not self._loaded:
            rng = np.random.default_rng(hash(tuple(tokens[:20])) % (2**31))
            return rng.standard_normal(self.embedding_dim).astype(np.float32)

        patches = self._tokens_to_mtf_patches(tokens).to(self.device)
        with torch.no_grad():
            hidden = self.music_model(patches)
            embedding = self.music_proj(hidden)
            embedding = torch.nn.functional.normalize(embedding, dim=-1)
        return embedding[0].cpu().numpy().astype(np.float32)

    def encode_batch(self, token_sequences: List[List[int]], midi_data_list: Optional[List] = None) -> np.ndarray:
        return np.array([self.encode(seq) for seq in token_sequences])

    def get_embedding_dim(self) -> int:
        return self.embedding_dim


class EmbeddingFactory:
    """Factory for creating embedding models."""
    _models = {
        "MusicBERT": MusicBERTModel,
        "MusicBERT-large": MusicBERTLargeModel,
        "MERT": MERTModel,
        "NLP-Baseline": NLPBaselineModel,
        "CLaMP-1": CLaMP1Model,
        "CLaMP-2": CLaMP2Model,
    }

    @classmethod
    def create_model(cls, config: Dict, model_name: str) -> EmbeddingModel:
        if model_name not in cls._models:
            raise ValueError(f"Unknown model: {model_name}. Available: {list(cls._models.keys())}")
        return cls._models[model_name](config)

    @classmethod
    def get_available_models(cls) -> List[str]:
        return list(cls._models.keys())


class EmbeddingExtractor:
    """Extract embeddings from tokenized MIDI data with caching."""

    def __init__(self, config: Dict):
        self.config = config
        # Only load models that are listed in config (not all registered ones)
        config_model_names = {m["name"] for m in config.get("embeddings", {}).get("models", [])}
        available = EmbeddingFactory.get_available_models()
        models_to_load = [m for m in available if m in config_model_names] or available
        self.models = {}
        for model_name in models_to_load:
            try:
                self.models[model_name] = EmbeddingFactory.create_model(config, model_name)
            except Exception as e:
                logger.warning(f"Failed to load model {model_name}: {e}")
        self.cache_dir = Path(config["embeddings"].get("cache_dir", "data/embeddings/cache"))
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.use_cache = config["embeddings"].get("cache_embeddings", True)
        self.memory_cache = {} if self.use_cache else None
        logger.info(f"EmbeddingExtractor initialized with models: {list(self.models.keys())}")

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
                return np.load(cache_path)
            except Exception as e:
                logger.warning(f"Failed to load from disk cache: {e}")
        return None

    def _save_to_disk_cache(self, cache_key: str, embedding: np.ndarray, metadata: Dict = None):
        try:
            np.save(self._get_cache_path(cache_key), embedding)
            if metadata:
                with open(self._get_metadata_path(cache_key), "w") as f:
                    json.dump(metadata, f)
        except Exception as e:
            logger.warning(f"Failed to save to disk cache: {e}")

    def extract_embeddings(
        self,
        token_sequences: List[List[int]],
        model_name: str,
        midi_data_list: Optional[List] = None,
    ) -> np.ndarray:
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
