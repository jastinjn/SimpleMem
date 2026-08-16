from __future__ import annotations

import asyncio
import hashlib
import logging
import math
from abc import ABC, abstractmethod

from marklymem.utils import telemetry

logger = logging.getLogger(__name__)


class BaseEmbedder(ABC):
    """Abstract base class for memory embedders."""

    @abstractmethod
    def encode(self, text: str) -> list[float]:
        """Encode a single text into a vector."""

    async def encode_batch(self, texts: list[str]) -> list[list[float]]:
        """Encode multiple texts in parallel.

        Default implementation runs each ``encode`` call in its own thread
        concurrently. Subclasses with native async batching (e.g. OpenAIEmbedder)
        should override this.
        """
        if not texts:
            return []
        return list(await asyncio.gather(
            *(asyncio.to_thread(self.encode, t) for t in texts)
        ))

    @property
    @abstractmethod
    def dimensions(self) -> int:
        """Return the dimensionality of produced vectors."""

    @property
    def model(self) -> str:
        """Return the model identifier for this embedder."""
        return ""


class HashingEmbedder(BaseEmbedder):
    """Deterministic lightweight embedder for phase-1 optional semantic retrieval."""

    def __init__(self, dimensions: int = 64):
        self._dimensions = max(int(dimensions), 8)

    @property
    def dimensions(self) -> int:
        return self._dimensions

    @property
    def model(self) -> str:
        return "hashing"

    def encode(self, text: str) -> list[float]:
        vector = [0.0] * self._dimensions
        for token in _tokenize(text):
            index = int(hashlib.sha256(token.encode("utf-8")).hexdigest(), 16) % self._dimensions
            vector[index] += 1.0
        return _l2_normalize(vector)


class SentenceTransformerEmbedder(BaseEmbedder):
    """Semantic embedder using sentence-transformers models.

    Requires the ``sentence-transformers`` package (``pip install sentence-transformers``).
    Falls back to HashingEmbedder if the package is not installed.

    Default model: ``all-MiniLM-L6-v2`` (384-dimensional, fast, good general quality).
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._model = None
        self._dimensions_cache: int | None = None
        self._load_model()

    def _load_model(self) -> None:
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore[import-untyped]
            self._model = SentenceTransformer(self.model_name)
            # Cache dimensions from a probe encode
            probe = self._model.encode("probe", convert_to_numpy=True)
            self._dimensions_cache = len(probe)
            logger.info(
                "SentenceTransformerEmbedder loaded model=%s dims=%d",
                self.model_name,
                self._dimensions_cache,
            )
        except ImportError:
            logger.warning(
                "sentence-transformers not installed; "
                "SentenceTransformerEmbedder will raise on encode(). "
                "Install with: pip install sentence-transformers"
            )
            self._model = None
            self._dimensions_cache = 384  # default for all-MiniLM-L6-v2
        except Exception as exc:
            logger.warning(
                "Failed to load sentence-transformers model %s: %s",
                self.model_name,
                exc,
            )
            self._model = None
            self._dimensions_cache = 384

    @property
    def dimensions(self) -> int:
        return self._dimensions_cache or 384

    @property
    def model(self) -> str:
        return self.model_name

    @property
    def is_available(self) -> bool:
        """Whether the underlying model loaded successfully."""
        return self._model is not None

    def encode(self, text: str) -> list[float]:
        if self._model is None:
            raise RuntimeError(
                f"SentenceTransformerEmbedder model '{self.model_name}' not available. "
                "Install sentence-transformers: pip install sentence-transformers"
            )
        embedding = self._model.encode(text, convert_to_numpy=True, normalize_embeddings=True)
        return embedding.tolist()

    def encode_batch(self, texts: list[str]) -> list[list[float]]:
        if self._model is None:
            raise RuntimeError(
                f"SentenceTransformerEmbedder model '{self.model_name}' not available."
            )
        if not texts:
            return []
        embeddings = self._model.encode(
            texts, convert_to_numpy=True, normalize_embeddings=True, batch_size=32,
        )
        return [e.tolist() for e in embeddings]


class OpenAIEmbedder(BaseEmbedder):
    """Embedder using OpenAI text-embedding models (text-embedding-3-small/large).

    ``encode`` uses a synchronous client with retry/backoff.
    ``encode_batch`` uses an async client, splits input into chunks of
    ``batch_size`` (default 10), and runs all chunks in parallel.

    Note: ``dimensions`` truncation is only supported by text-embedding-3-* models,
    not text-embedding-ada-002.
    """

    _MAX_RETRIES: int = 3
    _CHUNK_SIZE: int = 10

    def __init__(self, dimensions: int = 1024):
        from ..config import get_settings
        try:
            from openai import AsyncOpenAI, OpenAI
        except ImportError:
            raise RuntimeError(
                "openai package not installed. Install with: pip install openai"
            )
        s = get_settings()
        self._client = OpenAI(api_key=s.OPENAI_API_KEY)
        self._async_client = AsyncOpenAI(api_key=s.OPENAI_API_KEY)
        self._model = s.OPENAI_EMBEDDING_MODEL
        self._dimensions = dimensions

    @property
    def dimensions(self) -> int:
        return self._dimensions

    @property
    def model(self) -> str:
        return self._model

    def _call_api(self, texts: list[str]) -> tuple[list[list[float]], int]:
        """Synchronous API call with retry/backoff. Returns (embeddings, total_tokens)."""
        import time
        for attempt in range(self._MAX_RETRIES):
            try:
                response = self._client.embeddings.create(
                    input=texts,
                    model=self._model,
                    dimensions=self._dimensions,
                )
                tokens: int = getattr(getattr(response, "usage", None), "total_tokens", 0) or 0
                return [d.embedding for d in sorted(response.data, key=lambda x: x.index)], tokens
            except Exception as exc:
                logger.debug("[Embedder] attempt %d error: %s", attempt + 1, exc)
                if attempt == self._MAX_RETRIES - 1:
                    raise
                time.sleep(2 * (attempt + 1))
        return [], 0  # unreachable

    async def _acall_api(self, texts: list[str]) -> tuple[list[list[float]], int]:
        """Async API call with retry/backoff. Returns (embeddings, total_tokens)."""
        for attempt in range(self._MAX_RETRIES):
            try:
                response = await self._async_client.embeddings.create(
                    input=texts,
                    model=self._model,
                    dimensions=self._dimensions,
                )
                tokens: int = getattr(getattr(response, "usage", None), "total_tokens", 0) or 0
                return [d.embedding for d in sorted(response.data, key=lambda x: x.index)], tokens
            except Exception as exc:
                logger.debug("[Embedder] async attempt %d error: %s", attempt + 1, exc)
                if attempt == self._MAX_RETRIES - 1:
                    raise
                await asyncio.sleep(2 * (attempt + 1))
        return [], 0  # unreachable

    def encode(self, text: str) -> list[float]:
        with telemetry.span(
            "embedding",
            embedder_type=type(self).__name__,
            model=self._model,
            embedding_dim=self._dimensions,
            text_count=1,
        ) as sp:
            embs, tokens = self._call_api([text])
            telemetry.record_usage(sp, input_tokens=tokens or None, output_tokens=None)
        return embs[0]

    async def encode_batch(self, texts: list[str]) -> list[list[float]]:
        """Encode texts in parallel chunks of ``_CHUNK_SIZE``.

        Emits a parent ``embedding`` span covering the whole batch, with one
        ``embedding.chunk`` generation span per API call nested inside it.
        """
        if not texts:
            return []
        chunks = [texts[i:i + self._CHUNK_SIZE] for i in range(0, len(texts), self._CHUNK_SIZE)]

        async def _call_chunk(chunk: list[str]) -> tuple[list[list[float]], int]:
            with telemetry.generation("embedding.chunk", model=self._model, text_count=len(chunk)) as gen:
                embs, tokens = await self._acall_api(chunk)
                telemetry.record_usage(gen, input_tokens=tokens or None, output_tokens=None)
            return embs, tokens

        with telemetry.span(
            "embedding",
            embedder_type=type(self).__name__,
            model=self._model,
            embedding_dim=self._dimensions,
            text_count=len(texts),
        ) as batch_span:
            chunk_results = await asyncio.gather(*(_call_chunk(chunk) for chunk in chunks))
            results: list[list[float]] = []
            total_tokens = 0
            for embs, tokens in chunk_results:
                results.extend(embs)
                total_tokens += tokens
            telemetry.record_usage(batch_span, input_tokens=total_tokens or None, output_tokens=None)
        return results


def create_embedder(mode: str = "hashing", dimensions: int = 1024) -> BaseEmbedder:
    """Create an embedder by mode.

    Args:
        mode: "hashing" for HashingEmbedder, "semantic" for OpenAIEmbedder.
        dimensions: Output vector size (passed to both embedder types).
    """
    if mode == "semantic":
        return OpenAIEmbedder(dimensions=dimensions)
    return HashingEmbedder(dimensions=dimensions)


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    return sum(a * b for a, b in zip(left, right))


def _l2_normalize(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(v * v for v in vector))
    if norm <= 0.0:
        return vector
    return [v / norm for v in vector]


def _tokenize(text: str) -> list[str]:
    token = []
    out: list[str] = []
    for ch in text.lower():
        if ch.isalnum() or ch in {"_", "-"}:
            token.append(ch)
            continue
        if token:
            out.append("".join(token))
            token = []
    if token:
        out.append("".join(token))
    return out
