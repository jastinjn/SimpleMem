# pyright: reportMissingImports=false
from __future__ import annotations

import hashlib
import math
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from marklymem.evolver.embeddings import (
    HashingEmbedder,
    OpenAIEmbedder,
    cosine_similarity,
    create_embedder,
)


class TestHashingEmbedder:
    def test_encode_deterministic(self):
        e = HashingEmbedder(dimensions=64)
        v1 = e.encode("the quick brown fox")
        v2 = e.encode("the quick brown fox")
        assert v1 == v2

    def test_encode_l2_normalized(self):
        e = HashingEmbedder(dimensions=64)
        v = e.encode("some text here")
        norm = math.sqrt(sum(x * x for x in v))
        assert norm == pytest.approx(1.0, abs=1e-6)

    def test_encode_empty_string(self):
        e = HashingEmbedder(dimensions=64)
        v = e.encode("")
        assert all(x == 0.0 for x in v)

    def test_encode_single_token_hits_expected_index(self):
        dims = 64
        e = HashingEmbedder(dimensions=dims)
        token = "hello"
        idx = int(hashlib.sha256(token.encode("utf-8")).hexdigest(), 16) % dims
        v = e.encode(token)
        # Only one bucket is non-zero (before normalisation its value was 1.0;
        # after L2-norm of a unit vector it stays 1.0 since norm=1).
        assert v[idx] == pytest.approx(1.0)
        assert sum(1 for x in v if x != 0.0) == 1

    def test_encode_dimensions_respected(self):
        e = HashingEmbedder(dimensions=16)
        v = e.encode("test input")
        assert len(v) == 16

    def test_different_texts_produce_different_vectors(self):
        e = HashingEmbedder(dimensions=64)
        assert e.encode("apple") != e.encode("orange")


class TestCosineSimilarity:
    def test_identical_vectors(self):
        e = HashingEmbedder(dimensions=64)
        v = e.encode("hello world")
        assert cosine_similarity(v, v) == pytest.approx(1.0, abs=1e-6)

    def test_empty_left(self):
        e = HashingEmbedder(dimensions=64)
        v = e.encode("hello")
        assert cosine_similarity([], v) == 0.0

    def test_empty_right(self):
        e = HashingEmbedder(dimensions=64)
        v = e.encode("hello")
        assert cosine_similarity(v, []) == 0.0

    def test_length_mismatch(self):
        assert cosine_similarity([1.0, 0.0], [1.0, 0.0, 0.0]) == 0.0

    def test_orthogonal_vectors(self):
        # Two unit vectors with no overlap.
        a = [1.0, 0.0, 0.0, 0.0]
        b = [0.0, 1.0, 0.0, 0.0]
        assert cosine_similarity(a, b) == pytest.approx(0.0)

    def test_partial_similarity(self):
        e = HashingEmbedder(dimensions=64)
        v_db = e.encode("database")
        v_db2 = e.encode("database backup")
        sim = cosine_similarity(v_db, v_db2)
        assert 0.0 < sim < 1.0


class TestHashingEmbedderBatch:
    async def test_encode_batch_matches_individual_encode(self):
        e = HashingEmbedder(dimensions=64)
        texts = ["hello world", "foo bar", "another text here"]
        batch = await e.encode_batch(texts)
        individual = [e.encode(t) for t in texts]
        assert batch == individual

    async def test_encode_batch_empty_returns_empty(self):
        e = HashingEmbedder(dimensions=64)
        assert await e.encode_batch([]) == []

    async def test_encode_batch_preserves_order(self):
        e = HashingEmbedder(dimensions=64)
        texts = [f"text number {i}" for i in range(20)]
        batch = await e.encode_batch(texts)
        assert batch == [e.encode(t) for t in texts]


def _make_openai_embedder(mock_settings=None) -> OpenAIEmbedder:
    """Build an OpenAIEmbedder with all OpenAI clients replaced by mocks."""
    if mock_settings is None:
        mock_settings = MagicMock()
        mock_settings.OPENAI_API_KEY = "sk-test"
        mock_settings.OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"

    with (
        patch("marklymem.evolver.embeddings.OpenAIEmbedder.__init__.__globals__"),
        patch("marklymem.config.get_settings", return_value=mock_settings),
        patch("openai.OpenAI"),
        patch("openai.AsyncOpenAI"),
    ):
        embedder = OpenAIEmbedder.__new__(OpenAIEmbedder)
        embedder._model = "text-embedding-3-small"
        embedder._dimensions = 64
        embedder._MAX_RETRIES = 3
        embedder._CHUNK_SIZE = 10
    return embedder


def _fake_embeddings(texts: list[str], dim: int = 4) -> list[list[float]]:
    """Return a unique deterministic embedding per text."""
    import hashlib
    result = []
    for i, t in enumerate(texts):
        h = int(hashlib.md5(t.encode()).hexdigest(), 16)
        result.append([float((h >> (j * 4)) & 0xF) / 15.0 for j in range(dim)])
    return result


class TestOpenAIEmbedderBatch:
    def _make_async_mock(self, chunk_size: int = 10, dim: int = 4):
        """Return (embedder, mock) where mock replaces _acall_api for call tracking."""
        embedder = OpenAIEmbedder.__new__(OpenAIEmbedder)
        embedder._model = "text-embedding-3-small"
        embedder._dimensions = dim
        embedder._MAX_RETRIES = 3
        embedder._CHUNK_SIZE = chunk_size

        async def fake_acall(texts):
            return _fake_embeddings(texts, dim), 0

        mock = AsyncMock(side_effect=fake_acall)
        embedder._acall_api = mock  # type: ignore[method-assign]
        return embedder, mock

    async def test_single_chunk_one_api_call(self):
        e, mock = self._make_async_mock(chunk_size=10)
        texts = [f"text {i}" for i in range(5)]
        result = await e.encode_batch(texts)
        assert mock.await_count == 1
        assert len(result) == 5

    async def test_multiple_chunks_correct_call_count(self):
        e, mock = self._make_async_mock(chunk_size=10)
        texts = [f"text {i}" for i in range(25)]
        result = await e.encode_batch(texts)
        # 25 texts → chunks of 10, 10, 5 → 3 calls
        assert mock.await_count == 3
        assert len(result) == 25

    async def test_results_in_input_order(self):
        e, _ = self._make_async_mock(chunk_size=10)
        texts = [f"unique text {i}" for i in range(15)]
        result = await e.encode_batch(texts)
        expected = _fake_embeddings(texts[:10]) + _fake_embeddings(texts[10:])
        assert result == expected

    async def test_empty_returns_empty(self):
        e, mock = self._make_async_mock()
        assert await e.encode_batch([]) == []
        mock.assert_not_awaited()

    async def test_retry_recovers_transient_error(self):
        embedder = OpenAIEmbedder.__new__(OpenAIEmbedder)
        embedder._model = "text-embedding-3-small"
        embedder._dimensions = 4
        embedder._MAX_RETRIES = 3
        embedder._CHUNK_SIZE = 10
        attempts = {"n": 0}

        def _make_data(texts):
            data = []
            for i, _ in enumerate(texts):
                m = MagicMock()
                m.index = i
                m.embedding = [0.1, 0.2, 0.3, 0.4]
                data.append(m)
            resp = MagicMock()
            resp.data = data
            return resp

        async def flaky_create(**kwargs):
            attempts["n"] += 1
            if attempts["n"] < 2:
                raise RuntimeError("transient")
            return _make_data(kwargs["input"])

        mock_async_client = MagicMock()
        mock_async_client.embeddings.create = AsyncMock(side_effect=flaky_create)
        embedder._async_client = mock_async_client

        with patch("marklymem.evolver.embeddings.asyncio.sleep", new_callable=AsyncMock):
            result, _ = await embedder._acall_api(["hello world test"])

        assert attempts["n"] == 2
        assert len(result) == 1


class TestCreateEmbedder:
    def test_hashing_mode_returns_hashing_embedder(self):
        emb = create_embedder(mode="hashing", dimensions=32)
        assert isinstance(emb, HashingEmbedder)
        assert emb.dimensions == 32

    def test_default_mode_is_hashing(self):
        emb = create_embedder()
        assert isinstance(emb, HashingEmbedder)
