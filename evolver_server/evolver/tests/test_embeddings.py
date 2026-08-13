# pyright: reportMissingImports=false
from __future__ import annotations

import hashlib
import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from evolver_server.evolver.embeddings import (
    HashingEmbedder,
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


class TestCreateEmbedder:
    def test_hashing_mode_returns_hashing_embedder(self):
        emb = create_embedder(mode="hashing", dimensions=32)
        assert isinstance(emb, HashingEmbedder)
        assert emb.dimensions == 32

    def test_default_mode_is_hashing(self):
        emb = create_embedder()
        assert isinstance(emb, HashingEmbedder)
