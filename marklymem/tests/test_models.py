# pyright: reportMissingImports=false
"""Unit tests for Pydantic request model validators — no DB or HTTP required."""

from __future__ import annotations

import pytest

from marklymem.utils.sanitize import normalise_namespace as _normalise_namespace


class TestNormaliseNamespace:
    def test_none_passthrough(self):
        assert _normalise_namespace(None) is None

    def test_valid_flat(self):
        assert _normalise_namespace("project") == "project"

    def test_valid_nested(self):
        assert _normalise_namespace("project/backend/auth") == "project/backend/auth"

    def test_valid_with_hyphen(self):
        assert _normalise_namespace("my-project") == "my-project"

    def test_valid_with_underscore(self):
        assert _normalise_namespace("project_ok") == "project_ok"

    def test_trailing_slash_stripped(self):
        assert _normalise_namespace("project/") == "project"

    def test_trailing_slash_nested_stripped(self):
        assert _normalise_namespace("project/backend/") == "project/backend"

    def test_leading_slash_allowed(self):
        assert _normalise_namespace("/project") == "/project"

    def test_leading_slash_nested_allowed(self):
        # AWS AgentCore convention: /agent/customer-id/preferences
        assert _normalise_namespace("/retail-agent/customer-123/preferences") == "/retail-agent/customer-123/preferences"

    def test_bare_slash_returns_none(self):
        # "/" strips to "" → treated as global namespace
        assert _normalise_namespace("/") is None

    def test_empty_string_returns_none(self):
        assert _normalise_namespace("") is None

    def test_double_slash_raises(self):
        with pytest.raises(ValueError, match="must not contain"):
            _normalise_namespace("project//backend")

    def test_invalid_char_percent_raises(self):
        with pytest.raises(ValueError, match="segments must contain only"):
            _normalise_namespace("project%scope")

    def test_invalid_char_space_raises(self):
        with pytest.raises(ValueError, match="segments must contain only"):
            _normalise_namespace("project scope")

    def test_invalid_char_dot_raises(self):
        with pytest.raises(ValueError, match="segments must contain only"):
            _normalise_namespace("project.scope")

    def test_underscore_within_segment_allowed(self):
        # "_" is a SQL LIKE wildcard but is allowed within identifiers
        assert _normalise_namespace("a_b") == "a_b"
