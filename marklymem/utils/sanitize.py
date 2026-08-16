from __future__ import annotations

import re

_NAMESPACE_SEGMENT_RE = re.compile(r"^[a-zA-Z0-9_-]+$")


def normalise_namespace(v: object) -> str | None:
    if v is None:
        return None
    if not isinstance(v, str):
        return v  # type: ignore[return-value]  # let Pydantic raise the type error
    v = v.rstrip("/")
    if not v:
        return None  # bare "/" or "" → global namespace
    if "//" in v:
        raise ValueError("namespace must not contain '//'")
    for segment in v.split("/"):
        if segment and not _NAMESPACE_SEGMENT_RE.match(segment):
            raise ValueError(
                "namespace segments must contain only letters, digits, hyphens, or underscores"
            )
    return v
