"""Structured JSON / JSONL logging helpers for hardware staging scripts.

Consolidated into :mod:`controller_core.logging_utils`; this module is a
compatibility re-export so existing hardware imports keep working.
"""

from __future__ import annotations

from controller_core.logging_utils import (  # noqa: F401
    JsonlTraceWriter,
    json_dumps_safe,
    write_json,
)

# Historical name used by the hardware staging scripts.
JsonlWriter = JsonlTraceWriter

__all__ = ["JsonlWriter", "JsonlTraceWriter", "json_dumps_safe", "write_json"]
