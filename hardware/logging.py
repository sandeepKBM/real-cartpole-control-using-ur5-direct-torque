"""Structured JSON / JSONL logging helpers for hardware staging scripts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, TextIO


def _convert(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _convert(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_convert(v) for v in obj]
    if hasattr(obj, "tolist"):
        return obj.tolist()
    return obj


def json_dumps_safe(obj: Any, *, indent: int | None = None) -> str:
    """Serialize an object while converting numpy arrays to lists."""

    return json.dumps(_convert(obj), indent=indent, separators=(",", ":") if indent is None else None)


def write_json(path: str | Path, payload: Any, *, indent: int = 2) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json_dumps_safe(payload, indent=indent), encoding="utf-8")


class JsonlWriter:
    """Append one JSON object per line."""

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._fp: TextIO | None = None

    def __enter__(self) -> "JsonlWriter":
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._fp = self._path.open("w", encoding="utf-8")
        return self

    def __exit__(self, *args: Any) -> None:
        if self._fp is not None:
            self._fp.close()
            self._fp = None

    def write_row(self, row: dict[str, Any]) -> None:
        if self._fp is None:
            raise RuntimeError("JsonlWriter not opened; use as a context manager")
        self._fp.write(json_dumps_safe(row) + "\n")
