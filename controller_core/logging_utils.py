"""Small helpers for structured logging (JSON / JSONL) without ROS."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, TextIO


def json_dumps_safe(obj: Any, *, indent: int | None = None) -> str:
    """Serialize ``obj`` with numpy arrays converted to lists."""

    def _conv(o: Any) -> Any:
        if isinstance(o, dict):
            return {str(k): _conv(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [_conv(v) for v in o]
        if hasattr(o, "tolist"):
            return o.tolist()
        return o

    return json.dumps(
        _conv(obj),
        indent=indent,
        separators=(",", ":") if indent is None else None,
    )


def write_json(path: str | Path, payload: Any, *, indent: int = 2) -> None:
    """Write ``payload`` as pretty-printed JSON, creating parent dirs."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json_dumps_safe(payload, indent=indent), encoding="utf-8")


class JsonlTraceWriter:
    """Append one JSON object per line."""

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._fp: TextIO | None = None

    def __enter__(self) -> "JsonlTraceWriter":
        self._fp = self._path.open("w", encoding="utf-8")
        return self

    def __exit__(self, *args: Any) -> None:
        if self._fp is not None:
            self._fp.close()
            self._fp = None

    def write_row(self, row: dict[str, Any]) -> None:
        if self._fp is None:
            raise RuntimeError("JsonlTraceWriter not opened; use context manager")
        self._fp.write(json_dumps_safe(row) + "\n")
