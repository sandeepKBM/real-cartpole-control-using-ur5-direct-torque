"""Unit test for tools/tuning_common.py's plotting helpers.

Regression test (2026-07-29 bug audit): ``_plot_candidate_rates`` referenced
``collections.defaultdict`` without importing it, so any default
(non-``--no-plot``) invocation of ``tune_ur5e_residual_impedance_transport.py``
that produced at least one row crashed with ``NameError`` after
``summary.csv``/``summary.json``/``best_settings.json``/``README.md`` were
already written but before the driver could exit 0.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from tools.tuning_common import _plot_candidate_rates  # noqa: E402


def test_plot_candidate_rates_does_not_crash(tmp_path: Path) -> None:
    rows = [
        {"candidate_label": "cand_a", "valid_transport": True},
        {"candidate_label": "cand_a", "valid_transport": False},
        {"candidate_label": "cand_b", "valid_transport": True},
    ]
    output_path = tmp_path / "candidate_rates.png"
    result = _plot_candidate_rates(rows, output_path, title="test")
    assert result == output_path
    assert output_path.exists()
    assert output_path.stat().st_size > 0
