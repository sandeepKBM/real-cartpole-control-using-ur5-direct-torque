#!/usr/bin/env python3
"""
CoppeliaSim-only torque diagnostics smoke test ladder.

Requires a running CoppeliaSim ZMQ server (see launch_coppeliasim_x_axis_headless.sh).
Does not touch MuJoCo.

Usage::

    # Terminal 1: start CoppeliaSim + RPC
    bash simulation/launch_coppeliasim_x_axis_headless.sh --probe-only

    # Terminal 2: run smoke ladder
    python simulation/run_coppelia_torque_diagnostics_smoke.py \\
        --host 127.0.0.1 --port 23000 \\
        --config ros2_ws/src/ur5_x_axis_controller_ros/config/controller_coppelia_bringup.yaml
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = (
    REPO_ROOT
    / "ros2_ws"
    / "src"
    / "ur5_x_axis_controller_ros"
    / "config"
    / "controller_coppelia_bringup.yaml"
)
DEFAULT_OUT = REPO_ROOT / "outputs" / "control_runs" / "coppelia_torque_diagnostics"
RUNNER = REPO_ROOT / "simulation" / "run_coppeliasim_x_axis_headless.py"
LAUNCHER = REPO_ROOT / "simulation" / "launch_coppeliasim_x_axis_headless.sh"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=23000)
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--duration", type=float, default=3.0)
    p.add_argument("--tests", nargs="*", default=[], help="Subset of test names to run.")
    p.add_argument(
        "--use-launcher",
        action="store_true",
        help="Start/stop CoppeliaSim per test via launch_coppeliasim_x_axis_headless.sh.",
    )
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def build_tests(duration: float) -> list[dict]:
    return [
        {
            "name": "passive",
            "label": "01_passive_no_control",
            "extra_args": [
                "--torque-diagnostics-mode",
                "passive",
                "--duration",
                str(duration),
                "--no-video",
                "--enable-coppelia-torque-diagnostics",
                "--save-controller-logs",
                "--save-controller-plots",
            ],
        },
        {
            "name": "zero_torque",
            "label": "02_zero_torque",
            "extra_args": [
                "--torque-diagnostics-mode",
                "passive",
                "--duration",
                str(duration),
                "--no-video",
                "--enable-coppelia-torque-diagnostics",
                "--save-controller-logs",
                "--save-controller-plots",
                "--torque-diagnostics-run-label",
                "02_zero_torque",
            ],
        },
        {
            "name": "hold_soft",
            "label": "03_hold_soft_impedance",
            "extra_args": [
                "--torque-diagnostics-mode",
                "hold_soft",
                "--impedance-gain-scale",
                "0.05",
                "--duration",
                str(duration),
                "--no-video",
                "--enable-coppelia-torque-diagnostics",
                "--save-controller-logs",
                "--save-controller-plots",
            ],
        },
        {
            "name": "gain_sweep",
            "label": "04_gain_sweep",
            "sweep": True,
            "scales": [0.05, 0.1, 0.2, 0.35, 0.5],
        },
        {
            "name": "sinusoid_joint",
            "label": "05_sinusoid_joint",
            "extra_args": [
                "--torque-diagnostics-mode",
                "sinusoid_joint",
                "--torque-diagnostics-joint-index",
                "5",
                "--impedance-gain-scale",
                "0.05",
                "--duration",
                str(duration),
                "--no-video",
            ],
        },
        {
            "name": "tiny_x_motion",
            "label": "06_tiny_x_motion",
            "extra_args": [
                "--torque-diagnostics-mode",
                "tiny_x_motion",
                "--duration",
                str(duration),
                "--no-video",
            ],
        },
        {
            "name": "ref_discontinuity",
            "label": "07_ref_discontinuity",
            "pair": True,
        },
    ]


def run_one(
    *,
    host: str,
    port: int,
    config: Path,
    output_dir: Path,
    label: str,
    extra_args: list[str],
    dry_run: bool,
    use_launcher: bool,
) -> dict:
    config_path = str(config.resolve())
    common_args = [
        "--host",
        host,
        "--port",
        str(port),
        "--config",
        config_path,
        "--enable-coppelia-torque-diagnostics",
        "--save-controller-logs",
        "--save-controller-plots",
        "--diagnostics-output-dir",
        str(output_dir),
        "--torque-diagnostics-run-label",
        label,
        "--trace-name",
        f"{label}.jsonl",
        "--summary-name",
        f"{label}_runner_summary.json",
        *extra_args,
    ]
    if use_launcher:
        cmd = ["bash", str(LAUNCHER), *common_args]
    else:
        cmd = [sys.executable, str(RUNNER), *common_args]
    print(f"\n=== RUN {label} ===", flush=True)
    print(" ".join(cmd), flush=True)
    if dry_run:
        return {"label": label, "pass": None, "dry_run": True}
    proc = subprocess.run(cmd, cwd=str(REPO_ROOT), check=False)
    summary_candidates = [
        output_dir / f"{label}_summary.json",
        output_dir / f"{label}_runner_summary.json",
        REPO_ROOT / "outputs" / "control_runs" / f"{label}_runner_summary.json",
    ]
    summary_path = next((p for p in summary_candidates if p.exists()), None)
    if summary_path is None:
        return {
            "label": label,
            "pass": False,
            "suspected_failure_reason": (
                "missing summary; checked: "
                + ", ".join(str(p) for p in summary_candidates)
            ),
            "exit_code": proc.returncode,
        }
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    diag = summary.get("torque_diagnostics") or summary
    return {
        "label": label,
        "pass": bool(diag.get("pass", summary.get("success"))),
        "suspected_failure_reason": diag.get("suspected_failure_reason"),
        "max_tau_fraction_overall": diag.get("max_tau_fraction_overall"),
        "exit_code": proc.returncode,
        "summary_path": str(summary_path),
    }


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    selected = set(args.tests) if args.tests else None
    results: list[dict] = []

    for spec in build_tests(args.duration):
        name = spec["name"]
        if selected is not None and name not in selected:
            continue
        if spec.get("sweep"):
            for scale in spec["scales"]:
                label = f"{spec['label']}_kp_kd_{scale:.2f}"
                extra = [
                    "--torque-diagnostics-mode",
                    "hold_soft",
                    "--impedance-gain-scale",
                    str(scale),
                    "--duration",
                    str(args.duration),
                    "--no-video",
                    "--enable-coppelia-torque-diagnostics",
                    "--save-controller-logs",
                ]
                results.append(
                    run_one(
                        host=args.host,
                        port=args.port,
                        config=args.config,
                        output_dir=args.output_dir,
                        label=label,
                        extra_args=extra,
                        dry_run=args.dry_run,
                        use_launcher=args.use_launcher,
                    )
                )
            continue
        if spec.get("pair"):
            for mode, suffix in (("ref_step", "raw_step"), ("ref_smooth", "smoothed")):
                label = f"{spec['label']}_{suffix}"
                extra = [
                    "--torque-diagnostics-mode",
                    mode,
                    "--duration",
                    str(args.duration),
                    "--no-video",
                ]
                results.append(
                    run_one(
                        host=args.host,
                        port=args.port,
                        config=args.config,
                        output_dir=args.output_dir,
                        label=label,
                        extra_args=extra,
                        dry_run=args.dry_run,
                        use_launcher=args.use_launcher,
                    )
                )
            continue
        results.append(
            run_one(
                host=args.host,
                port=args.port,
                config=args.config,
                output_dir=args.output_dir,
                label=spec["label"],
                extra_args=spec["extra_args"],
                dry_run=args.dry_run,
                use_launcher=args.use_launcher,
            )
        )

    ladder_summary = {
        "output_dir": str(args.output_dir),
        "plots_dir": str(args.output_dir),
        "logs_dir": str(args.output_dir),
        "pass_criteria": {
            "max_tau_fraction": 0.50,
            "torque_clipping": "none for hold/sinusoid/tiny_x",
            "torque_rate_clipping": "rare or zero",
            "nan_inf": "none",
            "guardrails": "none",
        },
        "results": results,
        "all_passed": all(r.get("pass") for r in results if r.get("pass") is not None),
    }
    out_path = args.output_dir / "smoke_ladder_summary.json"
    out_path.write_text(json.dumps(ladder_summary, indent=2), encoding="utf-8")
    print(json.dumps(ladder_summary, indent=2), flush=True)
    print(f"Saved ladder summary: {out_path}", flush=True)
    if args.dry_run:
        return 0
    return 0 if ladder_summary["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
