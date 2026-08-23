"""Machine-enforced config <-> pose pairing.

WHY THIS EXISTS. AGENTS.md sec.7 has carried the rule "A gain belongs to the
case it was derived for. Never carry one across" since 2026-08-16, in prose.
It was violated again anyway, silently, and cost a ~20 h search plus five
capture-envelope grids:

    config/ur5e_mujoco_torque_x_task_yz_corridor_qp_orientation_cbf_balance.yaml
    states in its own header that it is for ARM_Q0 with wrist_2 = -90 deg and
    pendulum_attachment_realrod.xml, and that its kp_x = 1532.672 is
    400 * Lambda_xx with Lambda_xx = 3.831680 MEASURED AT THAT POSE. Every run
    dispatched it at wrist_2 ~ 0 (the SINGULAR ARM_Q0) with
    pendulum_attachment.xml, where the same header records Lambda_xx = 5.9298
    -- i.e. kp_x too small by 1.55x. Nothing errored. The YAML parsed, the runs
    completed, the envelopes came back 0/117, and the failure looked like a
    control-design problem rather than a dispatch problem.

That is the signature this repo keeps getting caught by: the operation appears
to succeed (AGENTS.md sec.7, "VERIFY THE EFFECT, NOT THE INVOCATION"). A comment
in a YAML header is not enforcement -- nothing reads it. This module makes the
pairing a machine-checked precondition instead, in the same spirit as
simulation.ur5e_pendulum_compose.PENDULUM_ASSET_ARM_Q, which already made the
asset<->pose pairing a table rather than a convention.

CONTRACT. A config may declare, at top level:

    provenance:
      derived_for:
        arm_q_rad: [-2.3688, -2.1801, -1.8838, -0.7962, -1.5707963, 0.0206]
        pendulum_xml: pendulum_attachment_realrod.xml   # basename or path
      notes: "kp_x = 400 * Lambda_xx, Lambda_xx = 3.831680 measured here"

``check_config_pose`` then REFUSES to let that config run at any other pose or
asset. A config with no ``provenance:`` block is permitted (so this lands
without breaking the existing config set) but is reported as undeclared, and
setting the environment variable ``REAL_CARTPOLE_STRICT_PROVENANCE=1`` promotes
undeclared to a hard error for runs that want the stricter gate.

Deliberately NOT a tolerance-tuning knob: ``atol_rad`` defaults to 1e-6, i.e.
"the same pose", not "a nearby pose". A gain that is only valid within some
neighbourhood has never been characterised in this repo, so there is no
defensible non-trivial tolerance to offer. Widening it is a decision with
evidence behind it, not a default.

numpy-only, no simulator import -- controller_core stays simulator-independent.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np

__all__ = [
    "ConfigPoseMismatchError",
    "ConfigProvenance",
    "parse_provenance",
    "check_config_pose",
    "describe_provenance",
]

STRICT_ENV_VAR = "REAL_CARTPOLE_STRICT_PROVENANCE"

JOINT_NAMES = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow",
    "wrist_1",
    "wrist_2",
    "wrist_3",
)


class ConfigPoseMismatchError(RuntimeError):
    """Raised when a config is dispatched at a pose/asset it was not derived for."""


@dataclass(frozen=True)
class ConfigProvenance:
    """The (pose, asset) pair a config's gains were actually derived at."""

    arm_q_rad: np.ndarray | None = None
    pendulum_xml: str | None = None
    # The controller FAMILY the gains were fitted for. A gain is only valid for
    # its (controller, frame, pose, row set, role) -- AGENTS.md sec.7 -- and
    # controller was the one axis this guard originally ignored. Concretely:
    # the corridor-QP balance configs carry kp_x = 400*Lambda_xx = 1532.672,
    # which plain OSC reads as a raw P-gain against its own validated 400, i.e.
    # 3.8x too stiff -- and the pose/asset check passes cleanly. Separately,
    # those configs declare task_excluded_joints: [0] to keep shoulder_pan OUT
    # of the task; plain OSC has no exclusion mechanism, so dispatching one of
    # them as `impedance` silently frees pan (measured: 17.5 deg of pan swing in
    # a run that was supposed to hold it fixed).
    controller_kind: str | None = None
    notes: str = ""
    source: str = ""
    mismatches: tuple[str, ...] = field(default_factory=tuple)

    @property
    def declared(self) -> bool:
        return (self.arm_q_rad is not None or self.pendulum_xml is not None
                or self.controller_kind is not None)

    def as_dict(self) -> dict[str, Any]:
        return {
            "arm_q_rad": None if self.arm_q_rad is None else self.arm_q_rad.tolist(),
            "pendulum_xml": self.pendulum_xml,
            "controller_kind": self.controller_kind,
            "notes": self.notes,
            "source": self.source,
            "mismatches": list(self.mismatches),
        }


def parse_provenance(cfg: dict, *, source: str = "") -> ConfigProvenance:
    """Reads the optional top-level ``provenance:`` block out of a parsed config.

    Returns an undeclared ConfigProvenance when the block is absent, rather
    than raising -- absence is a known state of the existing config set, and
    the caller decides how strict to be about it.
    """
    block = (cfg or {}).get("provenance") or {}
    if not isinstance(block, dict):
        raise ConfigPoseMismatchError(
            f"{source or '<config>'}: 'provenance' must be a mapping, got {type(block).__name__}"
        )
    derived = block.get("derived_for") or {}
    if not isinstance(derived, dict):
        raise ConfigPoseMismatchError(
            f"{source or '<config>'}: 'provenance.derived_for' must be a mapping, "
            f"got {type(derived).__name__}"
        )

    arm_q = derived.get("arm_q_rad")
    if arm_q is not None:
        arm_q = np.asarray(arm_q, dtype=np.float64).reshape(-1)
        if arm_q.size != 6:
            raise ConfigPoseMismatchError(
                f"{source or '<config>'}: provenance.derived_for.arm_q_rad must have 6 "
                f"entries, got {arm_q.size}"
            )

    pendulum_xml = derived.get("pendulum_xml")
    ck = derived.get("controller_kind")
    return ConfigProvenance(
        arm_q_rad=arm_q,
        pendulum_xml=None if pendulum_xml is None else str(pendulum_xml),
        controller_kind=None if ck is None else str(ck),
        notes=str(block.get("notes", "")),
        source=source,
    )


def _fmt_pose_diff(declared: np.ndarray, actual: np.ndarray, atol_rad: float) -> str:
    lines = [
        f"  {'joint':<14} {'declared':>14} {'actual':>14} {'delta':>14}",
    ]
    for name, d, a in zip(JOINT_NAMES, declared, actual):
        delta = float(a - d)
        flag = "   <-- DIFFERS" if abs(delta) > atol_rad else ""
        lines.append(f"  {name:<14} {d:14.7f} {a:14.7f} {delta:14.7f}{flag}")
    return "\n".join(lines)


def check_config_pose(
    cfg: dict,
    arm_q: Sequence[float] | np.ndarray,
    pendulum_xml: str | Path | None = None,
    *,
    controller_kind: str | None = None,
    config_name: str = "",
    allow_mismatch: bool = False,
    atol_rad: float = 1e-6,
    strict_undeclared: bool | None = None,
) -> ConfigProvenance:
    """Refuses to run ``cfg`` at a pose/asset it was not derived for.

    Returns the parsed provenance (with ``.mismatches`` populated when
    ``allow_mismatch`` suppressed a real mismatch, so the run's own output can
    record that it ran off-provenance). Raises ConfigPoseMismatchError
    otherwise.

    ``allow_mismatch`` exists for the deliberate cross-pose experiment -- the
    case where running a config off its derivation pose IS the measurement. It
    must be passed explicitly at the call site; there is no config field that
    can grant it, because the whole failure mode this guards is a config being
    trusted about its own applicability.
    """
    prov = parse_provenance(cfg, source=config_name)
    if strict_undeclared is None:
        strict_undeclared = os.environ.get(STRICT_ENV_VAR, "") not in ("", "0", "false")

    if not prov.declared:
        if strict_undeclared:
            raise ConfigPoseMismatchError(
                f"{config_name or '<config>'} declares no 'provenance.derived_for' block "
                f"and {STRICT_ENV_VAR} is set. Add the pose/asset its gains were derived "
                f"at, or unset the variable."
            )
        return prov

    actual_q = np.asarray(arm_q, dtype=np.float64).reshape(-1)
    if actual_q.size != 6:
        raise ConfigPoseMismatchError(
            f"arm_q must have 6 entries, got {actual_q.size}"
        )

    problems: list[str] = []

    if prov.arm_q_rad is not None:
        delta = np.abs(actual_q - prov.arm_q_rad)
        if np.any(delta > atol_rad):
            worst = int(np.argmax(delta))
            problems.append(
                f"POSE MISMATCH (largest: {JOINT_NAMES[worst]} off by "
                f"{delta[worst]:.6f} rad = {np.degrees(delta[worst]):.3f} deg)\n"
                + _fmt_pose_diff(prov.arm_q_rad, actual_q, atol_rad)
            )

    if prov.pendulum_xml is not None and pendulum_xml is not None:
        want = Path(str(prov.pendulum_xml)).name
        got = Path(str(pendulum_xml)).name
        if want != got:
            problems.append(f"ASSET MISMATCH: declared {want!r}, dispatched {got!r}")

    if (prov.controller_kind is not None and controller_kind is not None
            and str(controller_kind) != prov.controller_kind):
        problems.append(
            f"CONTROLLER MISMATCH: gains declared for {prov.controller_kind!r}, "
            f"dispatched as {str(controller_kind)!r}"
        )

    if not problems:
        return prov

    detail = "\n".join(problems)
    header = (
        f"{config_name or '<config>'} was derived for a different case and is being "
        f"dispatched off it.\n{detail}"
    )
    if prov.notes:
        header += f"\n  provenance notes: {prov.notes}"

    if allow_mismatch:
        return ConfigProvenance(
            arm_q_rad=prov.arm_q_rad,
            pendulum_xml=prov.pendulum_xml,
            notes=prov.notes,
            source=prov.source,
            mismatches=tuple(problems),
        )

    raise ConfigPoseMismatchError(
        header
        + "\n\nGains are only valid at the (controller, task frame, pose, row set, role) "
          "they were fitted at -- see AGENTS.md sec.7. Either dispatch at the declared "
          "case, re-derive the gains for this one into a NEW config, or pass "
          "allow_mismatch=True at the call site if running off-provenance is itself the "
          "measurement."
    )


def describe_provenance(prov: ConfigProvenance) -> str:
    if not prov.declared:
        return "provenance: UNDECLARED (config does not state the pose its gains came from)"
    bits = ["provenance: declared"]
    if prov.pendulum_xml:
        bits.append(f"  asset : {Path(prov.pendulum_xml).name}")
    if prov.arm_q_rad is not None:
        bits.append(f"  pose  : {np.round(prov.arm_q_rad, 7).tolist()}")
    if prov.mismatches:
        bits.append("  !! RUNNING OFF-PROVENANCE (explicitly allowed):")
        bits.extend(f"     {m.splitlines()[0]}" for m in prov.mismatches)
    return "\n".join(bits)
