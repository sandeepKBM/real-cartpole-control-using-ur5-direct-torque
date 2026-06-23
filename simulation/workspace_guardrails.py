"""Workspace guardrails extracted from the external Einksul/mujocoSim scene.

This module is diagnostic and visualization only. It is not a robot safety
layer and it must not be wired directly into real-arm emergency stop logic.

The extracted scene lives in MuJoCo world coordinates and uses SI units
(meters / radians). The current config only encodes the exact primitives found
in the inspected scene XML:

- floor plane
- slanted wall plane
- tools-side box obstacle
- desk / PC-side box obstacle

Unknown lab-specific boundaries (door side, robot base exclusion, cartpole rail
safe range) are documented as unresolved placeholders in the YAML config.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

try:  # pragma: no cover - imported lazily in environments without PyYAML
    import yaml
except Exception as exc:  # pragma: no cover - exercised when yaml is missing
    yaml = None  # type: ignore[assignment]
    _YAML_IMPORT_ERROR = exc
else:  # pragma: no cover - trivial
    _YAML_IMPORT_ERROR = None

try:  # pragma: no cover - PIL is available in the repo runtime, but keep import defensive.
    from PIL import Image, ImageDraw, ImageFont
except Exception as exc:  # pragma: no cover
    Image = ImageDraw = ImageFont = None  # type: ignore[assignment]
    _PIL_IMPORT_ERROR = exc
else:  # pragma: no cover - trivial
    _PIL_IMPORT_ERROR = None


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GUARDRAIL_CONFIG = REPO_ROOT / "config" / "lab_workspace_guardrails.yaml"


def _as_float_array(value: Any, *, name: str, length: int | None = None) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64).reshape(-1)
    if length is not None and arr.shape[0] != length:
        raise ValueError(f"{name} must have length {length}; got shape {np.asarray(value).shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains NaN/Inf")
    return arr


def _as_list_of_float_arrays(rows: Iterable[Any] | None, *, name: str) -> list[np.ndarray]:
    if rows is None:
        return []
    out: list[np.ndarray] = []
    for idx, row in enumerate(rows):
        out.append(_as_float_array(row, name=f"{name}[{idx}]"))
    return out


def _normalize_quaternion_wxyz(quat: Any, *, name: str) -> np.ndarray:
    arr = _as_float_array(quat, name=name, length=4)
    norm = float(np.linalg.norm(arr))
    if norm <= 0.0 or not math.isfinite(norm):
        raise ValueError(f"{name} must be a non-zero finite quaternion")
    return arr / norm


def _quat_wxyz_to_rot(quat: Any) -> np.ndarray:
    w, x, y, z = _normalize_quaternion_wxyz(quat, name="quaternion")
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _rot_to_quat_wxyz(rot: np.ndarray) -> np.ndarray:
    """Convert a 3x3 rotation matrix to a wxyz quaternion."""
    r = np.asarray(rot, dtype=np.float64).reshape(3, 3)
    trace = float(np.trace(r))
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (r[2, 1] - r[1, 2]) / s
        y = (r[0, 2] - r[2, 0]) / s
        z = (r[1, 0] - r[0, 1]) / s
    elif r[0, 0] > r[1, 1] and r[0, 0] > r[2, 2]:
        s = math.sqrt(1.0 + r[0, 0] - r[1, 1] - r[2, 2]) * 2.0
        w = (r[2, 1] - r[1, 2]) / s
        x = 0.25 * s
        y = (r[0, 1] + r[1, 0]) / s
        z = (r[0, 2] + r[2, 0]) / s
    elif r[1, 1] > r[2, 2]:
        s = math.sqrt(1.0 + r[1, 1] - r[0, 0] - r[2, 2]) * 2.0
        w = (r[0, 2] - r[2, 0]) / s
        x = (r[0, 1] + r[1, 0]) / s
        y = 0.25 * s
        z = (r[1, 2] + r[2, 1]) / s
    else:
        s = math.sqrt(1.0 + r[2, 2] - r[0, 0] - r[1, 1]) * 2.0
        w = (r[1, 0] - r[0, 1]) / s
        x = (r[0, 2] + r[2, 0]) / s
        y = (r[1, 2] + r[2, 1]) / s
        z = 0.25 * s
    quat = np.array([w, x, y, z], dtype=np.float64)
    quat /= max(float(np.linalg.norm(quat)), 1e-12)
    return quat


def _world_to_local(point: np.ndarray, pose_position: np.ndarray, pose_quaternion: np.ndarray) -> np.ndarray:
    rot = _quat_wxyz_to_rot(pose_quaternion)
    return rot.T @ (point - pose_position)


def _local_to_world(point: np.ndarray, pose_position: np.ndarray, pose_quaternion: np.ndarray) -> np.ndarray:
    rot = _quat_wxyz_to_rot(pose_quaternion)
    return rot @ point + pose_position


@dataclass
class SourceRef:
    file: str
    lines: list[int] = field(default_factory=list)
    repo: str | None = None
    note: str | None = None


@dataclass
class BoundarySpec:
    name: str
    primitive: str
    frame: str
    pose_position_m: np.ndarray | None = None
    pose_quaternion_wxyz: np.ndarray | None = None
    size_m: np.ndarray | None = None
    margin_m: float = 0.0
    rough_margin_m: float = 0.0
    confidence: str = "exact"
    description: str = ""
    source: SourceRef | None = None
    active: bool = True
    unresolved_reason: str | None = None
    tags: list[str] = field(default_factory=list)

    def validate(self) -> None:
        if not self.name:
            raise ValueError("boundary.name is required")
        if not self.primitive:
            raise ValueError(f"{self.name}: primitive is required")
        if not self.frame:
            raise ValueError(f"{self.name}: frame is required")
        if self.primitive not in {"plane", "box", "cylinder", "capsule", "polygon", "halfspace"}:
            raise ValueError(f"{self.name}: unsupported primitive {self.primitive!r}")
        if self.margin_m < 0.0 or not math.isfinite(self.margin_m):
            raise ValueError(f"{self.name}: margin_m must be finite and non-negative")
        if self.rough_margin_m < 0.0 or not math.isfinite(self.rough_margin_m):
            raise ValueError(f"{self.name}: rough_margin_m must be finite and non-negative")
        if self.pose_position_m is not None:
            _as_float_array(self.pose_position_m, name=f"{self.name}.pose_position_m", length=3)
        if self.pose_quaternion_wxyz is not None:
            _normalize_quaternion_wxyz(self.pose_quaternion_wxyz, name=f"{self.name}.pose_quaternion_wxyz")
        if self.size_m is not None:
            size = _as_float_array(self.size_m, name=f"{self.name}.size_m")
            if self.primitive == "box" and size.shape[0] != 3:
                raise ValueError(f"{self.name}: box size_m must have length 3")

    @property
    def effective_margin_m(self) -> float:
        return float(self.margin_m + self.rough_margin_m)

    def position(self) -> np.ndarray:
        if self.pose_position_m is None:
            raise ValueError(f"{self.name}: pose_position_m missing")
        return _as_float_array(self.pose_position_m, name=f"{self.name}.pose_position_m", length=3)

    def quaternion(self) -> np.ndarray:
        if self.pose_quaternion_wxyz is None:
            raise ValueError(f"{self.name}: pose_quaternion_wxyz missing")
        return _normalize_quaternion_wxyz(self.pose_quaternion_wxyz, name=f"{self.name}.pose_quaternion_wxyz")

    def half_sizes(self) -> np.ndarray:
        if self.size_m is None:
            raise ValueError(f"{self.name}: size_m missing")
        return _as_float_array(self.size_m, name=f"{self.name}.size_m", length=3)


@dataclass
class GuardrailConfig:
    frame: str
    units: dict[str, str]
    boundaries: list[BoundarySpec] = field(default_factory=list)
    unresolved: list[dict[str, Any]] = field(default_factory=list)
    source_repo: str | None = None
    source_checkout: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if not self.frame:
            raise ValueError("frame is required")
        if not isinstance(self.units, dict):
            raise ValueError("units must be a mapping")
        if self.units.get("position") not in {"m", "meter", "meters"}:
            raise ValueError("units.position must be meters")
        if self.units.get("angle") not in {"rad", "radian", "radians"}:
            raise ValueError("units.angle must be radians")
        if not self.boundaries:
            raise ValueError("at least one active boundary is required")
        for boundary in self.boundaries:
            boundary.validate()

    def accepted_frames(self) -> set[str]:
        frames = {self.frame}
        aliases = self.raw.get("frame_aliases", [])
        if isinstance(aliases, list):
            frames.update(str(v) for v in aliases)
        return frames


@dataclass
class BoundaryAssessment:
    name: str
    primitive: str
    state: str
    signed_distance_m: float | None
    distance_m: float | None
    margin_m: float
    message: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "primitive": self.primitive,
            "state": self.state,
            "signed_distance_m": self.signed_distance_m,
            "distance_m": self.distance_m,
            "margin_m": self.margin_m,
            "message": self.message,
        }


@dataclass
class GuardrailDecision:
    state: str
    message: str
    frame: str
    margin_m: float
    boundary_name: str | None = None
    signed_distance_m: float | None = None
    distance_m: float | None = None
    violated_boundary_names: list[str] = field(default_factory=list)
    near_boundary_names: list[str] = field(default_factory=list)
    assessments: list[BoundaryAssessment] = field(default_factory=list)
    timestamp_ns: int | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "state": self.state,
            "message": self.message,
            "frame": self.frame,
            "margin_m": self.margin_m,
            "boundary_name": self.boundary_name,
            "signed_distance_m": self.signed_distance_m,
            "distance_m": self.distance_m,
            "violated_boundary_names": list(self.violated_boundary_names),
            "near_boundary_names": list(self.near_boundary_names),
            "assessments": [a.as_dict() for a in self.assessments],
            "timestamp_ns": self.timestamp_ns,
        }


def _load_yaml(path: Path) -> dict[str, Any]:
    if yaml is None:  # pragma: no cover - depends on environment
        raise RuntimeError(
            "PyYAML is required to load guardrail configs"
        ) from _YAML_IMPORT_ERROR
    data = yaml.safe_load(Path(path).read_text())
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a YAML mapping at the top level")
    return data


def load_guardrail_config(path: str | Path = DEFAULT_GUARDRAIL_CONFIG) -> GuardrailConfig:
    raw = _load_yaml(Path(path))
    frame = str(raw.get("frame", "")).strip()
    units = raw.get("units", {})
    source_repo = raw.get("source_repo")
    source_checkout = raw.get("source_checkout")
    boundaries_raw = raw.get("boundaries", [])
    unresolved = raw.get("unresolved", [])
    boundaries: list[BoundarySpec] = []
    for item in boundaries_raw:
        if not isinstance(item, dict):
            raise ValueError("Each boundary entry must be a mapping")
        source = None
        if "source" in item and isinstance(item["source"], dict):
            src = item["source"]
            lines = src.get("lines", [])
            if isinstance(lines, int):
                lines = [lines]
            source = SourceRef(
                repo=str(src.get("repo")) if src.get("repo") is not None else None,
                file=str(src.get("file", "")),
                lines=[int(v) for v in lines],
                note=str(src.get("note")) if src.get("note") is not None else None,
            )
        pose = item.get("pose", {})
        pose_position = None
        pose_quaternion = None
        if isinstance(pose, dict):
            if "position" in pose:
                pose_position = _as_float_array(pose["position"], name=f"{item.get('name')}.pose.position", length=3)
            if "quaternion" in pose:
                pose_quaternion = _normalize_quaternion_wxyz(
                    pose["quaternion"], name=f"{item.get('name')}.pose.quaternion"
                )
        size = item.get("size")
        size_arr = None if size is None else _as_float_array(size, name=f"{item.get('name')}.size")
        boundary = BoundarySpec(
            name=str(item.get("name", "")),
            primitive=str(item.get("primitive", "")),
            frame=str(item.get("frame", frame)),
            pose_position_m=pose_position,
            pose_quaternion_wxyz=pose_quaternion,
            size_m=size_arr,
            margin_m=float(item.get("margin_m", 0.0)),
            rough_margin_m=float(item.get("rough_margin_m", 0.0)),
            confidence=str(item.get("confidence", "exact")),
            description=str(item.get("description", "")),
            source=source,
            active=bool(item.get("active", True)),
            unresolved_reason=str(item.get("unresolved_reason")) if item.get("unresolved_reason") is not None else None,
            tags=[str(v) for v in item.get("tags", [])] if isinstance(item.get("tags", []), list) else [],
        )
        boundary.validate()
        boundaries.append(boundary)

    config = GuardrailConfig(
        frame=frame,
        units=units if isinstance(units, dict) else {},
        boundaries=boundaries,
        unresolved=list(unresolved) if isinstance(unresolved, list) else [],
        source_repo=str(source_repo) if source_repo is not None else None,
        source_checkout=str(source_checkout) if source_checkout is not None else None,
        raw=raw,
    )
    config.validate()
    return config


def _plane_signed_distance(point: np.ndarray, boundary: BoundarySpec) -> float:
    pos = boundary.position()
    rot = _quat_wxyz_to_rot(boundary.quaternion())
    normal = rot[:, 2]
    return float(np.dot(normal, point - pos))


def _box_signed_distance(point: np.ndarray, boundary: BoundarySpec) -> float:
    pos = boundary.position()
    rot = _quat_wxyz_to_rot(boundary.quaternion())
    local = rot.T @ (point - pos)
    half = boundary.half_sizes()
    delta = np.abs(local) - half
    outside = np.maximum(delta, 0.0)
    outside_dist = float(np.linalg.norm(outside))
    inside_dist = float(min(np.max(delta), 0.0))
    return outside_dist + inside_dist


def _project_to_xy(boundary: BoundarySpec) -> np.ndarray:
    if boundary.primitive == "box":
        pos = boundary.position()
        rot = _quat_wxyz_to_rot(boundary.quaternion())
        half = boundary.half_sizes()
        corners = []
        for sx in (-1.0, 1.0):
            for sy in (-1.0, 1.0):
                local = np.array([sx * half[0], sy * half[1], 0.0], dtype=np.float64)
                world = _local_to_world(local, pos, boundary.quaternion())
                corners.append(world[:2])
        return np.asarray(corners, dtype=np.float64)
    if boundary.primitive == "plane":
        pos = boundary.position()
        rot = _quat_wxyz_to_rot(boundary.quaternion())
        normal = rot[:, 2]
        normal_xy = normal[:2]
        if np.linalg.norm(normal_xy) < 1e-9:
            return np.asarray([[pos[0], pos[1]]], dtype=np.float64)
        return np.asarray([[pos[0], pos[1], normal_xy[0], normal_xy[1]]], dtype=np.float64)
    return np.zeros((0, 2), dtype=np.float64)


def _assess_boundary(point: np.ndarray, boundary: BoundarySpec, margin_m: float | None = None) -> BoundaryAssessment:
    eff_margin = boundary.effective_margin_m if margin_m is None else float(margin_m)
    if not boundary.active:
        return BoundaryAssessment(
            name=boundary.name,
            primitive=boundary.primitive,
            state="unknown",
            signed_distance_m=None,
            distance_m=None,
            margin_m=eff_margin,
            message=boundary.unresolved_reason or "inactive boundary",
        )
    if boundary.primitive == "plane":
        signed = _plane_signed_distance(point, boundary)
        if signed < 0.0:
            state = "outside"
            msg = f"{boundary.name} plane crossed by {abs(signed):.4f} m"
        elif signed < eff_margin:
            state = "near_boundary"
            msg = f"{boundary.name} plane within {eff_margin:.4f} m"
        else:
            state = "inside"
            msg = f"{boundary.name} plane safe by {signed:.4f} m"
        return BoundaryAssessment(
            name=boundary.name,
            primitive=boundary.primitive,
            state=state,
            signed_distance_m=float(signed),
            distance_m=float(abs(signed)),
            margin_m=eff_margin,
            message=msg,
        )
    if boundary.primitive == "box":
        signed = _box_signed_distance(point, boundary)
        if signed < 0.0:
            state = "outside"
            msg = f"{boundary.name} box violated by {abs(signed):.4f} m"
        elif signed < eff_margin:
            state = "near_boundary"
            msg = f"{boundary.name} box within {eff_margin:.4f} m"
        else:
            state = "inside"
            msg = f"{boundary.name} box safe by {signed:.4f} m"
        return BoundaryAssessment(
            name=boundary.name,
            primitive=boundary.primitive,
            state=state,
            signed_distance_m=float(signed),
            distance_m=float(abs(signed)),
            margin_m=eff_margin,
            message=msg,
        )
    return BoundaryAssessment(
        name=boundary.name,
        primitive=boundary.primitive,
        state="unknown",
        signed_distance_m=None,
        distance_m=None,
        margin_m=eff_margin,
        message=f"primitive {boundary.primitive!r} is not implemented in the checker",
    )


def _combine_assessments(
    *,
    frame: str,
    margin_m: float,
    assessments: list[BoundaryAssessment],
    timestamp_ns: int | None = None,
) -> GuardrailDecision:
    violation = next((a for a in assessments if a.state == "outside"), None)
    near = [a.name for a in assessments if a.state == "near_boundary"]
    unknown = any(a.state == "unknown" for a in assessments)
    if violation is not None:
        return GuardrailDecision(
            state="outside",
            message=violation.message,
            frame=frame,
            margin_m=margin_m,
            boundary_name=violation.name,
            signed_distance_m=violation.signed_distance_m,
            distance_m=violation.distance_m,
            violated_boundary_names=[violation.name],
            near_boundary_names=near,
            assessments=assessments,
            timestamp_ns=timestamp_ns,
        )
    if near:
        first = next(a for a in assessments if a.name == near[0])
        return GuardrailDecision(
            state="near_boundary",
            message=first.message,
            frame=frame,
            margin_m=margin_m,
            boundary_name=first.name,
            signed_distance_m=first.signed_distance_m,
            distance_m=first.distance_m,
            violated_boundary_names=[],
            near_boundary_names=near,
            assessments=assessments,
            timestamp_ns=timestamp_ns,
        )
    if unknown:
        first = next(a for a in assessments if a.state == "unknown")
        return GuardrailDecision(
            state="unknown",
            message=first.message,
            frame=frame,
            margin_m=margin_m,
            boundary_name=first.name,
            signed_distance_m=first.signed_distance_m,
            distance_m=first.distance_m,
            violated_boundary_names=[],
            near_boundary_names=[],
            assessments=assessments,
            timestamp_ns=timestamp_ns,
        )
    return GuardrailDecision(
        state="inside",
        message="inside all active guardrails",
        frame=frame,
        margin_m=margin_m,
        violated_boundary_names=[],
        near_boundary_names=[],
        assessments=assessments,
        timestamp_ns=timestamp_ns,
    )


def _resolve_frame(config: GuardrailConfig, frame: str | None) -> tuple[str, bool]:
    if frame is None:
        return config.frame, True
    frame = str(frame).strip()
    if frame in config.accepted_frames():
        return frame, True
    return frame, False


def check_point(
    point: Any,
    config: GuardrailConfig,
    *,
    frame: str | None = None,
    margin_m: float = 0.0,
    timestamp_ns: int | None = None,
) -> GuardrailDecision:
    point_arr = _as_float_array(point, name="point", length=3)
    resolved_frame, frame_ok = _resolve_frame(config, frame)
    if not frame_ok:
        return GuardrailDecision(
            state="unknown",
            message=f"frame {resolved_frame!r} is not compatible with guardrail frame {config.frame!r}",
            frame=resolved_frame,
            margin_m=float(margin_m),
            assessments=[],
        )
    effective_margin = float(margin_m)
    assessments = [_assess_boundary(point_arr, boundary, effective_margin) for boundary in config.boundaries]
    return _combine_assessments(frame=resolved_frame, margin_m=effective_margin, assessments=assessments, timestamp_ns=timestamp_ns)


def check_tcp_pose(
    tcp_pose: Any,
    config: GuardrailConfig,
    *,
    frame: str | None = None,
    margin_m: float = 0.0,
    timestamp_ns: int | None = None,
) -> GuardrailDecision:
    pose_arr = _as_float_array(tcp_pose, name="tcp_pose")
    if pose_arr.shape[0] < 3:
        raise ValueError("tcp_pose must contain at least 3 position elements")
    return check_point(
        pose_arr[:3],
        config,
        frame=frame,
        margin_m=margin_m,
        timestamp_ns=timestamp_ns,
    )


def check_trajectory(
    points: Any,
    config: GuardrailConfig,
    *,
    frame: str | None = None,
    margin_m: float = 0.0,
    timestamp_ns: Sequence[int] | None = None,
) -> GuardrailDecision:
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] < 3:
        raise ValueError("trajectory must have shape (N, 3+) or be convertible to such")
    resolved_frame, frame_ok = _resolve_frame(config, frame)
    if not frame_ok:
        return GuardrailDecision(
            state="unknown",
            message=f"frame {resolved_frame!r} is not compatible with guardrail frame {config.frame!r}",
            frame=resolved_frame,
            margin_m=float(margin_m),
            assessments=[],
        )
    if timestamp_ns is not None and len(timestamp_ns) != pts.shape[0]:
        raise ValueError("timestamp_ns length must match trajectory length")
    assessments: list[BoundaryAssessment] = []
    all_assessments: list[BoundaryAssessment] = []
    overall = "inside"
    violated_name = None
    near_names: list[str] = []
    for idx, point in enumerate(pts[:, :3]):
        per_point = [_assess_boundary(point, boundary, margin_m) for boundary in config.boundaries]
        all_assessments.extend(per_point)
        decision = _combine_assessments(
            frame=resolved_frame,
            margin_m=margin_m,
            assessments=per_point,
            timestamp_ns=None if timestamp_ns is None else int(timestamp_ns[idx]),
        )
        if decision.state == "outside":
            return GuardrailDecision(
                state="outside",
                message=f"trajectory violates {decision.boundary_name}",
                frame=resolved_frame,
                margin_m=margin_m,
                boundary_name=decision.boundary_name,
                signed_distance_m=decision.signed_distance_m,
                distance_m=decision.distance_m,
                violated_boundary_names=[decision.boundary_name] if decision.boundary_name else [],
                near_boundary_names=near_names,
                assessments=all_assessments,
                timestamp_ns=None if timestamp_ns is None else int(timestamp_ns[idx]),
            )
        if decision.state == "near_boundary":
            near_names = sorted(set(near_names + decision.near_boundary_names))
            overall = "near_boundary"
            assessments = per_point
            violated_name = decision.boundary_name
        elif decision.state == "unknown" and overall == "inside":
            overall = "unknown"
            assessments = per_point
            violated_name = decision.boundary_name
    if overall == "near_boundary":
        first = next((a for a in assessments if a.state == "near_boundary"), None)
        return GuardrailDecision(
            state="near_boundary",
            message=first.message if first is not None else "trajectory nears a guardrail",
            frame=resolved_frame,
            margin_m=margin_m,
            boundary_name=violated_name,
            signed_distance_m=None if first is None else first.signed_distance_m,
            distance_m=None if first is None else first.distance_m,
            violated_boundary_names=[],
            near_boundary_names=near_names,
            assessments=all_assessments,
        )
    if overall == "unknown":
        first = next((a for a in assessments if a.state == "unknown"), None)
        return GuardrailDecision(
            state="unknown",
            message=first.message if first is not None else "trajectory has unknown guardrail status",
            frame=resolved_frame,
            margin_m=margin_m,
            boundary_name=violated_name,
            signed_distance_m=None,
            distance_m=None,
            violated_boundary_names=[],
            near_boundary_names=[],
            assessments=all_assessments,
        )
    return GuardrailDecision(
        state="inside",
        message="trajectory stays inside all active guardrails",
        frame=resolved_frame,
        margin_m=margin_m,
        assessments=all_assessments,
    )


def _topdown_bounds(
    config: GuardrailConfig,
    *,
    points_xy: np.ndarray | None = None,
    padding_m: float = 0.25,
) -> tuple[float, float, float, float]:
    xs: list[float] = []
    ys: list[float] = []
    for boundary in config.boundaries:
        if not boundary.active:
            continue
        if boundary.primitive == "box":
            corners = _project_to_xy(boundary)
            if corners.size:
                xs.extend(corners[:, 0].tolist())
                ys.extend(corners[:, 1].tolist())
        elif boundary.primitive == "plane":
            pos = boundary.position()
            xs.append(float(pos[0]))
            ys.append(float(pos[1]))
    if points_xy is not None and points_xy.size:
        xs.extend(points_xy[:, 0].tolist())
        ys.extend(points_xy[:, 1].tolist())
    if not xs or not ys:
        return (-1.5, 1.5, -1.5, 1.5)
    return (
        float(min(xs) - padding_m),
        float(max(xs) + padding_m),
        float(min(ys) - padding_m),
        float(max(ys) + padding_m),
    )


def _xy_to_px(
    x: float,
    y: float,
    *,
    bounds: tuple[float, float, float, float],
    rect: tuple[int, int, int, int],
) -> tuple[float, float]:
    xmin, xmax, ymin, ymax = bounds
    left, top, right, bottom = rect
    width = max(1.0, float(right - left))
    height = max(1.0, float(bottom - top))
    px = left + (x - xmin) / max(xmax - xmin, 1e-9) * width
    py = bottom - (y - ymin) / max(ymax - ymin, 1e-9) * height
    return float(px), float(py)


def _draw_plane_line(
    draw: Any,
    boundary: BoundarySpec,
    *,
    bounds: tuple[float, float, float, float],
    rect: tuple[int, int, int, int],
    color: tuple[int, int, int, int],
    width: int = 2,
) -> None:
    pos = boundary.position()
    rot = _quat_wxyz_to_rot(boundary.quaternion())
    normal = rot[:, 2]
    normal_xy = normal[:2]
    if np.linalg.norm(normal_xy) < 1e-9:
        y = _xy_to_px(pos[0], pos[1], bounds=bounds, rect=rect)[1]
        draw.line((rect[0], y, rect[2], y), fill=color, width=width)
        return
    xmin, xmax, ymin, ymax = bounds
    d = float(np.dot(normal_xy, pos[:2]))
    candidates: list[tuple[float, float]] = []
    # Intersections with x = xmin/xmax
    for x in (xmin, xmax):
        if abs(normal_xy[1]) > 1e-12:
            y = (d - normal_xy[0] * x) / normal_xy[1]
            if ymin - 1e-9 <= y <= ymax + 1e-9:
                candidates.append((x, y))
    # Intersections with y = ymin/ymax
    for y in (ymin, ymax):
        if abs(normal_xy[0]) > 1e-12:
            x = (d - normal_xy[1] * y) / normal_xy[0]
            if xmin - 1e-9 <= x <= xmax + 1e-9:
                candidates.append((x, y))
    if len(candidates) < 2:
        return
    a = _xy_to_px(*candidates[0], bounds=bounds, rect=rect)
    b = _xy_to_px(*candidates[1], bounds=bounds, rect=rect)
    draw.line((*a, *b), fill=color, width=width)


def _draw_box_polygon(
    draw: Any,
    boundary: BoundarySpec,
    *,
    bounds: tuple[float, float, float, float],
    rect: tuple[int, int, int, int],
    outline: tuple[int, int, int, int],
    fill: tuple[int, int, int, int],
    show_labels: bool = False,
) -> None:
    pos = boundary.position()
    quat = boundary.quaternion()
    half = boundary.half_sizes()
    corners_local = [
        np.array([sx * half[0], sy * half[1], 0.0], dtype=np.float64)
        for sx in (-1.0, 1.0)
        for sy in (-1.0, 1.0)
    ]
    corners_world = np.array([_local_to_world(corner, pos, quat)[:2] for corner in corners_local], dtype=np.float64)
    center_xy = np.mean(corners_world, axis=0)
    # Order corners around the centroid for a proper polygon.
    angles = np.arctan2(corners_world[:, 1] - center_xy[1], corners_world[:, 0] - center_xy[0])
    order = np.argsort(angles)
    ordered = corners_world[order]
    px_points = [_xy_to_px(float(x), float(y), bounds=bounds, rect=rect) for x, y in ordered]
    draw.polygon(px_points, outline=outline, fill=fill)
    if show_labels:
        cx, cy = _xy_to_px(float(center_xy[0]), float(center_xy[1]), bounds=bounds, rect=rect)
        draw.text((cx + 3, cy - 8), boundary.name, fill=outline, font=ImageFont.load_default())


def overlay_guardrails_on_frame(
    frame: np.ndarray,
    config: GuardrailConfig,
    *,
    trajectory_xyz: Any | None = None,
    current_xyz: Any | None = None,
    desired_xyz: Any | None = None,
    decision: GuardrailDecision | None = None,
    guardrail_margin_m: float = 0.0,
    show_labels: bool = False,
    inset_size: tuple[int, int] = (320, 260),
    inset_corner: str = "bottom-right",
) -> np.ndarray:
    if Image is None:  # pragma: no cover - defensive fallback
        raise RuntimeError("Pillow is required for guardrail overlays") from _PIL_IMPORT_ERROR
    image = Image.fromarray(np.asarray(frame, dtype=np.uint8))
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    font = ImageFont.load_default()

    traj_xy = None
    if trajectory_xyz is not None:
        arr = np.asarray(trajectory_xyz, dtype=np.float64)
        if arr.size:
            arr = np.atleast_2d(arr)
            if arr.shape[1] >= 2:
                traj_xy = arr[:, :2]

    current_xy = None if current_xyz is None else _as_float_array(current_xyz, name="current_xyz", length=3)[:2]
    desired_xy = None if desired_xyz is None else _as_float_array(desired_xyz, name="desired_xyz", length=3)[:2]

    bounds = _topdown_bounds(config, points_xy=traj_xy, padding_m=max(guardrail_margin_m, 0.25))
    inset_w, inset_h = inset_size
    margin_px = 16
    if inset_corner == "bottom-right":
        left = image.size[0] - inset_w - margin_px
        top = image.size[1] - inset_h - margin_px
    elif inset_corner == "bottom-left":
        left = margin_px
        top = image.size[1] - inset_h - margin_px
    elif inset_corner == "top-left":
        left = margin_px
        top = margin_px
    elif inset_corner == "top-right":
        left = image.size[0] - inset_w - margin_px
        top = margin_px
    else:
        raise ValueError(
            f"Unknown inset_corner={inset_corner!r}; expected top-left, top-right, bottom-left, or bottom-right"
        )
    rect = (left, top, left + inset_w, top + inset_h)

    draw.rounded_rectangle(rect, radius=10, fill=(12, 12, 12, 170), outline=(255, 255, 255, 64))

    # Draw the workspace boundaries.
    for boundary in config.boundaries:
        if not boundary.active:
            continue
        if boundary.primitive == "plane":
            _draw_plane_line(draw, boundary, bounds=bounds, rect=rect, color=(255, 96, 96, 220), width=3)
            if show_labels:
                pos = boundary.position()
                px, py = _xy_to_px(float(pos[0]), float(pos[1]), bounds=bounds, rect=rect)
                draw.text((px + 4, py + 4), boundary.name, fill=(255, 180, 180, 255), font=font)
        elif boundary.primitive == "box":
            _draw_box_polygon(
                draw,
                boundary,
                bounds=bounds,
                rect=rect,
                outline=(255, 96, 96, 230),
                fill=(255, 96, 96, 55),
                show_labels=show_labels,
            )

    # Draw the trajectory path.
    if traj_xy is not None and traj_xy.shape[0] >= 2:
        pts = [_xy_to_px(float(x), float(y), bounds=bounds, rect=rect) for x, y in traj_xy]
        draw.line(pts, fill=(96, 200, 255, 220), width=2)

    # Current and desired points.
    def _draw_point(xy: np.ndarray | None, fill: tuple[int, int, int, int], outline: tuple[int, int, int, int], radius: int = 5) -> None:
        if xy is None:
            return
        px, py = _xy_to_px(float(xy[0]), float(xy[1]), bounds=bounds, rect=rect)
        draw.ellipse((px - radius, py - radius, px + radius, py + radius), fill=fill, outline=outline, width=2)

    if current_xy is not None:
        if decision is not None and decision.state == "outside":
            color = (255, 72, 72, 240)
        elif decision is not None and decision.state == "near_boundary":
            color = (255, 192, 64, 240)
        else:
            color = (64, 255, 120, 240)
        _draw_point(current_xy, fill=color, outline=(255, 255, 255, 240), radius=6)
    if desired_xy is not None:
        _draw_point(desired_xy, fill=(70, 180, 255, 180), outline=(255, 255, 255, 220), radius=4)

    # Small legend / summary.
    status = "unknown"
    violated = "none"
    if decision is not None:
        status = decision.state
        violated = ", ".join(decision.violated_boundary_names or decision.near_boundary_names or [decision.boundary_name or "none"])
    legend = [
        f"guardrails: {status}",
        f"boundary: {violated}",
        f"frame: {config.frame}",
    ]
    text_y = top + 8
    for line in legend:
        draw.text((left + 10, text_y), line, fill=(245, 245, 245, 255), font=font)
        text_y += 15

    combined = Image.alpha_composite(image.convert("RGBA"), overlay)
    return np.asarray(combined.convert("RGB"))


def serialize_decision(decision: GuardrailDecision) -> str:
    return json.dumps(decision.as_dict(), separators=(",", ":"), sort_keys=True)


def boundary_summary(config: GuardrailConfig) -> dict[str, Any]:
    return {
        "frame": config.frame,
        "units": config.units,
        "source_repo": config.source_repo,
        "source_checkout": config.source_checkout,
        "boundaries": [
            {
                "name": b.name,
                "primitive": b.primitive,
                "frame": b.frame,
                "pose_position_m": None if b.pose_position_m is None else np.asarray(b.pose_position_m, dtype=np.float64).tolist(),
                "pose_quaternion_wxyz": None if b.pose_quaternion_wxyz is None else np.asarray(b.pose_quaternion_wxyz, dtype=np.float64).tolist(),
                "size_m": None if b.size_m is None else np.asarray(b.size_m, dtype=np.float64).tolist(),
                "margin_m": b.margin_m,
                "rough_margin_m": b.rough_margin_m,
                "effective_margin_m": b.effective_margin_m,
                "confidence": b.confidence,
                "description": b.description,
                "source": None
                if b.source is None
                else {
                    "repo": b.source.repo,
                    "file": b.source.file,
                    "lines": list(b.source.lines),
                    "note": b.source.note,
                },
                "active": b.active,
                "unresolved_reason": b.unresolved_reason,
                "tags": list(b.tags),
            }
            for b in config.boundaries
        ],
        "unresolved": list(config.unresolved),
    }
