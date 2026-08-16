"""Unit tests for the reduced-task (X + orientation) QP with a Y/Z corridor
(``controller_core/x_task_yz_corridor_qp/``).

These check the MATH and the STRUCTURE in isolation, against analytic
Jacobians and hand-computed constraint rows, so a sign error or a leaked Y/Z
term cannot hide behind a plausible-looking closed loop. The closed-loop proof
that the mechanism does anything on the real robot model lives in
``tests/mujoco/test_x_task_yz_corridor_qp_closed_loop.py``; this file
deliberately does not try to infer robot behavior from a synthetic Jacobian
(this repo has been burned by exactly that -- AGENTS.md).

THE CENTRAL, FALSIFIABLE CLAIM of the whole design is tested in section 3:
the QP Hessian is built from ``J_reduced`` (X + the 3 orientation rows) ONLY,
so with the soft Y/Z gains at zero, ANY change to the Y/Z state or target must
leave the Hessian and the nominal task torque BYTE-identical -- not merely
close. If that ever stops holding, this is not a reduced task, it is
zeroed-gain OSC.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from controller_core.box_qp import solve_box_qp  # noqa: E402
from controller_core.x_task_yz_corridor_qp import (  # noqa: E402
    XTaskYZCorridorQPConfig,
    XTaskYZCorridorQPController,
)
from controller_core.x_task_yz_corridor_qp import controller as controller_mod  # noqa: E402
from controller_core.x_task_yz_corridor_qp.parsing import (  # noqa: E402
    _parse_axis_row_sets,
    _parse_corridor_half_width,
    _parse_task_excluded_joints,
)


# --------------------------------------------------------------------------- #
# Fixtures: a synthetic but non-degenerate 6x6 Jacobian and mass matrix.
# --------------------------------------------------------------------------- #
def make_jacobian(seed: int = 3) -> np.ndarray:
    rng = np.random.default_rng(seed)
    jac = np.eye(6) + 0.25 * rng.normal(0.0, 1.0, (6, 6))
    return jac


def make_mass(seed: int = 5) -> np.ndarray:
    rng = np.random.default_rng(seed)
    a = rng.normal(0.0, 1.0, (6, 6))
    return a @ a.T + 6.0 * np.eye(6)


def make_state(
    *,
    jac: np.ndarray | None = None,
    mass: np.ndarray | None = None,
    ee_pos=(0.4, -0.2, 0.3),
    ee_vel=(0.0, 0.0, 0.0),
    qd=None,
    target_x: float = 0.45,
    target_ee_pos=None,
    mass_matrix: bool = True,
) -> dict:
    jac = make_jacobian() if jac is None else jac
    mass = make_mass() if mass is None else mass
    state = {
        "time": 0.0,
        "q": np.array([0.1, -1.2, 0.7, -0.9, 0.05, 0.2], dtype=np.float64),
        "qd": np.zeros(6) if qd is None else np.asarray(qd, dtype=np.float64),
        "ee_pos": np.asarray(ee_pos, dtype=np.float64),
        "ee_quat": np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
        "ee_lin_vel": np.asarray(ee_vel, dtype=np.float64),
        "ee_ang_vel": np.zeros(3, dtype=np.float64),
        "target_x": float(target_x),
        "target_x_vel": 0.0,
        "jacobian": jac,
    }
    if target_ee_pos is not None:
        state["target_ee_pos"] = np.asarray(target_ee_pos, dtype=np.float64)
    if mass_matrix:
        state["mass_matrix"] = mass
    return state


def make_config(**kwargs) -> XTaskYZCorridorQPConfig:
    base = dict(
        kp_x=400.0, kd_x=40.0,
        kp_y=0.0, kd_y=0.0, kp_z=0.0, kd_z=0.0,
        kp_rot=5.0, kd_rot=2.0,
        kp_posture=1.0, kd_posture=0.2, kd_joint=0.5,
        enforce_velocity_torque_bounds=False,
    )
    base.update(kwargs)
    return XTaskYZCorridorQPConfig(**base)


def capture_qp(monkeypatch):
    """Record every argument ``solve_constrained_box_qp`` is called with.

    The Hessian is an internal of ``compute()`` and is deliberately not part
    of the output dataclass (nothing in the torque path needs it), but the
    central claim of this design is a statement ABOUT the Hessian -- so it is
    intercepted at the one place it is consumed rather than re-derived by the
    test (which would prove the test's arithmetic, not the controller's).
    """
    calls: list[dict] = []
    real = controller_mod.solve_constrained_box_qp

    def spy(hessian, linear, lower, upper, a_ineq=None, b_ineq=None, **kw):
        calls.append(
            {
                "hessian": np.array(hessian, copy=True),
                "linear": np.array(linear, copy=True),
                "lower": np.array(lower, copy=True),
                "upper": np.array(upper, copy=True),
                "a_ineq": None if a_ineq is None else np.array(a_ineq, copy=True),
                "b_ineq": None if b_ineq is None else np.array(b_ineq, copy=True),
            }
        )
        return real(hessian, linear, lower, upper, a_ineq, b_ineq, **kw)

    monkeypatch.setattr(controller_mod, "solve_constrained_box_qp", spy)
    return calls


# --------------------------------------------------------------------------- #
# 1. The HOCBF constraint rows, against hand-computed algebra.
# --------------------------------------------------------------------------- #
def test_corridor_rows_match_hand_computed_case():
    j_row = np.array([0.3, -0.7, 0.2, 0.0, 0.1, -0.4])
    m_inv = np.diag([1.0, 0.5, 2.0, 4.0, 0.25, 1.5])
    bias = np.array([1.0, -2.0, 0.5, 0.0, 0.25, -0.75])
    qd = np.array([0.1, 0.2, -0.3, 0.05, 0.0, 0.4])
    value, lower, upper = 0.02, -0.05, 0.05
    a1, a2 = 7.0, 11.0

    a_max, b_max, a_min, b_min = XTaskYZCorridorQPController._corridor_rows(
        j_row=j_row, m_inv=m_inv, bias=bias, qd=qd,
        value=value, lower=lower, upper=upper, alpha1=a1, alpha2=a2,
    )

    lie = j_row @ m_inv
    v_axis = float(j_row @ qd)
    assert np.allclose(a_max[0], lie, atol=0.0, rtol=0.0)
    assert np.allclose(a_min[0], -lie, atol=0.0, rtol=0.0)
    assert b_max == pytest.approx(
        float(lie @ bias) - (a1 + a2) * v_axis + a1 * a2 * (upper - value), rel=1e-12
    )
    assert b_min == pytest.approx(
        -float(lie @ bias) + (a1 + a2) * v_axis + a1 * a2 * (value - lower), rel=1e-12
    )


def test_corridor_rows_encode_the_hocbf_condition_itself():
    """``A tau <= b`` must be algebraically identical to
    ``hddot + (a1+a2) hdot + a1 a2 h >= 0`` with
    ``hddot = -J_y M^-1 (tau - bias)`` for the upper wall."""
    rng = np.random.default_rng(17)
    j_row = rng.normal(0.0, 1.0, 6)
    m_inv = np.diag(rng.uniform(0.2, 2.0, 6))
    bias = rng.normal(0.0, 1.0, 6)
    qd = rng.normal(0.0, 0.5, 6)
    tau = rng.normal(0.0, 20.0, 6)
    value, lower, upper, a1, a2 = 0.01, -0.05, 0.05, 3.0, 9.0

    a_max, b_max, a_min, b_min = XTaskYZCorridorQPController._corridor_rows(
        j_row=j_row, m_inv=m_inv, bias=bias, qd=qd,
        value=value, lower=lower, upper=upper, alpha1=a1, alpha2=a2,
    )

    h = upper - value
    h_dot = -float(j_row @ qd)
    h_ddot = -float(j_row @ m_inv @ (tau - bias))
    hocbf = h_ddot + (a1 + a2) * h_dot + a1 * a2 * h
    assert float(a_max[0] @ tau - b_max) == pytest.approx(-hocbf, rel=1e-10, abs=1e-12)

    h = value - lower
    h_dot = +float(j_row @ qd)
    h_ddot = +float(j_row @ m_inv @ (tau - bias))
    hocbf = h_ddot + (a1 + a2) * h_dot + a1 * a2 * h
    assert float(a_min[0] @ tau - b_min) == pytest.approx(-hocbf, rel=1e-10, abs=1e-12)


def test_corridor_rows_are_exact_opposites_in_direction():
    """The two walls of one axis push along the same line, opposite ways --
    a sign slip here would make both rows fight the SAME direction."""
    j_row = np.array([1.0, 0.0, -0.5, 0.25, 0.0, 0.1])
    m_inv = np.eye(6)
    a_max, _b_max, a_min, _b_min = XTaskYZCorridorQPController._corridor_rows(
        j_row=j_row, m_inv=m_inv, bias=np.zeros(6), qd=np.zeros(6),
        value=0.0, lower=-1.0, upper=1.0, alpha1=1.0, alpha2=1.0,
    )
    assert np.array_equal(a_max, -a_min)


def test_corridor_row_slack_shrinks_as_the_wall_is_approached():
    """b_max must decrease monotonically as y -> y_max: the row gets tighter,
    which is the whole content of ``a1 a2 h``."""
    j_row = np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0])
    prev = None
    for y in (0.0, 0.01, 0.02, 0.03, 0.04, 0.049):
        _a, b, _a2, _b2 = XTaskYZCorridorQPController._corridor_rows(
            j_row=j_row, m_inv=np.eye(6), bias=np.zeros(6), qd=np.zeros(6),
            value=y, lower=-0.05, upper=0.05, alpha1=10.0, alpha2=10.0,
        )
        if prev is not None:
            assert b < prev
        prev = b


# --------------------------------------------------------------------------- #
# 2. Flags off == a plain box QP, exactly.
# --------------------------------------------------------------------------- #
def test_both_flags_off_builds_zero_inequality_rows_and_is_exactly_solve_box_qp(monkeypatch):
    calls = capture_qp(monkeypatch)
    ctrl = XTaskYZCorridorQPController(make_config())
    state = make_state()
    ctrl.reset_from_state(state)
    out = ctrl.compute(state)

    assert out.qp_num_ineq_rows == 0
    assert calls[-1]["a_ineq"] is None
    assert calls[-1]["b_ineq"] is None
    # Byte-identical, not approximately: with no rows,
    # solve_constrained_box_qp forwards straight to solve_box_qp.
    expected = solve_box_qp(
        calls[-1]["hessian"], calls[-1]["linear"], calls[-1]["lower"], calls[-1]["upper"]
    )
    assert np.array_equal(out.tau_preclip, expected)
    assert out.yz_corridor_active_rows == (False, False, False, False)
    assert out.manipulability_cbf_active is False
    assert out.manipulability is None


def test_unconstrained_minimizer_is_exactly_tau_des(monkeypatch):
    """The claim the whole design rests on: absent active rows and absent box
    saturation, the QP returns ``tau_des`` -- so Y/Z authority really is only
    ``tau_yz_soft`` plus the corridor rows.

    ``task_excluded_joints=()`` explicitly, because a pinned joint IS a form of
    box saturation: the claim above is scoped to "absent box saturation" and
    the pin's whole purpose is to violate it for the excluded index. The pinned
    counterpart of this claim -- every OTHER coordinate still solves the same
    QP, and the pinned one is exactly ``tau_hold`` -- is asserted in section 8.
    """
    calls = capture_qp(monkeypatch)
    cfg = make_config(
        kp_y=5.0, kd_y=2.0, kp_z=5.0, kd_z=2.0, tau_max_nm=np.full(6, 1.0e6),
        task_excluded_joints=(),
    )
    ctrl = XTaskYZCorridorQPController(cfg)
    state = make_state()
    ctrl.reset_from_state(state)
    out = ctrl.compute(state)

    tau_des = (
        out.tau_task_nominal + out.tau_damping + out.tau_posture + out.tau_yz_soft + out.tau_gravity
    )
    h, f = calls[-1]["hessian"], calls[-1]["linear"]
    assert np.allclose(-np.linalg.solve(h, f), tau_des, atol=1e-8)
    assert np.allclose(out.tau_preclip, tau_des, atol=1e-6)


# --------------------------------------------------------------------------- #
# 3. THE reduced-Jacobian claim, numerically, byte-identical.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "mutate",
    [
        pytest.param(lambda s: s.__setitem__("ee_pos", np.array([0.4, 3.7, -2.1])), id="ee_pos_yz"),
        pytest.param(lambda s: s.__setitem__("ee_lin_vel", np.array([0.0, -4.0, 9.0])), id="ee_vel_yz"),
        pytest.param(
            lambda s: s.__setitem__("target_ee_pos", np.array([0.45, 11.0, -6.0])), id="target_yz"
        ),
    ],
)
def test_yz_state_cannot_touch_the_hessian_or_the_task_torque(monkeypatch, mutate):
    """With the soft gains at zero, Y/Z is not in the task at ALL. Perturbing
    Y/Z position, Y/Z velocity, or the Y/Z target by absurd amounts must leave
    the QP Hessian and the nominal task torque BIT-for-bit unchanged.

    This is the test that separates "a genuinely reduced 4-row task" from
    "6-row OSC with two gains set to zero" -- the latter would still put
    ``J[1:3,:]`` into ``J^T W J`` (with a 1e-6 floor weight, as every
    ``max(kp, 1e-6)`` in this repo's QP controllers uses) and would fail here.
    """
    calls = capture_qp(monkeypatch)
    cfg = make_config()  # kp_y = kd_y = kp_z = kd_z = 0
    base_state = make_state(target_ee_pos=np.array([0.45, -0.2, 0.3]))

    ctrl_a = XTaskYZCorridorQPController(cfg)
    ctrl_a.reset_from_state(base_state)
    out_a = ctrl_a.compute(base_state)
    hess_a = calls[-1]["hessian"]

    mutated = dict(base_state)
    mutate(mutated)
    ctrl_b = XTaskYZCorridorQPController(cfg)
    ctrl_b.reset_from_state(base_state)  # same reference anchor
    out_b = ctrl_b.compute(mutated)
    hess_b = calls[-1]["hessian"]

    assert np.array_equal(hess_a, hess_b), "Y/Z leaked into the QP Hessian"
    assert np.array_equal(out_a.tau_task_nominal, out_b.tau_task_nominal), (
        "Y/Z leaked into the reduced task torque"
    )
    assert np.array_equal(out_a.wrench_reduced, out_b.wrench_reduced)


def test_the_hessian_is_literally_built_from_four_rows(monkeypatch):
    """Independent of the perturbation test above: reconstruct the Hessian
    from ``J[0:1,:]`` + ``J[3:6,:]`` and require an exact match, so the claim
    is checked directly rather than only through an invariance."""
    calls = capture_qp(monkeypatch)
    # task_excluded_joints=() explicitly: this test is about which ROWS build
    # the Hessian. Column exclusion is a separate mechanism, asserted in
    # section 7b (including that it zeroes exactly one column of this same
    # matrix), and leaving it on here would conflate the two claims.
    cfg = make_config(task_excluded_joints=())
    jac = make_jacobian()
    state = make_state(jac=jac)
    ctrl = XTaskYZCorridorQPController(cfg)
    ctrl.reset_from_state(state)
    ctrl.compute(state)

    j_reduced = np.vstack([jac[0:1, :], jac[3:6, :]])
    w = np.diag([cfg.kp_x, cfg.kp_rot, cfg.kp_rot, cfg.kp_rot])
    expected = 2.0 * (j_reduced.T @ w @ j_reduced + cfg.posture_regularization * np.eye(6))
    assert np.allclose(calls[-1]["hessian"], expected, atol=0.0, rtol=1e-15)


def test_soft_yz_gains_move_only_the_linear_term(monkeypatch):
    """Turning the soft gains up changes ``tau_yz_soft`` (and therefore the
    QP's linear term) but must never change the Hessian or the task torque."""
    calls = capture_qp(monkeypatch)
    state = make_state(ee_pos=(0.4, -0.25, 0.32))

    ctrl_a = XTaskYZCorridorQPController(make_config())
    ctrl_a.reset_from_state(make_state())
    out_a = ctrl_a.compute(state)
    hess_a, lin_a = calls[-1]["hessian"], calls[-1]["linear"]

    ctrl_b = XTaskYZCorridorQPController(make_config(kp_y=500.0, kp_z=500.0))
    ctrl_b.reset_from_state(make_state())
    out_b = ctrl_b.compute(state)
    hess_b, lin_b = calls[-1]["hessian"], calls[-1]["linear"]

    assert np.array_equal(hess_a, hess_b)
    assert np.array_equal(out_a.tau_task_nominal, out_b.tau_task_nominal)
    assert np.allclose(out_a.tau_yz_soft, 0.0)
    assert not np.allclose(out_b.tau_yz_soft, 0.0)
    assert not np.allclose(lin_a, lin_b)


# --------------------------------------------------------------------------- #
# 4. Corridor rows activate at the right wall, and nowhere else.
# --------------------------------------------------------------------------- #
def _wall_state(axis: int, sign: float, jac: np.ndarray) -> dict:
    """A state parked just inside one wall and moving toward it fast."""
    pos = [0.4, -0.2, 0.3]
    pos[axis] += sign * 0.0495  # corridor half-width is 0.05
    vel = [0.0, 0.0, 0.0]
    vel[axis] = sign * 0.5
    # qd chosen so J[axis,:] @ qd reproduces that Cartesian velocity.
    qd = np.linalg.solve(jac, np.array([0.0, vel[1], vel[2], 0.0, 0.0, 0.0]) if axis else
                         np.array([vel[0], 0.0, 0.0, 0.0, 0.0, 0.0]))
    return make_state(jac=jac, ee_pos=tuple(pos), ee_vel=tuple(vel), qd=qd)


@pytest.mark.parametrize(
    "axis,sign,row_index",
    [(1, +1.0, 0), (1, -1.0, 1), (2, +1.0, 2), (2, -1.0, 3)],
    ids=["y_max", "y_min", "z_max", "z_min"],
)
def test_each_corridor_row_activates_at_its_own_wall(axis, sign, row_index):
    jac = make_jacobian()
    cfg = make_config(yz_corridor_enabled=True)
    ctrl = XTaskYZCorridorQPController(cfg)
    ctrl.reset_from_state(make_state(jac=jac))  # corridor centered on the start pose
    out = ctrl.compute(_wall_state(axis, sign, jac))

    assert out.qp_num_ineq_rows == 4
    assert out.yz_corridor_active_rows[row_index] is True, (
        f"row {row_index} should be binding at this wall"
    )
    # The OPPOSITE wall of the same axis must not also fire -- moving toward
    # y_max cannot simultaneously threaten y_min.
    opposite = row_index + 1 if row_index % 2 == 0 else row_index - 1
    assert out.yz_corridor_active_rows[opposite] is False


def test_corridor_is_inert_in_the_middle_of_the_corridor():
    jac = make_jacobian()
    cfg = make_config(yz_corridor_enabled=True)
    ctrl = XTaskYZCorridorQPController(cfg)
    state = make_state(jac=jac)
    ctrl.reset_from_state(state)
    out = ctrl.compute(state)
    assert out.qp_num_ineq_rows == 4  # rows are built ...
    assert out.yz_corridor_active_rows == (False, False, False, False)  # ... but idle


def test_corridor_bounds_are_captured_at_reset_and_reported():
    cfg = make_config(
        yz_corridor_enabled=True, y_corridor_half_width_m=0.04, z_corridor_half_width_m=0.07
    )
    ctrl = XTaskYZCorridorQPController(cfg)
    state = make_state(ee_pos=(0.4, -0.2, 0.3))
    ctrl.reset_from_state(state)
    out = ctrl.compute(state)
    assert out.y_min == pytest.approx(-0.24)
    assert out.y_max == pytest.approx(-0.16)
    assert out.z_min == pytest.approx(0.23)
    assert out.z_max == pytest.approx(0.37)
    # Moving the end effector does NOT move the corridor.
    out2 = ctrl.compute(make_state(ee_pos=(0.6, -0.21, 0.31)))
    assert (out2.y_min, out2.y_max) == (out.y_min, out.y_max)


def test_active_corridor_row_actually_changes_the_torque():
    """A flag that says "binding" while the torque is unchanged would be
    worthless; check the QP really moved."""
    jac = make_jacobian()
    state = _wall_state(1, +1.0, jac)
    ref = make_state(jac=jac)

    off = XTaskYZCorridorQPController(make_config())
    off.reset_from_state(ref)
    tau_off = off.compute(state).tau

    on = XTaskYZCorridorQPController(make_config(yz_corridor_enabled=True))
    on.reset_from_state(ref)
    out_on = on.compute(state)

    assert any(out_on.yz_corridor_active_rows)
    assert not np.allclose(tau_off, out_on.tau)


def test_corridor_solution_satisfies_the_rows_it_was_given():
    """The point of a hard constraint: the returned torque must actually be
    inside the half-space, not merely nearer to it."""
    jac = make_jacobian()
    # task_excluded_joints=(): a pinned coordinate is an extra box constraint
    # the corridor row knows nothing about, so with the pin on, "the returned
    # torque satisfies the row" is a claim about the intersection of two
    # constraints rather than about the corridor row itself.
    cfg = make_config(
        yz_corridor_enabled=True, tau_max_nm=np.full(6, 400.0), task_excluded_joints=()
    )
    ctrl = XTaskYZCorridorQPController(cfg)
    ref = make_state(jac=jac)
    ctrl.reset_from_state(ref)
    state = _wall_state(1, +1.0, jac)
    out = ctrl.compute(state)

    m_inv = np.linalg.inv(state["mass_matrix"])
    a_max, b_max, _a, _b = XTaskYZCorridorQPController._corridor_rows(
        j_row=jac[1, :], m_inv=m_inv, bias=np.zeros(6), qd=state["qd"],
        value=float(state["ee_pos"][1]), lower=out.y_min, upper=out.y_max,
        alpha1=cfg.yz_corridor_alpha1, alpha2=cfg.yz_corridor_alpha2,
    )
    assert out.yz_corridor_feasible
    assert float(a_max[0] @ out.tau_preclip) <= b_max + 1e-6


# --------------------------------------------------------------------------- #
# 5. Infeasibility is REPORTED, never silently clamped.
# --------------------------------------------------------------------------- #
def test_infeasible_corridor_row_is_reported():
    """Squeeze the torque box until the corridor cannot be met and check the
    controller says so rather than quietly returning its best effort."""
    jac = make_jacobian()
    cfg = make_config(
        yz_corridor_enabled=True,
        y_corridor_half_width_m=1.0e-4,
        z_corridor_half_width_m=1.0e-4,
        yz_corridor_alpha1=500.0,
        yz_corridor_alpha2=500.0,
        tau_max_nm=np.full(6, 1.0e-3),
        torque_headroom=1.0,
    )
    ctrl = XTaskYZCorridorQPController(cfg)
    ref = make_state(jac=jac)
    ctrl.reset_from_state(ref)
    state = _wall_state(1, +1.0, jac)
    out = ctrl.compute(state)
    assert out.yz_corridor_feasible is False


def test_feasible_is_true_when_the_rows_are_reachable():
    jac = make_jacobian()
    ctrl = XTaskYZCorridorQPController(make_config(yz_corridor_enabled=True))
    ctrl.reset_from_state(make_state(jac=jac))
    out = ctrl.compute(make_state(jac=jac))
    assert out.yz_corridor_feasible is True
    assert out.manipulability_cbf_feasible is True


# --------------------------------------------------------------------------- #
# 6. Manipulability row composition.
# --------------------------------------------------------------------------- #
def analytic_jacobian(q: np.ndarray) -> np.ndarray:
    """``mu(q) = |sin(q[4])|`` in closed form -- the same fixture
    tests/unit/test_manipulability_cbf.py uses."""
    q = np.asarray(q, dtype=np.float64).reshape(6)
    return np.diag([1.0, 1.0, 1.0, 1.0, float(np.sin(q[4])), 1.0])


def test_manipulability_row_is_added_alongside_the_corridor_rows():
    cfg = make_config(
        yz_corridor_enabled=True, manipulability_cbf=True, manipulability_cbf_epsilon=0.2
    )
    ctrl = XTaskYZCorridorQPController(cfg, jacobian_fn=analytic_jacobian)
    q = np.array([0.0, 0.0, 0.0, 0.0, 0.5, 0.0])
    state = make_state(jac=analytic_jacobian(q))
    state["q"] = q
    ctrl.reset_from_state(state)
    out = ctrl.compute(state)
    assert out.qp_num_ineq_rows == 5
    assert out.manipulability == pytest.approx(abs(np.sin(0.5)), rel=1e-9)
    assert out.manipulability_cbf_h == pytest.approx(abs(np.sin(0.5)) - 0.2, rel=1e-9)


def test_manipulability_only_gives_exactly_one_row():
    cfg = make_config(manipulability_cbf=True)
    ctrl = XTaskYZCorridorQPController(cfg, jacobian_fn=analytic_jacobian)
    q = np.array([0.0, 0.0, 0.0, 0.0, 0.5, 0.0])
    state = make_state(jac=analytic_jacobian(q))
    state["q"] = q
    ctrl.reset_from_state(state)
    assert ctrl.compute(state).qp_num_ineq_rows == 1


def test_gradient_plateau_skips_the_row_rather_than_emitting_a_degenerate_one():
    """At a ``mu`` maximum ``grad_mu == 0``: a ``0 @ tau <= b`` row is either
    vacuous or unsatisfiable-by-construction, so it must be skipped (the same
    branch ``manipulability_cbf_filter`` has, reimplemented locally)."""
    cfg = make_config(manipulability_cbf=True)
    ctrl = XTaskYZCorridorQPController(cfg, jacobian_fn=analytic_jacobian)
    q = np.array([0.0, 0.0, 0.0, 0.0, np.pi / 2.0, 0.0])  # d|sin|/dq = 0 here
    state = make_state(jac=analytic_jacobian(q))
    state["q"] = q
    ctrl.reset_from_state(state)
    out = ctrl.compute(state)
    assert out.qp_num_ineq_rows == 0
    assert out.manipulability == pytest.approx(1.0, rel=1e-9)  # still REPORTED
    assert out.manipulability_cbf_active is False


# --------------------------------------------------------------------------- #
# 7. Loud failures, not silent no-ops.
# --------------------------------------------------------------------------- #
def test_manipulability_cbf_without_jacobian_fn_raises():
    ctrl = XTaskYZCorridorQPController(make_config(manipulability_cbf=True))
    state = make_state()
    ctrl.reset_from_state(state)
    with pytest.raises(ValueError, match="jacobian_fn"):
        ctrl.compute(state)


def test_constraint_rows_without_a_mass_matrix_raise():
    ctrl = XTaskYZCorridorQPController(make_config(yz_corridor_enabled=True))
    state = make_state(mass_matrix=False)
    ctrl.reset_from_state(state)
    with pytest.raises(ValueError, match="mass_matrix"):
        ctrl.compute(state)


def test_off_x_transport_axis_raises_rather_than_silently_moving_x():
    ctrl = XTaskYZCorridorQPController(make_config())
    state = make_state()
    ctrl.reset_from_state(state)
    state["transport_axis_index"] = 1
    with pytest.raises(ValueError, match="world-X only"):
        ctrl.compute(state)


def test_compute_before_reset_raises():
    ctrl = XTaskYZCorridorQPController(make_config())
    with pytest.raises(RuntimeError, match="reset_from_state"):
        ctrl.compute(make_state())


def test_torque_never_exceeds_the_hard_limit():
    cfg = make_config(kp_x=1.0e6, tau_max_nm=np.array([150.0, 150.0, 150.0, 28.0, 28.0, 28.0]))
    ctrl = XTaskYZCorridorQPController(cfg)
    state = make_state(target_x=50.0)
    ctrl.reset_from_state(state)
    out = ctrl.compute(state)
    assert np.all(np.abs(out.tau) <= np.asarray(cfg.tau_max_nm) + 1e-9)


# --------------------------------------------------------------------------- #
# 7b. task_excluded_joints: the structural no-task-torque guarantee.
#
# The bug this fixes was found in closed loop (shoulder_pan swinging 4.3-13.2
# deg during ordinary X transport at ARM_Q0, where it is pinned for real
# wall/base clearance). What is testable HERE, and is the whole reason the
# mechanism is a box pin rather than a Jacobian-column zeroing, is that the
# guarantee is EXACT and unconditional: not "the task torque is removed", but
# "the commanded torque equals tau_hold, bit for bit, whatever else the QP is
# doing". The closed-loop consequence is asserted in
# tests/mujoco/test_x_task_yz_corridor_qp_closed_loop.py.
# --------------------------------------------------------------------------- #
def test_excluded_joint_default_is_shoulder_pan_and_is_reported():
    ctrl = XTaskYZCorridorQPController(XTaskYZCorridorQPConfig())
    state = make_state()
    ctrl.reset_from_state(state)
    out = ctrl.compute(state)
    assert XTaskYZCorridorQPConfig().task_excluded_joints == (0,)
    assert out.task_excluded_joints == (0,)


def test_tau_hold_is_exactly_the_non_task_bias():
    ctrl = XTaskYZCorridorQPController(make_config())
    state = make_state(qd=[0.3, -0.2, 0.1, 0.4, -0.5, 0.05])
    ctrl.reset_from_state(state)
    out = ctrl.compute(state)
    assert np.array_equal(
        out.tau_hold, out.tau_damping + out.tau_posture + out.tau_gravity
    )


@pytest.mark.parametrize(
    "extra",
    [
        pytest.param({}, id="no_rows"),
        pytest.param({"yz_corridor_enabled": True}, id="corridor_rows"),
        # A task the arm cannot possibly satisfy: the QP is pushed hard against
        # both the torque box and the corridor rows at once, which is exactly
        # the regime where a merely-preferential mechanism leaks.
        pytest.param(
            {"yz_corridor_enabled": True, "kp_x": 1.0e6, "y_corridor_half_width_m": 1.0e-4},
            id="saturated_and_walled",
        ),
    ],
)
def test_excluded_joint_torque_is_bit_exactly_the_hold_torque(extra):
    cfg = make_config(task_excluded_joints=(0,), **extra)
    ctrl = XTaskYZCorridorQPController(cfg)
    ref = XTaskYZCorridorQPController(make_config(task_excluded_joints=(), **extra))
    rng = np.random.default_rng(11)
    state = make_state(target_x=9.0)
    ctrl.reset_from_state(state)
    ref.reset_from_state(state)
    for _ in range(25):
        state["qd"] = rng.normal(0.0, 0.8, 6)
        state["ee_lin_vel"] = rng.normal(0.0, 0.3, 3)
        state["ee_ang_vel"] = rng.normal(0.0, 0.3, 3)
        state["ee_pos"] = np.array([0.4, -0.2, 0.3]) + rng.normal(0.0, 0.02, 3)
        out = ctrl.compute(state)
        # Bit-exact, not approximate: np.clip with lo == hi returns that value.
        assert out.tau_preclip[0] == out.tau_hold[0]
        assert out.tau[0] == np.clip(out.tau_hold[0], -150.0, 150.0)
        # And an unexcluded controller really would have driven it here --
        # otherwise the assertions above are vacuous.
        assert abs(ref.compute(state).tau_task_nominal[0]) > 1.0


def test_excluded_joint_gets_no_task_torque_at_all(monkeypatch):
    """Part (a) of the mechanism: the excluded column of ``J_reduced`` is
    zeroed, so ``tau_task_nominal = J_reduced.T @ wrench`` has an exact zero
    there -- and the Hessian is left DIAGONAL in that coordinate."""
    calls = capture_qp(monkeypatch)
    ctrl = XTaskYZCorridorQPController(make_config(task_excluded_joints=(0,)))
    state = make_state()
    ctrl.reset_from_state(state)
    out = ctrl.compute(state)

    assert out.tau_task_nominal[0] == 0.0
    h = calls[-1]["hessian"]  # captured before the reference run below
    free = [1, 2, 3, 4, 5]
    assert np.all(h[0, free] == 0.0)
    assert np.all(h[free, 0] == 0.0)
    assert h[0, 0] > 0.0  # the Tikhonov term still keeps it invertible

    # ... and the same task, unexcluded, really did want to drive it (else the
    # assertions above prove nothing about this pose/Jacobian).
    ref = XTaskYZCorridorQPController(make_config(task_excluded_joints=()))
    ref.reset_from_state(state)
    assert abs(ref.compute(state).tau_task_nominal[0]) > 1.0
    assert np.any(calls[-1]["hessian"][0, free] != 0.0)


def test_the_pin_costs_the_free_joints_nothing(monkeypatch):
    """The reason (a) and (b) are used TOGETHER. Because (a) leaves ``H``
    decoupled in the pinned coordinate, clamping that coordinate must not
    perturb the other five at all -- if it did, the QP would be re-optimizing
    them to make up a lost force projection, which is the pin-only failure
    mode that diverged in closed loop (see the config docstring)."""
    calls = capture_qp(monkeypatch)
    state = make_state()
    ctrl = XTaskYZCorridorQPController(
        make_config(
            task_excluded_joints=(0,), tau_max_nm=np.full(6, 1.0e6),
            # A non-zero soft Y/Z bias is what makes tau_des[0] differ from
            # tau_hold[0] at all once the column is zeroed -- without it the
            # pin would be trivially non-binding and this test vacuous.
            kp_y=50.0, kd_y=10.0, kp_z=50.0, kd_z=10.0,
        )
    )
    ctrl.reset_from_state(state)
    # Displace Y/Z away from the corridor centre captured at reset, so the soft
    # bias tau_yz_soft is genuinely non-zero at this joint.
    state["ee_pos"] = np.array([0.42, -0.17, 0.34])
    out = ctrl.compute(state)

    h, f = calls[-1]["hessian"], calls[-1]["linear"]
    lo, hi = calls[-1]["lower"].copy(), calls[-1]["upper"].copy()
    pin = float(out.tau_hold[0])
    free = [1, 2, 3, 4, 5]
    assert lo[0] == hi[0] == pin
    assert abs(out.tau_yz_soft[0]) > 1e-6

    # Same QP, but with index 0 left completely free.
    lo[0], hi[0] = -1.0e9, 1.0e9
    unpinned = solve_box_qp(h, f, lo, hi)
    assert np.allclose(out.tau_preclip[free], unpinned[free], atol=1e-9)
    # The pin really did bind (otherwise the comparison above is vacuous).
    assert abs(unpinned[0] - pin) > 1e-3


def test_empty_exclusion_reproduces_the_unpinned_solve_byte_identically(monkeypatch):
    calls = capture_qp(monkeypatch)
    state = make_state()
    ctrl = XTaskYZCorridorQPController(make_config(task_excluded_joints=()))
    ctrl.reset_from_state(state)
    out = ctrl.compute(state)
    assert out.task_excluded_joints == ()
    expected = solve_box_qp(
        calls[-1]["hessian"], calls[-1]["linear"], calls[-1]["lower"], calls[-1]["upper"]
    )
    assert np.array_equal(out.tau_preclip, expected)


def test_a_pin_can_never_widen_the_torque_box(monkeypatch):
    """The pin is applied LAST, but it is clipped into the box that was in
    force -- a hold torque larger than the headroom limit must be clamped, not
    granted. Without this, an excluded joint would be the one joint in the
    controller that could exceed its own torque limit."""
    calls = capture_qp(monkeypatch)
    # kp_posture huge => tau_hold[0] far outside the +-1 Nm headroom box.
    cfg = make_config(
        task_excluded_joints=(0,), kp_posture=1.0e4, tau_max_nm=np.full(6, 1.0),
        torque_headroom=1.0,
    )
    ctrl = XTaskYZCorridorQPController(cfg)
    state = make_state()
    ctrl.reset_from_state(state)
    state["q"] = np.asarray(state["q"], dtype=np.float64) + 0.5
    out = ctrl.compute(state)
    assert abs(out.tau_hold[0]) > 1.0
    assert calls[-1]["lower"][0] == calls[-1]["upper"][0]
    assert abs(calls[-1]["lower"][0]) == pytest.approx(1.0)
    assert abs(out.tau[0]) <= 1.0 + 1e-12


def test_two_joint_exclusion_pins_both():
    cfg = make_config(task_excluded_joints=(0, 5))
    ctrl = XTaskYZCorridorQPController(cfg)
    state = make_state(qd=[0.2, 0.1, -0.3, 0.05, -0.1, 0.4])
    ctrl.reset_from_state(state)
    out = ctrl.compute(state)
    assert out.tau_preclip[0] == out.tau_hold[0]
    assert out.tau_preclip[5] == out.tau_hold[5]
    assert out.task_excluded_joints == (0, 5)


@pytest.mark.parametrize(
    "bad",
    [
        pytest.param([6], id="out_of_range_high"),
        pytest.param([-1], id="negative_reads_as_wrist_3"),
        pytest.param([0, 0], id="duplicate"),
        pytest.param([0, 1, 2], id="more_than_two_underdetermines_the_task"),
        pytest.param(["pan"], id="not_an_int"),
    ],
)
def test_bad_task_excluded_joints_raises(bad):
    with pytest.raises(ValueError, match="task_excluded_joints"):
        _parse_task_excluded_joints(bad)


def test_task_excluded_joints_parser_normalizes():
    assert _parse_task_excluded_joints(None) is None       # "use the default"
    assert _parse_task_excluded_joints([]) == ()           # "exclude nothing"
    assert _parse_task_excluded_joints(0) == (0,)          # bare int
    assert _parse_task_excluded_joints([5, 0]) == (0, 5)   # sorted


def test_yaml_omitting_task_excluded_joints_keeps_the_default_rather_than_disabling():
    cfg = XTaskYZCorridorQPConfig.from_controller_yaml_section(_yaml_section())
    assert cfg.task_excluded_joints == (0,)
    cfg_off = XTaskYZCorridorQPConfig.from_controller_yaml_section(
        _yaml_section(task_excluded_joints=[])
    )
    assert cfg_off.task_excluded_joints == ()
    cfg_two = XTaskYZCorridorQPConfig.from_controller_yaml_section(
        _yaml_section(task_excluded_joints=[4, 0])
    )
    assert cfg_two.task_excluded_joints == (0, 4)


# --------------------------------------------------------------------------- #
# 8. Config: defaults, YAML round-trip, and loud rejection of bad values.
# --------------------------------------------------------------------------- #
def test_defaults_are_off_with_the_documented_values():
    cfg = XTaskYZCorridorQPConfig()
    assert cfg.task_excluded_joints == (0,)
    assert cfg.yz_corridor_enabled is False
    assert cfg.manipulability_cbf is False
    assert cfg.y_corridor_half_width_m == 0.05
    assert cfg.z_corridor_half_width_m == 0.05
    assert cfg.yz_corridor_alpha1 == 10.0
    assert cfg.yz_corridor_alpha2 == 10.0
    assert cfg.dual_sweeps == 4
    assert cfg.dual_root_iters == 10
    # The redefined soft-centering defaults, NOT the inherited hard-task ones.
    assert (cfg.kp_y, cfg.kd_y, cfg.kp_z, cfg.kd_z) == (5.0, 2.0, 5.0, 2.0)


def _yaml_section(**extra) -> dict:
    section = {
        "torque_limits_mode": "initial",
        "torque_limits_initial": {
            "shoulder_pan_joint": 150.0, "shoulder_lift_joint": 150.0, "elbow_joint": 150.0,
            "wrist_1_joint": 28.0, "wrist_2_joint": 28.0, "wrist_3_joint": 28.0,
        },
        "gains": {"kp_x": 2400.0, "kd_x": 240.0},
    }
    section.update(extra)
    return section


def test_yaml_roundtrip_reads_every_new_field():
    cfg = XTaskYZCorridorQPConfig.from_controller_yaml_section(
        _yaml_section(
            yz_corridor_enabled=True,
            y_corridor_half_width_m=0.07,
            z_corridor_half_width_m=0.09,
            yz_corridor_alpha1=4.0,
            yz_corridor_alpha2=6.0,
            manipulability_cbf=True,
            manipulability_cbf_epsilon=3.0e-4,
            manipulability_cbf_alpha1=8.0,
            manipulability_cbf_alpha2=12.0,
            manipulability_cbf_fd_step=2.0e-5,
            manipulability_cbf_curvature_step=5.0e-4,
            dual_sweeps=7,
            dual_root_iters=13,
        )
    )
    assert cfg.yz_corridor_enabled is True
    assert cfg.y_corridor_half_width_m == 0.07
    assert cfg.z_corridor_half_width_m == 0.09
    assert (cfg.yz_corridor_alpha1, cfg.yz_corridor_alpha2) == (4.0, 6.0)
    assert cfg.manipulability_cbf is True
    assert cfg.manipulability_cbf_epsilon == 3.0e-4
    assert (cfg.manipulability_cbf_alpha1, cfg.manipulability_cbf_alpha2) == (8.0, 12.0)
    assert cfg.manipulability_cbf_fd_step == 2.0e-5
    assert cfg.manipulability_cbf_curvature_step == 5.0e-4
    assert (cfg.dual_sweeps, cfg.dual_root_iters) == (7, 13)
    assert cfg.kp_x == 2400.0 and cfg.kd_x == 240.0


def test_yaml_default_is_both_flags_off_and_soft_yz_gains():
    """A YAML with no Y/Z gains must get 5.0/2.0, NOT the inherited 80/15 and
    120/20 -- the parser has to read `gains` itself rather than trust the
    TorqueTaskQPConfig base, which applies the hard-task defaults."""
    cfg = XTaskYZCorridorQPConfig.from_controller_yaml_section(_yaml_section())
    assert cfg.yz_corridor_enabled is False
    assert cfg.manipulability_cbf is False
    assert (cfg.kp_y, cfg.kd_y, cfg.kp_z, cfg.kd_z) == (5.0, 2.0, 5.0, 2.0)


def test_yaml_explicit_yz_gains_are_honored():
    cfg = XTaskYZCorridorQPConfig.from_controller_yaml_section(
        _yaml_section(gains={"kp_x": 2400.0, "kp_y": 33.0, "kd_z": 7.0})
    )
    assert cfg.kp_y == 33.0
    assert cfg.kd_z == 7.0
    assert cfg.kd_y == 2.0  # unspecified -> soft default, not 15.0


@pytest.mark.parametrize("bad", [0.0, -0.05, float("nan"), float("inf"), "wide"])
@pytest.mark.parametrize("field", ["y_corridor_half_width_m", "z_corridor_half_width_m"])
def test_bad_corridor_half_width_raises(field, bad):
    with pytest.raises(ValueError, match=field):
        XTaskYZCorridorQPConfig.from_controller_yaml_section(_yaml_section(**{field: bad}))


@pytest.mark.parametrize("bad", [0.0, -1.0, float("nan"), float("inf")])
@pytest.mark.parametrize("field", ["yz_corridor_alpha1", "yz_corridor_alpha2"])
def test_bad_corridor_alpha_raises(field, bad):
    with pytest.raises(ValueError, match=field):
        XTaskYZCorridorQPConfig.from_controller_yaml_section(_yaml_section(**{field: bad}))


@pytest.mark.parametrize("bad", [0.0, -1.0e-3, float("nan")])
def test_bad_manipulability_epsilon_still_raises_through_this_config(bad):
    with pytest.raises(ValueError, match="manipulability_cbf_epsilon"):
        XTaskYZCorridorQPConfig.from_controller_yaml_section(
            _yaml_section(manipulability_cbf_epsilon=bad)
        )


def test_corridor_half_width_parser_accepts_a_plain_positive_float():
    assert _parse_corridor_half_width(0.05, "y_corridor_half_width_m") == 0.05
    assert _parse_corridor_half_width("0.02", "y_corridor_half_width_m") == 0.02


def test_shipped_configs_parse_and_agree_where_they_should():
    """The two shipped YAMLs must differ ONLY in the two flags and the drift
    guards -- if a gain drifts apart, the closed-loop comparison silently
    stops being a comparison of the mechanisms."""
    import yaml as _yaml

    def load(name):
        with open(REPO_ROOT / "config" / name, "r", encoding="utf-8") as fh:
            return _yaml.safe_load(fh)

    off = load("ur5e_mujoco_torque_x_task_yz_corridor_qp.yaml")
    on = load("ur5e_mujoco_torque_x_task_yz_corridor_qp_enabled.yaml")
    cfg_off = XTaskYZCorridorQPConfig.from_controller_yaml_section(off["controller"])
    cfg_on = XTaskYZCorridorQPConfig.from_controller_yaml_section(on["controller"])

    assert cfg_off.yz_corridor_enabled is False and cfg_off.manipulability_cbf is False
    assert cfg_on.yz_corridor_enabled is True and cfg_on.manipulability_cbf is True
    for field in ("kp_x", "kd_x", "kp_y", "kd_y", "kp_z", "kd_z", "kp_rot", "kd_rot",
                  "kp_posture", "kd_posture", "kd_joint", "posture_regularization",
                  "task_excluded_joints",
                  "torque_headroom", "jacobian_singular_cond_max",
                  "y_corridor_half_width_m", "z_corridor_half_width_m",
                  "yz_corridor_alpha1", "yz_corridor_alpha2",
                  "manipulability_cbf_epsilon", "dual_sweeps", "dual_root_iters"):
        assert getattr(cfg_off, field) == getattr(cfg_on, field), field
    # ... and the enabled file really does carry the widened guards, on the
    # field the MuJoCo path actually enforces.
    assert on["controller"]["safety"]["max_abs_orthogonal_drift_m"] == 0.06
    assert off["controller"]["safety"]["max_abs_orthogonal_drift_m"] == 0.03
    # The corridor must be strictly INSIDE the guard, or the guard decides the
    # outcome and the mechanism can never be observed.
    assert cfg_on.y_corridor_half_width_m < on["controller"]["safety"]["max_abs_orthogonal_drift_m"]


# --------------------------------------------------------------------------- #
# 9. Diagnostics that a trace has to be able to rely on.
# --------------------------------------------------------------------------- #
def test_qp_solve_time_is_recorded_and_positive():
    ctrl = XTaskYZCorridorQPController(make_config(yz_corridor_enabled=True))
    state = make_state()
    ctrl.reset_from_state(state)
    out = ctrl.compute(state)
    assert out.qp_solve_time_s > 0.0
    assert np.isfinite(out.qp_solve_time_s)


def test_output_is_plain_vars_serializable():
    """``MujocoUR5eTorqueAdapter._controller_step`` uses ``dict(vars(output))``
    when there is no ``as_dict``; make sure that actually works here."""
    ctrl = XTaskYZCorridorQPController(make_config())
    state = make_state()
    ctrl.reset_from_state(state)
    d = dict(vars(ctrl.compute(state)))
    for key in ("tau", "wrench_reduced", "y_min", "y_max", "yz_corridor_active_rows",
                "qp_num_ineq_rows", "qp_solve_time_s"):
        assert key in d


def test_wrench_reduced_has_exactly_four_entries():
    ctrl = XTaskYZCorridorQPController(make_config())
    state = make_state()
    ctrl.reset_from_state(state)
    out = ctrl.compute(state)
    assert out.wrench_reduced.shape == (len(make_config().task_axis_rows) + 3,)
    assert out.wrench_reduced.shape == (4,)  # ... which is 4 for the X-only default
    assert out.wrench_reduced[0] == pytest.approx(
        400.0 * (state["target_x"] - state["ee_pos"][0])
    )


# --------------------------------------------------------------------------- #
# 10. Configurable task/corridor axis rows (2026-08-13).
#
# The single most important test in this section is the FIRST one: the
# generalization from hardcoded `vstack([J[0:1,:], J[3:6,:]])` to configurable
# row sets must reproduce the X-only behavior BIT-IDENTICALLY, or every
# closed-loop number validated for the X-only config silently stops applying.
# --------------------------------------------------------------------------- #
def _reference_x_only_torque(cfg, state, jac, p0, quat0, q_rest):
    """The pre-generalization formulas, written out literally.

    Not a re-expression of the new code in different variables -- these are the
    exact expressions the controller contained before ``task_axis_rows``
    existed, so agreement is evidence about the refactor rather than about the
    test's own arithmetic.
    """
    from controller_core.kinematics_utils import orientation_error_vec_wxyz
    from controller_core.torque_task_qp import _velocity_implied_torque_bounds

    p = np.asarray(state["ee_pos"], dtype=np.float64)
    v = np.asarray(state["ee_lin_vel"], dtype=np.float64)
    omega = np.asarray(state["ee_ang_vel"], dtype=np.float64)
    q = np.asarray(state["q"], dtype=np.float64)
    qd = np.asarray(state["qd"], dtype=np.float64)
    j_reduced = np.vstack([jac[0:1, :], jac[3:6, :]])
    x_des = float(state["target_x"])
    x_vel_des = float(state.get("target_x_vel", 0.0))
    fx = cfg.kp_x * (x_des - p[0]) + cfg.kd_x * (x_vel_des - v[0])
    e_rot = orientation_error_vec_wxyz(quat0, np.asarray(state["ee_quat"], dtype=np.float64))
    m_rot = cfg.kp_rot * e_rot - cfg.kd_rot * omega
    wrench = np.array([fx, m_rot[0], m_rot[1], m_rot[2]], dtype=np.float64)

    task_weights = np.diag(
        [max(cfg.kp_x, 1.0e-6)] + [max(cfg.kp_rot, 1.0e-6)] * 3
    ).astype(np.float64)
    lam_reg = float(max(cfg.posture_regularization, 1.0e-6))
    hessian = 2.0 * (j_reduced.T @ task_weights @ j_reduced + lam_reg * np.eye(6))
    tau_task = j_reduced.T @ wrench

    fy_soft = cfg.kp_y * (p0[1] - p[1]) - cfg.kd_y * v[1]
    fz_soft = cfg.kp_z * (p0[2] - p[2]) - cfg.kd_z * v[2]
    tau_yz_soft = jac[1, :] * fy_soft + jac[2, :] * fz_soft

    tau_damping = -cfg.kd_joint * qd
    tau_posture = cfg.kp_posture * (q_rest - q) - cfg.kd_posture * qd
    gravity = np.zeros(6)
    tau_des = tau_task + tau_damping + tau_posture + tau_yz_soft + gravity
    return hessian, tau_task, tau_yz_soft, tau_des


def test_default_row_sets_reproduce_the_x_only_controller_bit_identically(monkeypatch):
    """THE refactor guard. ``task_axis_rows=(0,)`` / ``corridor_axis_rows=(1,2)``
    must be the old hardcoded controller to the last bit -- Hessian, task
    torque, soft bias and the QP's own answer."""
    calls = capture_qp(monkeypatch)
    jac = make_jacobian()
    cfg = make_config(
        kp_y=5.0, kd_y=2.0, kp_z=7.0, kd_z=3.0, task_excluded_joints=(),
    )
    assert (cfg.task_axis_rows, cfg.corridor_axis_rows) == ((0,), (1, 2))
    state = make_state(jac=jac, ee_vel=(0.11, -0.07, 0.05), qd=[0.2, -0.1, 0.3, 0.05, -0.2, 0.1])
    ctrl = XTaskYZCorridorQPController(cfg)
    ctrl.reset_from_state(state)
    ref_p0 = np.asarray(state["ee_pos"], dtype=np.float64).copy()
    ref_quat0 = np.asarray(state["ee_quat"], dtype=np.float64).copy()
    ref_q_rest = np.asarray(state["q"], dtype=np.float64).copy()
    # Move off the captured reference so every term is genuinely non-zero.
    state["ee_pos"] = np.array([0.43, -0.17, 0.34])
    out = ctrl.compute(state)

    ref_h, ref_task, ref_soft, ref_des = _reference_x_only_torque(
        cfg, state, jac, p0=ref_p0, quat0=ref_quat0, q_rest=ref_q_rest
    )
    assert np.array_equal(calls[-1]["hessian"], ref_h)
    assert np.array_equal(out.tau_task_nominal, ref_task)
    assert np.array_equal(out.tau_yz_soft, ref_soft)
    assert out.wrench_reduced.shape == (4,)
    # The QP's own answer, byte-identical to solving the reference problem.
    expected = solve_box_qp(
        ref_h, -ref_h @ ref_des, calls[-1]["lower"], calls[-1]["upper"]
    )
    assert np.array_equal(out.tau_preclip, expected)
    assert out.task_axis_rows == (0,) and out.corridor_axis_rows == (1, 2)


def test_x_plus_z_builds_a_five_row_task_and_only_two_corridor_rows(monkeypatch):
    calls = capture_qp(monkeypatch)
    jac = make_jacobian()
    cfg = make_config(
        task_axis_rows=(0, 2), corridor_axis_rows=(1,),
        kp_z=390.6, kd_z=65.1, yz_corridor_enabled=True, task_excluded_joints=(),
    )
    ctrl = XTaskYZCorridorQPController(cfg)
    state = make_state(jac=jac)
    ctrl.reset_from_state(state)
    out = ctrl.compute(state)

    # Five task rows: world X, world Z, and the three orientation rows.
    assert out.wrench_reduced.shape == (5,)
    expected_j = np.vstack([jac[[0, 2], :], jac[3:6, :]])
    w = np.diag([cfg.kp_x, cfg.kp_z] + [max(cfg.kp_rot, 1e-6)] * 3)
    assert np.allclose(
        calls[-1]["hessian"],
        2.0 * (expected_j.T @ w @ expected_j + cfg.posture_regularization * np.eye(6)),
        atol=0.0, rtol=1e-15,
    )
    # Two corridor rows (Y only), not four -- Z's disappear when it is tracked.
    assert out.qp_num_ineq_rows == 2
    assert out.task_axis_rows == (0, 2) and out.corridor_axis_rows == (1,)


def test_a_tracked_axis_gets_no_soft_bias(monkeypatch):
    """The soft centering bias is for corridor axes only -- an axis promoted to
    a task row would otherwise be double-counted, and with a gain deliberately
    sized to be negligible next to a task gain."""
    jac = make_jacobian()
    common = dict(kp_y=5.0, kd_y=2.0, kp_z=7.0, kd_z=3.0, task_excluded_joints=())
    state = make_state(jac=jac, ee_vel=(0.0, 0.1, 0.2))
    xz = XTaskYZCorridorQPController(
        make_config(task_axis_rows=(0, 2), corridor_axis_rows=(1,), **common)
    )
    xz.reset_from_state(state)
    state["ee_pos"] = np.array([0.43, -0.17, 0.34])
    out_xz = xz.compute(state)

    fy = common["kp_y"] * (-0.2 - (-0.17)) - common["kd_y"] * 0.1
    assert np.allclose(out_xz.tau_yz_soft, jac[1, :] * fy, atol=0.0, rtol=1e-15)
    # ... and Z's error is still REPORTED even though the bias no longer uses it.
    assert out_xz.z_error == pytest.approx(0.3 - 0.34)


def test_corridor_active_rows_keep_their_slots_when_an_axis_is_tracked():
    """``yz_corridor_active_rows`` is (y_max, y_min, z_max, z_min) whatever the
    row sets are, so a trace stays readable; a tracked axis reports False."""
    jac = make_jacobian()
    cfg = make_config(
        task_axis_rows=(0, 2), corridor_axis_rows=(1,), yz_corridor_enabled=True,
        tau_max_nm=np.full(6, 400.0), task_excluded_joints=(),
    )
    ctrl = XTaskYZCorridorQPController(cfg)
    ref = make_state(jac=jac)
    ctrl.reset_from_state(ref)
    out = ctrl.compute(_wall_state(1, +1.0, jac))
    assert out.yz_corridor_active_rows[2] is False  # z_max slot, Z is tracked
    assert out.yz_corridor_active_rows[3] is False  # z_min slot
    assert any(out.yz_corridor_active_rows[:2])     # a Y wall did fire


def test_empty_corridor_axis_rows_gives_no_corridor_rows_at_all():
    cfg = make_config(
        task_axis_rows=(0, 2), corridor_axis_rows=(), yz_corridor_enabled=True,
        task_excluded_joints=(),
    )
    ctrl = XTaskYZCorridorQPController(cfg)
    state = make_state()
    ctrl.reset_from_state(state)
    out = ctrl.compute(state)
    assert out.qp_num_ineq_rows == 0
    assert out.yz_corridor_active_rows == (False, False, False, False)
    assert np.array_equal(out.tau_yz_soft, np.zeros(6))


@pytest.mark.parametrize(
    "task_rows,corridor_rows",
    [
        pytest.param([0, 1], [1, 2], id="overlap"),
        pytest.param([1], [2], id="transport_axis_not_tracked"),
        pytest.param([0, 3], None, id="index_out_of_range"),
        pytest.param([0, 0], None, id="duplicate"),
        pytest.param([], None, id="empty_task"),
        pytest.param([0, 1, 2], [0], id="x_in_corridor"),
    ],
)
def test_bad_axis_row_sets_raise(task_rows, corridor_rows):
    with pytest.raises(ValueError):
        _parse_axis_row_sets(task_rows, corridor_rows)


def test_yaml_roundtrip_reads_the_axis_row_sets():
    cfg = XTaskYZCorridorQPConfig.from_controller_yaml_section(_yaml_section())
    assert (cfg.task_axis_rows, cfg.corridor_axis_rows) == ((0,), (1, 2))
    cfg2 = XTaskYZCorridorQPConfig.from_controller_yaml_section(
        _yaml_section(task_axis_rows=[2, 0], corridor_axis_rows=[1])
    )
    assert (cfg2.task_axis_rows, cfg2.corridor_axis_rows) == ((0, 2), (1,))


def test_shipped_xz_config_parses_and_differs_only_in_the_row_sets_and_gains():
    import yaml as _yaml

    def load(name):
        with open(REPO_ROOT / "config" / name, "r", encoding="utf-8") as fh:
            return _yaml.safe_load(fh)

    x_only = XTaskYZCorridorQPConfig.from_controller_yaml_section(
        load("ur5e_mujoco_torque_x_task_yz_corridor_qp_enabled.yaml")["controller"]
    )
    xz = XTaskYZCorridorQPConfig.from_controller_yaml_section(
        load("ur5e_mujoco_torque_x_z_task_y_corridor_qp_enabled.yaml")["controller"]
    )
    assert (xz.task_axis_rows, xz.corridor_axis_rows) == ((0, 2), (1,))
    assert (x_only.task_axis_rows, x_only.corridor_axis_rows) == ((0,), (1, 2))
    # Same mechanisms, same guards, same joint exclusion -- only the row sets
    # and the gains that had to be re-derived for them may differ.
    for field in ("yz_corridor_enabled", "manipulability_cbf", "task_excluded_joints",
                  "y_corridor_half_width_m", "z_corridor_half_width_m",
                  "yz_corridor_alpha1", "yz_corridor_alpha2",
                  "manipulability_cbf_epsilon", "torque_headroom",
                  "posture_regularization", "kp_posture", "kd_posture", "kd_joint",
                  "kp_y", "kd_y", "dual_sweeps", "dual_root_iters"):
        assert getattr(x_only, field) == getattr(xz, field), field
    # Z went from a soft bias to a real task gain -- by ~78x, not by a nudge.
    assert x_only.kp_z == 5.0 and xz.kp_z > 300.0


# --------------------------------------------------------------------------- #
# 12. Tool-face task frame (2026-08-14).
#
# `task_frame: "tool"` pre-rotates the Jacobian's three POSITION rows by
# R_tool^T before row selection, so the row indices mean tool X/Y/Z. These
# lock down (a) that the world default is untouched, (b) that the rotation is
# the one actually claimed, and (c) that the three update modes are distinct
# mechanisms rather than aliases.
# --------------------------------------------------------------------------- #
from controller_core.kinematics_utils import quat_to_rotmat  # noqa: E402
from controller_core.x_task_yz_corridor_qp.parsing import (  # noqa: E402
    TASK_FRAME_UPDATES,
    TASK_FRAMES,
    _parse_task_frame,
)


def _quat_about_z(angle_rad: float) -> np.ndarray:
    return np.array([np.cos(angle_rad / 2.0), 0.0, 0.0, np.sin(angle_rad / 2.0)])


@pytest.mark.parametrize("bad", ["", "tcp", "base", "world_frame", 3])
def test_bad_task_frame_raises(bad):
    with pytest.raises(ValueError):
        _parse_task_frame(bad, None)


@pytest.mark.parametrize("bad", ["", "always", "static", 7])
def test_bad_task_frame_update_raises(bad):
    with pytest.raises(ValueError):
        _parse_task_frame("tool", bad)


def test_task_frame_update_without_tool_frame_is_rejected_not_ignored():
    """A config that sets an update mode in the world frame has written
    something with provably no effect -- that is a mistake, not a default."""
    with pytest.raises(ValueError, match="provably has no effect"):
        _parse_task_frame("world", "live")
    with pytest.raises(ValueError, match="provably has no effect"):
        _parse_task_frame(None, "frozen")


def test_task_frame_parser_normalizes_and_defaults():
    assert _parse_task_frame(None, None) == (None, None)
    assert _parse_task_frame("  TOOL ", " Live ") == ("tool", "live")
    for f in TASK_FRAMES:
        assert _parse_task_frame(f, None)[0] == f
    for u in TASK_FRAME_UPDATES:
        assert _parse_task_frame("tool", u)[1] == u


def test_world_frame_is_the_default_and_skips_the_rotation_entirely():
    cfg = make_config()
    assert cfg.task_frame == "world"
    c = XTaskYZCorridorQPController(cfg)
    # None, deliberately, rather than eye(3): the callers branch on it to skip
    # the matmul, which is what keeps the world path byte-identical.
    assert c._task_frames(np.array([1.0, 0.0, 0.0, 0.0])) == (None, None)


def test_tool_frame_rotates_the_position_rows_by_R_tool_transpose(monkeypatch):
    """The reduced task Jacobian must be exactly [R^T J_pos ; J_rot] rows."""
    quat = _quat_about_z(0.7)
    rot = quat_to_rotmat(quat)
    jac = make_jacobian()
    state = make_state(jac=jac)
    state["ee_quat"] = quat

    # task_excluded_joints=() so this measures the FRAME ROTATION alone --
    # the default (0,) zeroes shoulder_pan's column and would dominate H.
    cfg = make_config(task_frame="tool", task_frame_update="frozen",
                      task_axis_rows=(0, 1), corridor_axis_rows=(2,),
                      task_excluded_joints=())
    c = XTaskYZCorridorQPController(cfg)
    c.reset_from_state(state)
    calls = capture_qp(monkeypatch)
    c.compute(state)

    j_tool = np.vstack([rot.T @ jac[0:3, :], jac[3:6, :]])
    expected = np.vstack([j_tool[[0, 1], :], j_tool[3:6, :]])
    weights = np.diag([max(cfg.kp_x, 1e-6), max(cfg.kp_y, 1e-6)] + [max(cfg.kp_rot, 1e-6)] * 3)
    lam = max(cfg.posture_regularization, 1e-6)
    want = 2.0 * (expected.T @ weights @ expected + lam * np.eye(6))
    assert np.allclose(calls[0]["hessian"], want, atol=1e-12)


def test_tool_frame_reset_snapshots_the_rotation():
    quat = _quat_about_z(0.4)
    state = make_state()
    state["ee_quat"] = quat
    c = XTaskYZCorridorQPController(make_config(task_frame="tool"))
    c.reset_from_state(state)
    assert np.allclose(c._R0, quat_to_rotmat(quat), atol=1e-12)


def test_frozen_live_and_hybrid_pick_the_documented_rotations():
    """The three modes are distinct mechanisms: which R the TASK rows see and
    which the CORRIDOR row sees differ, and that difference only appears once
    the tool has actually rotated away from its reset attitude."""
    q0 = _quat_about_z(0.0)
    q1 = _quat_about_z(0.9)
    r0, r1 = quat_to_rotmat(q0), quat_to_rotmat(q1)
    state = make_state()
    state["ee_quat"] = q0

    got = {}
    for mode in TASK_FRAME_UPDATES:
        c = XTaskYZCorridorQPController(
            make_config(task_frame="tool", task_frame_update=mode)
        )
        c.reset_from_state(state)
        got[mode] = c._task_frames(q1)

    assert np.allclose(got["frozen"][0], r0) and np.allclose(got["frozen"][1], r0)
    assert np.allclose(got["live"][0], r1) and np.allclose(got["live"][1], r1)
    # hybrid: live task rows, frozen corridor row -- the whole point of it.
    assert np.allclose(got["hybrid"][0], r1) and np.allclose(got["hybrid"][1], r0)
    assert not np.allclose(r0, r1), "vacuous unless the tool really rotated"


def test_tool_frame_corridor_bounds_follow_the_corridor_frame():
    """A tool-frame corridor must bracket the START position expressed in the
    corridor frame, not the world one, or the barrier bounds a different
    quantity than the row it is attached to."""
    quat = _quat_about_z(0.6)
    rot = quat_to_rotmat(quat)
    p0 = np.array([0.4, -0.2, 0.3])
    state = make_state(ee_pos=tuple(p0))
    state["ee_quat"] = quat
    cfg = make_config(task_frame="tool", task_frame_update="frozen",
                      task_axis_rows=(0, 1), corridor_axis_rows=(2,),
                      yz_corridor_enabled=True, z_corridor_half_width_m=0.05)
    c = XTaskYZCorridorQPController(cfg)
    c.reset_from_state(state)
    out = c.compute(state)
    centre = float((rot.T @ p0)[2])
    assert out.z_min == pytest.approx(centre - 0.05, abs=1e-12)
    assert out.z_max == pytest.approx(centre + 0.05, abs=1e-12)


def test_shipped_configs_still_parse_with_the_world_default():
    import yaml

    for name in (
        "ur5e_mujoco_torque_x_task_yz_corridor_qp_enabled.yaml",
        "ur5e_mujoco_torque_x_z_task_y_corridor_qp_enabled.yaml",
    ):
        with open(REPO_ROOT / "config" / name, "r", encoding="utf-8") as fh:
            ctrl = yaml.safe_load(fh)["controller"]
        cfg = XTaskYZCorridorQPConfig.from_controller_yaml_section(ctrl)
        assert cfg.task_frame == "world"
        assert cfg.task_frame_update == "frozen"


# --------------------------------------------------------------------------- #
# 10. Orientation HOCBF row (2026-08-15).
# --------------------------------------------------------------------------- #
def test_orientation_cbf_row_matches_hand_computed_case():
    e = np.array([0.05, -0.03, 0.08])
    jac_rot_ref = np.array(
        [
            [0.2, -0.1, 0.05, 0.0, 0.3, -0.2],
            [0.0, 0.15, -0.25, 0.1, 0.0, 0.05],
            [-0.1, 0.05, 0.2, -0.05, 0.1, 0.0],
        ]
    )
    m_inv = np.diag([1.0, 0.5, 2.0, 4.0, 0.25, 1.5])
    bias = np.array([1.0, -2.0, 0.5, 0.0, 0.25, -0.75])
    qd = np.array([0.1, 0.2, -0.3, 0.05, 0.0, 0.4])
    max_error, a1, a2 = 0.20, 6.0, 9.0

    a_row, b, h, hdot, edot = XTaskYZCorridorQPController._orientation_cbf_row(
        e=e, jac_rot_ref=jac_rot_ref, m_inv=m_inv, bias=bias, qd=qd,
        max_error_rad=max_error, alpha1=a1, alpha2=a2,
    )

    edot_expected = jac_rot_ref @ qd
    lie = (e @ jac_rot_ref) @ m_inv
    h_expected = max_error ** 2 - float(e @ e)
    hdot_expected = -2.0 * float(e @ edot_expected)
    b_expected = (
        -2.0 * float(edot_expected @ edot_expected)
        + 2.0 * float(lie @ bias)
        + (a1 + a2) * hdot_expected
        + a1 * a2 * h_expected
    )
    assert np.allclose(a_row[0], 2.0 * lie, atol=0.0, rtol=0.0)
    assert np.allclose(edot, edot_expected, atol=0.0, rtol=0.0)
    assert h == pytest.approx(h_expected, rel=1e-12)
    assert hdot == pytest.approx(hdot_expected, rel=1e-12)
    assert b == pytest.approx(b_expected, rel=1e-12)


def test_orientation_cbf_row_encodes_the_hocbf_condition_itself():
    """``A tau <= b`` must be algebraically identical to
    ``hddot + (a1+a2) hdot + a1 a2 h >= 0`` with
    ``h = theta_max^2 - e^T e`` and ``eddot = jac_rot_ref M^-1 (tau - bias)``
    -- the POSITIVE sign verified by finite difference (see the controller
    module docstring's item 6), not the naive ``eddot = -J_r M^-1(...)``."""
    rng = np.random.default_rng(31)
    e = rng.normal(0.0, 0.05, 3)
    jac_rot_ref = rng.normal(0.0, 1.0, (3, 6))
    m_inv = np.diag(rng.uniform(0.2, 2.0, 6))
    bias = rng.normal(0.0, 1.0, 6)
    qd = rng.normal(0.0, 0.5, 6)
    tau = rng.normal(0.0, 20.0, 6)
    max_error, a1, a2 = 0.20, 5.0, 8.0

    a_row, b, h, hdot, edot = XTaskYZCorridorQPController._orientation_cbf_row(
        e=e, jac_rot_ref=jac_rot_ref, m_inv=m_inv, bias=bias, qd=qd,
        max_error_rad=max_error, alpha1=a1, alpha2=a2,
    )
    eddot = jac_rot_ref @ m_inv @ (tau - bias)
    hddot = -2.0 * float(edot @ edot) - 2.0 * float(e @ eddot)
    hocbf = hddot + (a1 + a2) * hdot + a1 * a2 * h
    assert float(a_row[0] @ tau - b) == pytest.approx(-hocbf, rel=1e-9, abs=1e-9)


def test_orientation_cbf_row_uses_the_negative_sign_would_fail():
    """Sanity check that the algebra genuinely distinguishes the two sign
    hypotheses the module docstring reports checking: with the WRONG sign
    (``eddot = -jac_rot_ref M^-1(tau-bias)``, the naive/unverified choice),
    the HOCBF identity above must NOT hold in general."""
    rng = np.random.default_rng(32)
    e = rng.normal(0.0, 0.05, 3)
    jac_rot_ref = rng.normal(0.0, 1.0, (3, 6))
    m_inv = np.diag(rng.uniform(0.2, 2.0, 6))
    bias = rng.normal(0.0, 1.0, 6)
    qd = rng.normal(0.0, 0.5, 6)
    tau = rng.normal(0.0, 20.0, 6)
    max_error, a1, a2 = 0.20, 5.0, 8.0

    a_row, b, h, hdot, edot = XTaskYZCorridorQPController._orientation_cbf_row(
        e=e, jac_rot_ref=jac_rot_ref, m_inv=m_inv, bias=bias, qd=qd,
        max_error_rad=max_error, alpha1=a1, alpha2=a2,
    )
    eddot_wrong = -(jac_rot_ref @ m_inv @ (tau - bias))
    hddot_wrong = -2.0 * float(edot @ edot) - 2.0 * float(e @ eddot_wrong)
    hocbf_wrong = hddot_wrong + (a1 + a2) * hdot + a1 * a2 * h
    assert float(a_row[0] @ tau - b) != pytest.approx(-hocbf_wrong, rel=1e-9, abs=1e-9)


def test_orientation_cbf_row_matches_independent_closed_form_across_random_states(monkeypatch):
    """Builds the row THROUGH the real controller (non-identity reference
    orientation, real random q/qd/M/bias/Jacobian) at >=5 random states and
    checks it against an independently re-derived closed-form row -- not a
    re-statement of the controller's own expressions."""
    from controller_core.kinematics_utils import orientation_error_vec_wxyz

    rng = np.random.default_rng(7)
    n_checked = 0
    for i in range(6):
        jac = make_jacobian(seed=300 + i)
        mass = make_mass(seed=400 + i)
        quat_ref = _quat_about_z(float(rng.uniform(-1.2, 1.2)))
        quat_cur = _quat_about_z(float(rng.uniform(-1.2, 1.2)))
        q = rng.normal(0.0, 0.3, 6)
        qd = rng.normal(0.0, 0.4, 6)
        gravity = rng.normal(0.0, 3.0, 6)
        a1 = float(rng.uniform(3.0, 15.0))
        a2 = float(rng.uniform(3.0, 15.0))
        max_error = 0.20

        cfg = make_config(
            orientation_cbf=True,
            orientation_cbf_max_error_rad=max_error,
            orientation_cbf_alpha1=a1,
            orientation_cbf_alpha2=a2,
            task_excluded_joints=(),
        )
        ctrl = XTaskYZCorridorQPController(cfg)
        reset_state = make_state(jac=jac, mass=mass)
        reset_state["ee_quat"] = quat_ref
        ctrl.reset_from_state(reset_state)

        state = make_state(jac=jac, mass=mass)
        state["ee_quat"] = quat_cur
        state["q"] = q
        state["qd"] = qd
        state["gravity_torque"] = gravity

        calls = capture_qp(monkeypatch)
        out = ctrl.compute(state)
        assert out.qp_num_ineq_rows == 1
        a_ineq = calls[-1]["a_ineq"]
        b_ineq = calls[-1]["b_ineq"]
        assert a_ineq is not None and b_ineq is not None

        # Independent closed-form re-derivation.
        quat_ref_arr = np.asarray(quat_ref, dtype=np.float64)
        quat_cur_arr = np.asarray(quat_cur, dtype=np.float64)
        e_expected = orientation_error_vec_wxyz(quat_ref_arr, quat_cur_arr)
        r_ref = quat_to_rotmat(quat_ref_arr)
        jac_rot_ref_expected = r_ref.T @ jac[3:6, :]
        m_inv_expected = np.linalg.inv(mass)
        edot_expected = jac_rot_ref_expected @ qd
        lie_expected = (e_expected @ jac_rot_ref_expected) @ m_inv_expected
        h_expected = max_error ** 2 - float(e_expected @ e_expected)
        hdot_expected = -2.0 * float(e_expected @ edot_expected)
        a_expected = 2.0 * lie_expected
        b_expected = (
            -2.0 * float(edot_expected @ edot_expected)
            + 2.0 * float(lie_expected @ gravity)
            + (a1 + a2) * hdot_expected
            + a1 * a2 * h_expected
        )
        assert np.allclose(a_ineq[0], a_expected, atol=1e-9, rtol=1e-9)
        assert b_ineq[0] == pytest.approx(b_expected, rel=1e-9, abs=1e-9)
        assert out.orientation_cbf_h == pytest.approx(h_expected, rel=1e-9)
        n_checked += 1
    assert n_checked >= 5


def test_orientation_cbf_row_uses_the_r_ref_rotation_not_raw_jacobian(monkeypatch):
    """At a pose where the reference orientation is far from identity, the
    row built from the RAW (un-rotated) angular Jacobian must differ from the
    one the controller actually emits -- otherwise the rotation correction
    documented in the module docstring is a no-op and the test above could
    not have caught its absence."""
    from controller_core.kinematics_utils import rotvec_to_quat_wxyz

    jac = make_jacobian(seed=11)
    mass = make_mass(seed=12)
    # A general (non-axis-aligned) rotation: a rotation purely about Z would
    # leave the Z-component of `e` invariant under R_ref^T (Z is R_ref's own
    # axis), making the "raw vs rotated" row COINCIDENTALLY identical for
    # any e that lands purely along Z -- exactly the degenerate case this
    # test exists to avoid, found the hard way while writing it.
    quat_ref = rotvec_to_quat_wxyz(np.array([0.5, -0.7, 0.9]))
    cfg = make_config(orientation_cbf=True, task_excluded_joints=())
    ctrl = XTaskYZCorridorQPController(cfg)
    reset_state = make_state(jac=jac, mass=mass)
    reset_state["ee_quat"] = quat_ref
    ctrl.reset_from_state(reset_state)

    state = make_state(jac=jac, mass=mass)
    state["ee_quat"] = rotvec_to_quat_wxyz(np.array([-0.2, 0.4, -0.1]))
    state["q"] = np.array([0.05, -0.1, 0.2, -0.05, 0.1, -0.02])
    state["qd"] = np.array([0.2, -0.1, 0.15, 0.05, -0.2, 0.1])

    calls = capture_qp(monkeypatch)
    ctrl.compute(state)
    a_actual = calls[-1]["a_ineq"][0]

    r_ref = quat_to_rotmat(np.asarray(quat_ref, dtype=np.float64))
    assert not np.allclose(r_ref, np.eye(3), atol=1e-3), "test needs a non-identity reference"
    # Row that WOULD have been built from the raw, un-rotated Jacobian.
    from controller_core.kinematics_utils import orientation_error_vec_wxyz

    e = orientation_error_vec_wxyz(
        np.asarray(quat_ref, dtype=np.float64),
        np.asarray(state["ee_quat"], dtype=np.float64),
    )
    m_inv = np.linalg.inv(mass)
    a_wrong = 2.0 * ((e @ jac[3:6, :]) @ m_inv)
    assert not np.allclose(a_actual, a_wrong, atol=1e-6, rtol=1e-6)


def test_orientation_cbf_inactive_far_from_the_limit():
    cfg = make_config(orientation_cbf=True, orientation_cbf_max_error_rad=0.20,
                       task_excluded_joints=())
    ctrl = XTaskYZCorridorQPController(cfg)
    ref = make_state()
    ctrl.reset_from_state(ref)
    state = make_state()
    state["ee_quat"] = ref["ee_quat"]  # zero orientation error
    out = ctrl.compute(state)
    assert out.orientation_cbf_h == pytest.approx(0.20 ** 2, rel=1e-9)
    assert out.orientation_cbf_active is False


def test_orientation_cbf_active_near_the_limit():
    """A reference/current orientation pair whose error norm sits just under
    ``orientation_cbf_max_error_rad``, WITH the arm actively rotating further
    toward the wall (``qd`` chosen along the error-growing direction), must
    make the row BINDING (active at tau_des) -- the same "far away: inert,
    moving toward the wall: engaged" property the Y/Z corridor rows already
    have. (At ``qd=0`` this same (h, alpha) pair is provably NOT binding --
    ``hdot=0`` and ``a1 a2 h`` alone is positive -- so ``qd`` has to carry the
    real content of this test, not just the position error.)"""
    from controller_core.kinematics_utils import orientation_error_vec_wxyz

    max_error = 0.20
    quat_ref = _quat_about_z(0.0)
    # A pure-Z rotation error of just under max_error radians.
    quat_cur = _quat_about_z(max_error - 0.01)
    cfg = make_config(
        orientation_cbf=True, orientation_cbf_max_error_rad=max_error,
        orientation_cbf_alpha1=10.0, orientation_cbf_alpha2=10.0,
        kp_rot=0.0, kd_rot=0.0,  # no task authority pulling e back down on its own
        task_excluded_joints=(),
    )
    jac = make_jacobian()
    ctrl = XTaskYZCorridorQPController(cfg)
    ref = make_state(jac=jac)
    ref["ee_quat"] = quat_ref
    ctrl.reset_from_state(ref)

    state_static = make_state(jac=jac)
    state_static["ee_quat"] = quat_cur
    out_static = ctrl.compute(state_static)
    assert out_static.orientation_cbf_active is False  # qd=0: genuinely inert here

    e = orientation_error_vec_wxyz(quat_ref, quat_cur)
    state_moving = make_state(jac=jac)
    state_moving["ee_quat"] = quat_cur
    state_moving["qd"] = 200.0 * float(e[2]) * jac[5, :]  # drives ||e|| further up
    out_moving = ctrl.compute(state_moving)
    assert out_moving.orientation_cbf_h == pytest.approx(out_static.orientation_cbf_h, rel=1e-9)
    assert out_moving.orientation_cbf_active is True


def test_orientation_cbf_default_off_leaves_hessian_linear_and_tau_unchanged(monkeypatch):
    """The byte-identical guarantee: with ``orientation_cbf=False`` (the
    class default), NOTHING about the QP the controller solves may differ
    from before this mechanism existed -- not the row count, not the
    Hessian, not the linear term, not the returned torque."""
    jac = make_jacobian(seed=55)
    mass = make_mass(seed=56)
    ref = make_state(jac=jac, mass=mass)
    ref["ee_quat"] = _quat_about_z(0.5)
    state = make_state(jac=jac, mass=mass)
    state["ee_quat"] = _quat_about_z(0.2)

    calls_off = capture_qp(monkeypatch)
    cfg_off = make_config(yz_corridor_enabled=True, manipulability_cbf=False)
    ctrl_off = XTaskYZCorridorQPController(cfg_off)
    ctrl_off.reset_from_state(ref)
    out_off = ctrl_off.compute(state)
    hessian_off = calls_off[-1]["hessian"]
    linear_off = calls_off[-1]["linear"]
    n_rows_off = out_off.qp_num_ineq_rows

    calls_on = capture_qp(monkeypatch)
    # Explicitly setting the field to its own default -- proves "field
    # present but False" behaves identically to "field absent" too.
    cfg_default = make_config(yz_corridor_enabled=True, manipulability_cbf=False,
                               orientation_cbf=False)
    ctrl_default = XTaskYZCorridorQPController(cfg_default)
    ctrl_default.reset_from_state(ref)
    out_default = ctrl_default.compute(state)

    assert np.array_equal(hessian_off, calls_on[-1]["hessian"])
    assert np.array_equal(linear_off, calls_on[-1]["linear"])
    assert np.array_equal(out_off.tau, out_default.tau)
    assert np.array_equal(out_off.tau_preclip, out_default.tau_preclip)
    assert out_default.qp_num_ineq_rows == n_rows_off
    assert out_default.orientation_cbf_active is False
    assert out_default.orientation_cbf_h is None
    assert out_default.orientation_cbf_feasible is True


def test_orientation_cbf_row_stacks_alongside_corridor_and_manipulability_rows():
    cfg = make_config(
        yz_corridor_enabled=True, manipulability_cbf=True, manipulability_cbf_epsilon=0.2,
        orientation_cbf=True,
    )
    ctrl = XTaskYZCorridorQPController(cfg, jacobian_fn=analytic_jacobian)
    q = np.array([0.0, 0.0, 0.0, 0.0, 0.5, 0.0])
    state = make_state(jac=analytic_jacobian(q))
    state["q"] = q
    ctrl.reset_from_state(state)
    out = ctrl.compute(state)
    # 4 corridor rows + 1 manipulability row + 1 orientation row.
    assert out.qp_num_ineq_rows == 6
    assert out.orientation_cbf_h is not None


def test_orientation_cbf_without_mass_matrix_raises():
    ctrl = XTaskYZCorridorQPController(make_config(orientation_cbf=True))
    state = make_state(mass_matrix=False)
    ctrl.reset_from_state(state)
    with pytest.raises(ValueError, match="mass_matrix"):
        ctrl.compute(state)


def test_orientation_cbf_defaults_are_off_with_the_documented_values():
    cfg = XTaskYZCorridorQPConfig()
    assert cfg.orientation_cbf is False
    assert cfg.orientation_cbf_max_error_rad == 0.20
    assert cfg.orientation_cbf_alpha1 == 10.0
    assert cfg.orientation_cbf_alpha2 == 10.0


def test_orientation_cbf_yaml_roundtrip_reads_every_new_field():
    cfg = XTaskYZCorridorQPConfig.from_controller_yaml_section(
        _yaml_section(
            orientation_cbf=True,
            orientation_cbf_max_error_rad=0.15,
            orientation_cbf_alpha1=6.0,
            orientation_cbf_alpha2=13.0,
        )
    )
    assert cfg.orientation_cbf is True
    assert cfg.orientation_cbf_max_error_rad == 0.15
    assert (cfg.orientation_cbf_alpha1, cfg.orientation_cbf_alpha2) == (6.0, 13.0)


def test_orientation_cbf_yaml_default_is_off():
    cfg = XTaskYZCorridorQPConfig.from_controller_yaml_section(_yaml_section())
    assert cfg.orientation_cbf is False
    assert cfg.orientation_cbf_max_error_rad == 0.20
    assert cfg.orientation_cbf_alpha1 == 10.0
    assert cfg.orientation_cbf_alpha2 == 10.0


@pytest.mark.parametrize("bad", [0.0, -0.1, float("nan"), float("inf")])
def test_bad_orientation_cbf_max_error_raises(bad):
    with pytest.raises(ValueError, match="orientation_cbf_max_error_rad"):
        XTaskYZCorridorQPConfig.from_controller_yaml_section(
            _yaml_section(orientation_cbf_max_error_rad=bad)
        )


@pytest.mark.parametrize("bad", [0.0, -1.0, float("nan"), float("inf")])
@pytest.mark.parametrize("field", ["orientation_cbf_alpha1", "orientation_cbf_alpha2"])
def test_bad_orientation_cbf_alpha_raises(field, bad):
    with pytest.raises(ValueError, match=field):
        XTaskYZCorridorQPConfig.from_controller_yaml_section(_yaml_section(**{field: bad}))
