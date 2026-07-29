"""Numerical parity between the on-robot URScript OSC loop and the validated
Python controller.

Mode 3 ("urscript") runs a hand-ported reimplementation of
``controller_core.x_axis_cartesian_impedance.XAxisCartesianImpedanceController``
on the robot controller. There is no way to execute PolyScope URScript here, so
this test does the next best thing: it reimplements the *exact* arithmetic of
``assets/urscript/x_axis_osc_inner.script.template`` in Python
(``_urscript_reference_tau`` below, annotated with the template line it mirrors)
and diffs it against the ground-truth controller over a battery of shared states.

In the regime where the two are *supposed* to agree (no saturation, gravity
added by PolyScope not Python) they must match to floating-point tolerance,
INCLUDING nullspace-posture projection (2026-07-26), geometric task-scale
backtracking (2026-07-26), and cond(J) singular-value wrench scaling
(2026-07-29 -- the last of the three known divergences; URScript has no
built-in SVD, so cond(J) is *estimated* via jacobi_eigenvalues_sym6, not
computed exactly, so this one term is checked to a tight empirically-measured
tolerance rather than being a bit-exact port -- see
docs/status/urscript_singular_scaling_parity_2026-07-29.md). This is the
regression net: any future edit to either side that breaks the agreement trips
these tests.

NOTE: this proves numerical parity of the control math only. It says nothing
about whether the URScript runs at 500 Hz on real hardware or behaves correctly
on a real arm -- neither has been tested.
"""

from __future__ import annotations

import re

import numpy as np
import pytest

from controller_core.x_axis_cartesian_impedance import (
    CartesianImpedanceConfig,
    XAxisCartesianImpedanceController,
)
from hardware.urscript_gen import (
    DEFAULT_CONFIG,
    DEFAULT_TEMPLATE,
    UrscriptOscParams,
    load_params_from_yaml,
    render_urscript,
)

_HUGE_LIMIT = np.full(6, 1.0e6)  # non-saturating, isolates the control law


def _rand_spd(rng: np.random.Generator, n: int = 6) -> np.ndarray:
    a = rng.normal(size=(n, n))
    return a @ a.T + n * np.eye(n)


def _jac_with_cond(rng: np.random.Generator, cond: float) -> np.ndarray:
    """Random 6x6 Jacobian with a prescribed condition number."""
    u, _, vt = np.linalg.svd(rng.normal(size=(6, 6)))
    s = np.linspace(1.0, 1.0 / cond, 6)
    return u @ np.diag(s) @ vt


def _jacobi_eigenvalues_sym6(a_in: np.ndarray, sweeps: int) -> np.ndarray:
    """Python transcription of the ``jacobi_eigenvalues_sym6`` URScript helper
    (trig-free cyclic Jacobi eigenvalue algorithm for a 6x6 symmetric matrix).
    Same loop structure, same rotation formula, same sweep count as the
    template -- this is not calling numpy's eigensolver, it is the identical
    from-scratch algorithm, so this test measures whether the URScript port
    (as transcribed here) agrees with the exact-SVD Python controller, not
    whether Jacobi-vs-numpy agree with each other."""
    a = a_in.copy()
    n = 6
    for _sweep in range(sweeps):
        for p in range(n - 1):
            for q in range(p + 1, n):
                apq = a[p, q]
                if abs(apq) > 1.0e-13:
                    app = a[p, p]
                    aqq = a[q, q]
                    theta = (aqq - app) / (2.0 * apq)
                    sgn = 1.0 if theta >= 0.0 else -1.0
                    t = sgn / (abs(theta) + np.sqrt(theta * theta + 1.0))
                    c = 1.0 / np.sqrt(t * t + 1.0)
                    s = t * c
                    for k in range(n):
                        if k != p and k != q:
                            akp = a[k, p]
                            akq = a[k, q]
                            a[k, p] = c * akp - s * akq
                            a[p, k] = a[k, p]
                            a[k, q] = s * akp + c * akq
                            a[q, k] = a[k, q]
                    a[p, p] = app - t * apq
                    a[q, q] = aqq + t * apq
                    a[p, q] = 0.0
                    a[q, p] = 0.0
    return np.array([a[0, 0], a[1, 1], a[2, 2], a[3, 3], a[4, 4], a[5, 5]])


def _urscript_cond_estimate(J: np.ndarray, sweeps: int) -> float:
    """Python transcription of the template's on-robot cond(J) estimate:
    sigma_max(J) from the top eigenvalue of J^T*J, sigma_min(J) from the
    reciprocal square root of the top eigenvalue of inv(J)^T*inv(J) -- avoids
    squaring the conditioning the way reading J^T*J's smallest eigenvalue
    directly would. See the template header and
    docs/status/urscript_singular_scaling_parity_2026-07-29.md."""
    JT = J.T
    Jinv = np.linalg.inv(J)
    JinvT = Jinv.T
    AtA = JT @ J
    BtB = JinvT @ Jinv
    eig_AtA = _jacobi_eigenvalues_sym6(AtA, sweeps)
    eig_BtB = _jacobi_eigenvalues_sym6(BtB, sweeps)
    return float(np.sqrt(eig_AtA.max() * eig_BtB.max()))


def _urscript_backtrack_task_scale(
    tau_nominal: np.ndarray, tau_limit: np.ndarray, *, resample_factor: float, min_scale: float, max_iters: int
) -> np.ndarray:
    """Python transcription of the ``backtrack_task_scale`` URScript helper."""
    task_scale = 1.0
    tau_candidate = tau_nominal.copy()
    feasible = bool(np.all(np.abs(tau_candidate) <= tau_limit + 1e-12))
    iters = 0
    while (not feasible) and (iters < max_iters) and (task_scale > min_scale + 1e-12):
        next_scale = task_scale * resample_factor
        if next_scale < min_scale:
            next_scale = min_scale
        if next_scale >= task_scale - 1e-12:
            break
        task_scale = next_scale
        tau_candidate = task_scale * tau_nominal
        feasible = bool(np.all(np.abs(tau_candidate) <= tau_limit + 1e-12))
        iters += 1
    return tau_candidate


def _urscript_reference_tau(
    *,
    J: np.ndarray,
    M: np.ndarray,
    q: np.ndarray,
    qd: np.ndarray,
    tcp: np.ndarray,
    x_des: float,
    x_vel_des: float,
    y0: float,
    z0: float,
    q_rest: np.ndarray,
    gains: dict[str, float],
    tau_lim: np.ndarray,
    headroom: float,
    use_lambda: bool,
    lambda_reg: float,
    use_nullspace: bool = False,
    resample_factor: float = 0.5,
    resample_min_scale: float = 1.0 / 16384.0,
    resample_max_iters: int = 14,
    cond_max: float = 1.0e18,
    jacobi_sweeps: int = 8,
) -> dict[str, np.ndarray]:
    """Faithful Python transcription of ``x_axis_osc_inner.script.template``.

    Line references are to that template. This is the on-robot math, in Python,
    used only as a parity oracle -- it is never shipped to the robot.
    """
    twist = J @ qd  # template: twist = J * qd
    vx, vy, vz, wx, wy, wz = twist
    x_err = x_des - tcp[0]
    y_err = y0 - tcp[1]
    z_err = z0 - tcp[2]
    Fx = gains["kp_x"] * x_err + gains["kd_x"] * (x_vel_des - vx)
    Fy = gains["kp_y"] * y_err - gains["kd_y"] * vy
    Fz = gains["kp_z"] * z_err - gains["kd_z"] * vz
    Mx = -gains["kd_rot"] * wx  # damping only, no kp_rot term
    My = -gains["kd_rot"] * wy
    Mz = -gains["kd_rot"] * wz
    wrench = np.array([Fx, Fy, Fz, Mx, My, Mz], dtype=np.float64)

    # Singular-value (cond(J)) wrench scaling -- template: computed right
    # after the wrench, before wrench shaping. cond_max=1e18 (the default
    # here) matches the disabled/no-singular-scale direction used by every
    # OTHER test in this module; only test_gap_singular_scaling passes a
    # finite cond_max to exercise the real port.
    cond_j = _urscript_cond_estimate(J, jacobi_sweeps)
    singular_scale = 1.0
    if cond_j > cond_max and cond_max > 0.0:
        singular_scale = cond_max / cond_j

    lam: np.ndarray | None = None
    m_inv: np.ndarray | None = None
    if use_lambda or use_nullspace:
        m_inv = np.linalg.inv(M)
        lam = np.linalg.inv(J @ m_inv @ J.T + lambda_reg * np.eye(6))

    if use_lambda:
        wrench_eff = lam @ wrench
    else:
        wrench_eff = wrench
    wrench_scaled = wrench_eff * singular_scale

    tau_task = J.T @ wrench_scaled  # cond(J) singular_scale now ported (2026-07-29)
    tau_damp = -gains["kd_joint"] * qd
    tau_post_raw = gains["kp_posture"] * (q_rest - q) - gains["kd_posture"] * qd

    if use_nullspace:
        # Dynamically consistent nullspace projector, mirrors
        # controller_core's j_bar/nullspace_proj exactly.
        j_bar = m_inv @ J.T @ lam
        nullspace_proj = np.eye(6) - J.T @ j_bar.T
        tau_post = nullspace_proj @ tau_post_raw
    else:
        tau_post = tau_post_raw

    tau_nominal = tau_task + tau_damp + tau_post  # NB: no gravity; PolyScope adds it

    lim_head = headroom * tau_lim
    tau_backtracked = _urscript_backtrack_task_scale(
        tau_nominal, lim_head, resample_factor=resample_factor,
        min_scale=resample_min_scale, max_iters=resample_max_iters,
    )
    tau_clamped = np.clip(tau_backtracked, -tau_lim, tau_lim)  # final hard clamp_vec6(tau, tau_lim)

    return {
        "wrench": wrench,
        "wrench_eff": wrench_eff,
        "wrench_scaled": wrench_scaled,
        "cond_j": np.array([cond_j]),
        "singular_scale": np.array([singular_scale]),
        "tau_task": tau_task,
        "tau_damp": tau_damp,
        "tau_post": tau_post,
        "tau": tau_clamped,
        "tau_preclamp": tau_nominal,
    }


def _parse_baked_gains(script: str) -> dict[str, float]:
    """Pull the numeric gains the generator baked into the rendered script,
    so the parity oracle is driven by the ACTUAL generator output, not an
    independently-typed copy of the config."""
    wanted = {
        "kp_x": "kp_x", "kd_x": "kd_x", "kp_y": "kp_y", "kd_y": "kd_y",
        "kp_z": "kp_z", "kd_z": "kd_z", "kd_rot": "kd_rot",
        "kp_post": "kp_posture", "kd_post": "kd_posture", "kd_joint": "kd_joint",
    }
    out: dict[str, float] = {}
    for script_name, ref_name in wanted.items():
        m = re.search(rf"^\s*{re.escape(script_name)} = ([-\d.eE+]+)\s*$", script, re.M)
        assert m is not None, f"could not find baked gain {script_name!r} in script"
        out[ref_name] = float(m.group(1))
    return out


def _py_config(*, cond_max: float, nullspace: bool, tau_lim: np.ndarray) -> CartesianImpedanceConfig:
    return CartesianImpedanceConfig(
        kp_x=400.0, kd_x=40.0, kp_y=80.0, kd_y=15.0, kp_z=120.0, kd_z=20.0,
        kp_rot=0.0, kd_rot=10.0, kp_posture=25.0, kd_posture=6.0, kd_joint=4.0,
        tau_max_nm=tau_lim.copy(), jacobian_singular_cond_max=cond_max,
        torque_headroom=0.9, task_space_inertia_shaping=True,
        nullspace_posture=nullspace, lambda_regularization=0.1,
        posture_reanchor_on_settle=False,
    )


def _run_python(
    cfg: CartesianImpedanceConfig,
    *,
    J: np.ndarray,
    M: np.ndarray,
    q0: np.ndarray,
    q: np.ndarray,
    qd: np.ndarray,
    tcp0: np.ndarray,
    tcp: np.ndarray,
    x_des: float,
    x_vel_des: float,
):
    ctrl = XAxisCartesianImpedanceController(cfg)
    # reset defines x0/y0/z0/quat0/q_rest from the "start" state
    ctrl.reset_from_state(
        dict(
            time=0.0, q=q0, qd=np.zeros(6), ee_pos=tcp0[:3],
            ee_quat=np.array([1.0, 0.0, 0.0, 0.0]),
            ee_lin_vel=np.zeros(3), ee_ang_vel=np.zeros(3),
            target_x=x_des, jacobian=J, mass_matrix=M,
        )
    )
    twist = J @ qd
    # Gravity intentionally omitted (PolyScope adds it on the robot), and
    # ee_lin/ang_vel set to J@qd so the velocity terms match the URScript's twist.
    return ctrl.compute(
        dict(
            time=0.01, q=q, qd=qd, ee_pos=tcp[:3],
            ee_quat=np.array([1.0, 0.0, 0.0, 0.0]),
            ee_lin_vel=twist[:3], ee_ang_vel=twist[3:],
            target_x=x_des, target_x_vel=x_vel_des, jacobian=J, mass_matrix=M,
        )
    )


def _sample_state(rng: np.random.Generator, *, cond: float):
    q0 = rng.normal(size=6)
    return dict(
        q0=q0,
        q=q0 + 0.05 * rng.normal(size=6),
        qd=0.1 * rng.normal(size=6),
        J=_jac_with_cond(rng, cond),
        M=_rand_spd(rng),
        tcp0=rng.normal(size=6),
        tcp=None,  # filled below
    )


# --------------------------------------------------------------------------- #
# 1. The generator loop: baked gains == config gains.
# --------------------------------------------------------------------------- #
def test_baked_gains_match_tuned_config() -> None:
    params = load_params_from_yaml(
        DEFAULT_CONFIG, target_x_delta_m=0.02, move_duration_s=1.0, duration_s=3.0
    )
    script = render_urscript(params, template_path=DEFAULT_TEMPLATE)
    baked = _parse_baked_gains(script)
    assert baked["kp_x"] == pytest.approx(params.kp_x)
    assert baked["kd_x"] == pytest.approx(params.kd_x)
    assert baked["kd_rot"] == pytest.approx(params.kd_rot)
    assert baked["kp_posture"] == pytest.approx(params.kp_posture)
    assert baked["kd_joint"] == pytest.approx(params.kd_joint)


# --------------------------------------------------------------------------- #
# 2. Faithful-regime parity: must match to floating point.
#    (singular scaling disabled, nullspace off, no saturation, no gravity)
# --------------------------------------------------------------------------- #
def test_faithful_regime_matches_python() -> None:
    rng = np.random.default_rng(1234)
    params = load_params_from_yaml(
        DEFAULT_CONFIG, target_x_delta_m=0.02, move_duration_s=1.0, duration_s=3.0
    )
    gains = _parse_baked_gains(render_urscript(params))
    for _ in range(50):
        s = _sample_state(rng, cond=50.0)  # well conditioned
        s["tcp"] = s["tcp0"] + 0.02 * rng.normal(size=6)
        x_des = float(s["tcp0"][0]) + 0.02
        x_vel = 0.01
        cfg = _py_config(cond_max=1.0e18, nullspace=False, tau_lim=_HUGE_LIMIT)
        out = _run_python(
            cfg, J=s["J"], M=s["M"], q0=s["q0"], q=s["q"], qd=s["qd"],
            tcp0=s["tcp0"], tcp=s["tcp"], x_des=x_des, x_vel_des=x_vel,
        )
        ref = _urscript_reference_tau(
            J=s["J"], M=s["M"], q=s["q"], qd=s["qd"], tcp=s["tcp"],
            x_des=x_des, x_vel_des=x_vel, y0=float(s["tcp0"][1]),
            z0=float(s["tcp0"][2]), q_rest=s["q0"], gains=gains,
            tau_lim=_HUGE_LIMIT, headroom=params.torque_headroom,
            use_lambda=params.use_lambda, lambda_reg=params.lambda_regularization,
        )
        assert np.allclose(out.wrench, ref["wrench"], atol=1e-9)
        assert np.allclose(out.tau_task_nominal, ref["tau_task"], atol=1e-9)
        # unsaturated => task_backtrack_scale == 1, so tau_posture is raw
        assert out.task_backtrack_scale == pytest.approx(1.0)
        assert np.allclose(out.tau_posture, ref["tau_post"], atol=1e-9)
        assert np.allclose(out.tau, ref["tau"], atol=1e-9)


# --------------------------------------------------------------------------- #
# 3. Nullspace-posture projection now matches (fixed 2026-07-26; was Gap 1).
# --------------------------------------------------------------------------- #
def test_nullspace_projection_matches_python() -> None:
    rng = np.random.default_rng(7)
    params = load_params_from_yaml(
        DEFAULT_CONFIG, target_x_delta_m=0.02, move_duration_s=1.0, duration_s=3.0
    )
    gains = _parse_baked_gains(render_urscript(params))
    saw_nonzero_projection_effect = False
    for _ in range(30):
        s = _sample_state(rng, cond=50.0)
        s["tcp"] = s["tcp0"] + 0.02 * rng.normal(size=6)
        x_des = float(s["tcp0"][0]) + 0.02
        cfg = _py_config(cond_max=1.0e18, nullspace=True, tau_lim=_HUGE_LIMIT)
        out = _run_python(
            cfg, J=s["J"], M=s["M"], q0=s["q0"], q=s["q"], qd=s["qd"],
            tcp0=s["tcp0"], tcp=s["tcp"], x_des=x_des, x_vel_des=0.0,
        )
        ref = _urscript_reference_tau(
            J=s["J"], M=s["M"], q=s["q"], qd=s["qd"], tcp=s["tcp"],
            x_des=x_des, x_vel_des=0.0, y0=float(s["tcp0"][1]),
            z0=float(s["tcp0"][2]), q_rest=s["q0"], gains=gains,
            tau_lim=_HUGE_LIMIT, headroom=params.torque_headroom,
            use_lambda=True, lambda_reg=params.lambda_regularization,
            use_nullspace=True,
        )
        assert np.allclose(out.tau_task_nominal, ref["tau_task"], atol=1e-9)
        assert np.allclose(out.tau_posture, ref["tau_post"], atol=1e-8)
        assert np.allclose(out.tau, ref["tau"], atol=1e-8)
        # Sanity: the projector must actually be doing something on these random
        # states (else this test would pass trivially even with a broken
        # projector on either side).
        raw_posture = gains["kp_posture"] * (s["q0"] - s["q"]) - gains["kd_posture"] * s["qd"]
        if float(np.max(np.abs(out.tau_posture - raw_posture))) > 1e-3:
            saw_nonzero_projection_effect = True
    assert saw_nonzero_projection_effect, "expected the projector to measurably change posture torque on some sample"


# --------------------------------------------------------------------------- #
# 4. cond(J) singular-value wrench scaling now matches (fixed 2026-07-29; was
#    the last remaining gap). URScript has no built-in SVD, so cond(J) is
#    ESTIMATED via jacobi_eigenvalues_sym6 (see the template header and
#    docs/status/urscript_singular_scaling_parity_2026-07-29.md), not computed
#    exactly like np.linalg.cond -- these tests measure how close that
#    estimate lands to the exact-SVD Python controller, in both the regime
#    where the config's real threshold (jacobian_singular_cond_max=1e5,
#    baked from config/ur5e_mujoco_torque_osc_tuned.yaml, DEFAULT_CONFIG
#    above) leaves scaling off, and the regime where it engages.
# --------------------------------------------------------------------------- #
def test_singular_scaling_clean_regime_matches_python() -> None:
    """Well-conditioned poses (cond(J) << jacobian_singular_cond_max):
    singular_scale must land at exactly 1.0 on both sides, so this must be an
    exact match, same tolerance as the other unsaturated-regime tests."""
    rng = np.random.default_rng(11)
    params = load_params_from_yaml(
        DEFAULT_CONFIG, target_x_delta_m=0.02, move_duration_s=1.0, duration_s=3.0
    )
    gains = _parse_baked_gains(render_urscript(params))
    for _ in range(20):
        s = _sample_state(rng, cond=50.0)
        s["tcp"] = s["tcp0"] + 0.02 * rng.normal(size=6)
        x_des = float(s["tcp0"][0]) + 0.02
        ref = _urscript_reference_tau(
            J=s["J"], M=s["M"], q=s["q"], qd=s["qd"], tcp=s["tcp"],
            x_des=x_des, x_vel_des=0.0, y0=float(s["tcp0"][1]),
            z0=float(s["tcp0"][2]), q_rest=s["q0"], gains=gains,
            tau_lim=_HUGE_LIMIT, headroom=params.torque_headroom,
            use_lambda=True, lambda_reg=params.lambda_regularization,
            cond_max=params.jacobian_singular_cond_max,
            jacobi_sweeps=params.singular_scale_jacobi_sweeps,
        )
        out = _run_python(
            _py_config(cond_max=params.jacobian_singular_cond_max, nullspace=False, tau_lim=_HUGE_LIMIT),
            J=s["J"], M=s["M"], q0=s["q0"], q=s["q"], qd=s["qd"],
            tcp0=s["tcp0"], tcp=s["tcp"], x_des=x_des, x_vel_des=0.0,
        )
        assert out.singular_scale == pytest.approx(1.0)
        assert ref["singular_scale"][0] == pytest.approx(1.0)
        assert np.allclose(out.tau_task_nominal, ref["tau_task"], atol=1e-8)
        assert np.allclose(out.tau, ref["tau"], atol=1e-8)


def test_singular_scaling_near_singular_matches_python() -> None:
    """Near-singular poses (cond(J) well above jacobian_singular_cond_max):
    singular_scale actually engages and collapses task authority on both
    sides. The Jacobi cond(J) estimate is an approximation (not exact SVD),
    so this is not bit-for-bit floating point, but measured empirically at
    cond(J)=1e7 (the DEFAULT_CONFIG-baked jacobi_sweeps=8) the relative error
    in cond(J) itself is ~5.7e-11 and the resulting tau_task_nominal disagrees
    by ~1e-11 Nm absolute -- tight enough to use the same atol=1e-8 discipline
    as the other unsaturated-regime tests, not a looser tolerance."""
    rng = np.random.default_rng(99)
    params = load_params_from_yaml(
        DEFAULT_CONFIG, target_x_delta_m=0.02, move_duration_s=1.0, duration_s=3.0
    )
    gains = _parse_baked_gains(render_urscript(params))
    s = _sample_state(rng, cond=1.0e7)  # near singular
    s["tcp"] = s["tcp0"] + 0.02 * rng.normal(size=6)
    x_des = float(s["tcp0"][0]) + 0.02

    ref = _urscript_reference_tau(
        J=s["J"], M=s["M"], q=s["q"], qd=s["qd"], tcp=s["tcp"],
        x_des=x_des, x_vel_des=0.0, y0=float(s["tcp0"][1]),
        z0=float(s["tcp0"][2]), q_rest=s["q0"], gains=gains,
        tau_lim=_HUGE_LIMIT, headroom=params.torque_headroom,
        use_lambda=True, lambda_reg=params.lambda_regularization,
        cond_max=params.jacobian_singular_cond_max,
        jacobi_sweeps=params.singular_scale_jacobi_sweeps,
    )

    # DEFAULT_CONFIG bakes jacobian_singular_cond_max=1e5 (unset in the yaml,
    # falls back to the class default -- see hardware/urscript_gen.py and
    # CartesianImpedanceConfig.jacobian_singular_cond_max). At cond(J)=1e7
    # both sides must now collapse task authority the same way.
    out = _run_python(
        _py_config(cond_max=params.jacobian_singular_cond_max, nullspace=False, tau_lim=_HUGE_LIMIT),
        J=s["J"], M=s["M"], q0=s["q0"], q=s["q"], qd=s["qd"],
        tcp0=s["tcp0"], tcp=s["tcp"], x_des=x_des, x_vel_des=0.0,
    )
    assert out.singular_scale < 0.5
    assert ref["singular_scale"][0] < 0.5
    assert np.allclose(out.tau_task_nominal, ref["tau_task"], atol=1e-8)
    assert np.allclose(out.tau, ref["tau"], atol=1e-8)

    # Sanity: confirm the estimate is actually close to the true SVD-based
    # cond(J), not just coincidentally producing a matching tau.
    assert ref["cond_j"][0] == pytest.approx(out.jacobian_cond, rel=1e-6)

    # No-singular-scale variant (cond_max=1e18): both sides agree scaling is
    # off, matching config/ur5e_mujoco_torque_osc_tuned_no_singular_scale.yaml's
    # direction.
    ref_ns = _urscript_reference_tau(
        J=s["J"], M=s["M"], q=s["q"], qd=s["qd"], tcp=s["tcp"],
        x_des=x_des, x_vel_des=0.0, y0=float(s["tcp0"][1]),
        z0=float(s["tcp0"][2]), q_rest=s["q0"], gains=gains,
        tau_lim=_HUGE_LIMIT, headroom=params.torque_headroom,
        use_lambda=True, lambda_reg=params.lambda_regularization,
        cond_max=1.0e18, jacobi_sweeps=params.singular_scale_jacobi_sweeps,
    )
    out_ns = _run_python(
        _py_config(cond_max=1.0e18, nullspace=False, tau_lim=_HUGE_LIMIT),
        J=s["J"], M=s["M"], q0=s["q0"], q=s["q"], qd=s["qd"],
        tcp0=s["tcp0"], tcp=s["tcp"], x_des=x_des, x_vel_des=0.0,
    )
    assert out_ns.singular_scale == pytest.approx(1.0)
    assert np.allclose(out_ns.tau_task_nominal, ref_ns["tau_task"], atol=1e-8)


# --------------------------------------------------------------------------- #
# 5. Geometric backtracking under saturation now matches (fixed 2026-07-26;
#    was Gap 3 -- previously a per-joint clamp that distorted torque direction).
# --------------------------------------------------------------------------- #
def test_backtracking_matches_python_under_saturation() -> None:
    rng = np.random.default_rng(2024)
    params = load_params_from_yaml(
        DEFAULT_CONFIG, target_x_delta_m=0.02, move_duration_s=1.0, duration_s=3.0
    )
    gains = _parse_baked_gains(render_urscript(params))
    tau_lim = np.array([150.0, 150.0, 150.0, 28.0, 28.0, 28.0])

    saw_backtracking_engage = False
    for _ in range(60):
        s = _sample_state(rng, cond=50.0)
        # large x error to force saturation
        s["tcp"] = s["tcp0"] + 0.02 * rng.normal(size=6)
        x_des = float(s["tcp0"][0]) + 0.5
        cfg = _py_config(cond_max=1.0e18, nullspace=False, tau_lim=tau_lim)
        out = _run_python(
            cfg, J=s["J"], M=s["M"], q0=s["q0"], q=s["q"], qd=s["qd"],
            tcp0=s["tcp0"], tcp=s["tcp"], x_des=x_des, x_vel_des=0.0,
        )
        ref = _urscript_reference_tau(
            J=s["J"], M=s["M"], q=s["q"], qd=s["qd"], tcp=s["tcp"],
            x_des=x_des, x_vel_des=0.0, y0=float(s["tcp0"][1]),
            z0=float(s["tcp0"][2]), q_rest=s["q0"], gains=gains,
            tau_lim=tau_lim, headroom=params.torque_headroom,
            use_lambda=True, lambda_reg=params.lambda_regularization,
        )
        # Safety invariant that MUST hold on both sides: never exceed the hard limit.
        assert np.all(np.abs(ref["tau"]) <= tau_lim + 1e-9)
        assert np.all(np.abs(out.tau) <= tau_lim + 1e-9)
        # Same algorithm on both sides now -> must agree, saturating or not.
        assert np.allclose(out.tau, ref["tau"], atol=1e-6)
        if out.task_backtrack_scale < 1.0 - 1e-9:
            saw_backtracking_engage = True
    assert saw_backtracking_engage, "expected at least one sample to actually saturate and engage backtracking"
