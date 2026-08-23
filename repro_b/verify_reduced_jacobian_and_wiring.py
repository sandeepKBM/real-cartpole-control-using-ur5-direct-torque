"""Independent checks:

(A) REDUCED-JACOBIAN CLAIM. With the soft centering gains zeroed, the QP
    Hessian and the nominal task torque must be BYTE-identical for two states
    that differ only in the non-tracked (Y/Z) position and velocity.

(B) WIRING of the corridor rows through compute(): the A/b rows actually
    handed to the solver must match an independent closed-form derivation
    computed here from the raw state.
"""
from __future__ import annotations

import sys

import numpy as np

sys.path.insert(0, "/common/users/ss5772/real_Cartpole")

import controller_core.x_task_yz_corridor_qp.controller as ctrl_mod
from controller_core.x_task_yz_corridor_qp import (
    XTaskYZCorridorQPConfig,
    XTaskYZCorridorQPController,
)

CAPTURE = {}


def _capture_solver(hessian, linear, lo, hi, a_ineq, b_ineq, **kw):
    CAPTURE["hessian"] = np.array(hessian, copy=True)
    CAPTURE["linear"] = np.array(linear, copy=True)
    CAPTURE["lo"] = np.array(lo, copy=True)
    CAPTURE["hi"] = np.array(hi, copy=True)
    CAPTURE["a_ineq"] = None if a_ineq is None else np.array(a_ineq, copy=True)
    CAPTURE["b_ineq"] = None if b_ineq is None else np.array(b_ineq, copy=True)
    return np.zeros(6), np.zeros(0), True


def make_state(rng, *, ee_pos, ee_lin_vel, q=None, qd=None, J=None, M=None):
    n = 6
    if q is None:
        q = rng.normal(size=n) * 0.3
    if qd is None:
        qd = rng.normal(size=n) * 0.2
    if J is None:
        J = rng.normal(size=(6, n))
    if M is None:
        L = rng.normal(size=(n, n))
        M = L @ L.T + 3.0 * np.eye(n)
    return {
        "time": 0.0, "dt_s": 0.002,
        "q": q, "qd": qd,
        "ee_pos": np.asarray(ee_pos, float),
        "ee_quat": np.array([1.0, 0.0, 0.0, 0.0]),
        "ee_lin_vel": np.asarray(ee_lin_vel, float),
        "ee_ang_vel": np.array([0.01, -0.02, 0.03]),
        "jacobian": J,
        "mass_matrix": M,
        "target_x": 0.2,
        "target_x_vel": 0.1,
    }


def part_a():
    print("=" * 72)
    print("(A) reduced-Jacobian claim: Hessian + tau_task_nominal vs Y/Z state")
    print("=" * 72)
    real_solver = ctrl_mod.solve_constrained_box_qp
    ctrl_mod.solve_constrained_box_qp = _capture_solver
    try:
        rng = np.random.default_rng(7)
        worst_H = 0.0
        worst_tau = 0.0
        n_identical_H = 0
        n_identical_tau = 0
        trials = 6
        for k in range(trials):
            cfg = XTaskYZCorridorQPConfig(
                kp_y=0.0, kd_y=0.0, kp_z=0.0, kd_z=0.0,
                yz_corridor_enabled=True, manipulability_cbf=False,
            )
            q = rng.normal(size=6) * 0.3
            qd = rng.normal(size=6) * 0.2
            J = rng.normal(size=(6, 6))
            L = rng.normal(size=(6, 6))
            M = L @ L.T + 3.0 * np.eye(6)
            base_pos = np.array([0.4, 0.1, 0.3])
            base_vel = np.array([0.05, 0.0, 0.0])
            s1 = make_state(rng, ee_pos=base_pos, ee_lin_vel=base_vel, q=q, qd=qd, J=J, M=M)
            c = XTaskYZCorridorQPController(cfg)
            c.reset_from_state(s1)
            out1 = c.compute(s1)
            H1 = CAPTURE["hessian"].copy()

            # differ ONLY in Y/Z position and Y/Z velocity
            s2 = make_state(
                rng,
                ee_pos=base_pos + np.array([0.0, 0.037, -0.021]),
                ee_lin_vel=base_vel + np.array([0.0, 0.31, -0.44]),
                q=q, qd=qd, J=J, M=M,
            )
            out2 = c.compute(s2)
            H2 = CAPTURE["hessian"].copy()

            dH = np.max(np.abs(H1 - H2))
            dtau = np.max(np.abs(out1.tau_task_nominal - out2.tau_task_nominal))
            n_identical_H += int(np.array_equal(H1, H2))
            n_identical_tau += int(np.array_equal(out1.tau_task_nominal, out2.tau_task_nominal))
            worst_H = max(worst_H, dH)
            worst_tau = max(worst_tau, dtau)
            # sanity: the corridor rows themselves DID change (else the test is vacuous)
            print(f"  trial {k}: max|dH|={dH:.3e} bytes_eq={np.array_equal(H1,H2)}  "
                  f"max|d tau_task|={dtau:.3e} bytes_eq="
                  f"{np.array_equal(out1.tau_task_nominal, out2.tau_task_nominal)}  "
                  f"y_err {out1.y_error:.4f}->{out2.y_error:.4f}")
        print(f"\n  byte-identical Hessian:          {n_identical_H}/{trials}")
        print(f"  byte-identical tau_task_nominal: {n_identical_tau}/{trials}")
        print(f"  worst |dH| = {worst_H:.3e}   worst |d tau_task_nominal| = {worst_tau:.3e}")

        # And confirm tau_yz_soft is exactly zero with zeroed soft gains
        print(f"  ||tau_yz_soft|| with zeroed soft gains = {np.linalg.norm(out2.tau_yz_soft):.3e}")
    finally:
        ctrl_mod.solve_constrained_box_qp = real_solver


def part_b():
    print()
    print("=" * 72)
    print("(B) corridor rows as actually built inside compute() vs my derivation")
    print("=" * 72)
    real_solver = ctrl_mod.solve_constrained_box_qp
    ctrl_mod.solve_constrained_box_qp = _capture_solver
    try:
        rng = np.random.default_rng(99)
        worst = 0.0
        for k in range(6):
            w_y, w_z = 0.05, 0.04
            a1, a2 = 10.0, 6.0
            cfg = XTaskYZCorridorQPConfig(
                yz_corridor_enabled=True, manipulability_cbf=False,
                y_corridor_half_width_m=w_y, z_corridor_half_width_m=w_z,
                yz_corridor_alpha1=a1, yz_corridor_alpha2=a2,
            )
            p0 = np.array([0.4, 0.1, 0.3])
            s0 = make_state(rng, ee_pos=p0, ee_lin_vel=np.zeros(3))
            c = XTaskYZCorridorQPController(cfg)
            c.reset_from_state(s0)
            p = p0 + rng.normal(size=3) * 0.02
            s = make_state(rng, ee_pos=p, ee_lin_vel=rng.normal(size=3) * 0.1,
                           q=s0["q"], qd=s0["qd"], J=s0["jacobian"], M=s0["mass_matrix"])
            c.compute(s)
            A = CAPTURE["a_ineq"]
            b = CAPTURE["b_ineq"]

            J = s0["jacobian"]
            m_inv = np.linalg.inv(s0["mass_matrix"])
            qd = s0["qd"]
            bias = np.zeros(6)  # controller passes gravity, which is absent -> zeros
            expect_A, expect_b = [], []
            for axis, w in ((1, w_y), (2, w_z)):
                lie = J[axis, :] @ m_inv
                v = float(J[axis, :] @ qd)
                lower, upper = p0[axis] - w, p0[axis] + w
                val = float(p[axis])
                expect_A.append(lie)
                expect_b.append(float(lie @ bias) - (a1 + a2) * v + a1 * a2 * (upper - val))
                expect_A.append(-lie)
                expect_b.append(-float(lie @ bias) + (a1 + a2) * v + a1 * a2 * (val - lower))
            expect_A = np.vstack(expect_A)
            expect_b = np.array(expect_b)
            eA = float(np.max(np.abs(A - expect_A)))
            eb = float(np.max(np.abs(b - expect_b)))
            worst = max(worst, eA, eb)
            print(f"  trial {k}: rows={A.shape}  max|dA|={eA:.3e}  max|db|={eb:.3e}")
        print(f"\n  WORST error across compute()-built rows: {worst:.3e}")
    finally:
        ctrl_mod.solve_constrained_box_qp = real_solver


if __name__ == "__main__":
    part_a()
    part_b()
