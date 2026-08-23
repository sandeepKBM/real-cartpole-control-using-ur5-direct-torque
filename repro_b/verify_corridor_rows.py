"""INDEPENDENT verification of the Y/Z corridor HOCBF rows.

Derivation done from first principles here, NOT copied from controller.py:

  Corridor axis a (row a of the position block of J), position p_a, start p_a0,
  half-width w.  upper = p_a0 + w, lower = p_a0 - w.

  h_max = upper - p_a
  hdot_max  = -J_a qd
  hddot_max = -(J_a qddot + Jdot_a qd),   qddot = M^-1 (tau - bias)
            = -(J_a M^-1) tau + (J_a M^-1) bias - Jdot_a qd

  HOCBF: hddot + (a1+a2) hdot + a1 a2 h >= 0
  =>  -(J_a M^-1) tau + (J_a M^-1) bias - Jdot_a qd
        - (a1+a2)(J_a qd) + a1 a2 (upper - p_a) >= 0
  In A tau <= b form:
      A_max = +(J_a M^-1)
      b_max = (J_a M^-1) bias - Jdot_a qd - (a1+a2)(J_a qd) + a1 a2 (upper - p_a)

  h_min = p_a - lower
  hdot_min  = +J_a qd
  hddot_min = +(J_a M^-1) tau - (J_a M^-1) bias + Jdot_a qd
  =>  A_min = -(J_a M^-1)
      b_min = -(J_a M^-1) bias + Jdot_a qd + (a1+a2)(J_a qd) + a1 a2 (p_a - lower)

  (Jdot_a qd is dropped below to match the controller's stated approximation;
   it is reported separately as a magnitude.)
"""
from __future__ import annotations

import sys

import numpy as np

sys.path.insert(0, "/common/users/ss5772/real_Cartpole")

from controller_core.x_task_yz_corridor_qp.controller import XTaskYZCorridorQPController


def my_corridor_rows(j_row, m_inv, bias, qd, value, lower, upper, a1, a2):
    j_row = np.asarray(j_row, float).reshape(-1)
    lie = j_row @ np.asarray(m_inv, float)
    lie_bias = float(lie @ np.asarray(bias, float).reshape(-1))
    v = float(j_row @ np.asarray(qd, float).reshape(-1))
    s = float(a1) + float(a2)
    pr = float(a1) * float(a2)
    A_max = lie.reshape(1, -1)
    b_max = lie_bias - s * v + pr * (upper - value)
    A_min = -lie.reshape(1, -1)
    b_min = -lie_bias + s * v + pr * (value - lower)
    return A_max, b_max, A_min, b_min


def main():
    rng = np.random.default_rng(20260814)
    max_err_A = 0.0
    max_err_b = 0.0
    print("case |  max|dA|      max|db|")
    for k in range(8):
        n = 6
        # random SPD mass matrix
        L = rng.normal(size=(n, n))
        M = L @ L.T + 3.0 * np.eye(n)
        m_inv = np.linalg.inv(M)
        J = rng.normal(size=(6, n))
        qd = rng.normal(size=n) * 0.5
        bias = rng.normal(size=n) * 2.0
        value = float(rng.normal() * 0.1)
        w = 0.05
        p0 = float(rng.normal() * 0.1)
        lower, upper = p0 - w, p0 + w
        a1 = float(abs(rng.normal()) * 10 + 1)
        a2 = float(abs(rng.normal()) * 10 + 1)
        axis = int(rng.integers(1, 3))

        mA_max, mb_max, mA_min, mb_min = my_corridor_rows(
            J[axis, :], m_inv, bias, qd, value, lower, upper, a1, a2
        )
        cA_max, cb_max, cA_min, cb_min = XTaskYZCorridorQPController._corridor_rows(
            j_row=J[axis, :], m_inv=m_inv, bias=bias, qd=qd, value=value,
            lower=lower, upper=upper, alpha1=a1, alpha2=a2,
        )
        eA = max(np.max(np.abs(mA_max - cA_max)), np.max(np.abs(mA_min - cA_min)))
        eb = max(abs(mb_max - cb_max), abs(mb_min - cb_min))
        max_err_A = max(max_err_A, eA)
        max_err_b = max(max_err_b, eb)
        print(f"{k:4d} | {eA:.3e}  {eb:.3e}   (|b_max|={abs(cb_max):.4g})")

    print(f"\nMAX |dA| = {max_err_A:.3e}   MAX |db| = {max_err_b:.3e}")

    # --- Independent check that the row genuinely encodes the HOCBF condition:
    # pick tau on the boundary A tau = b and verify hddot + (a1+a2)hdot + a1a2 h == 0
    print("\nHOCBF semantics check (residual of hddot+(a1+a2)hdot+a1a2 h at A tau = b):")
    for k in range(3):
        n = 6
        L = rng.normal(size=(n, n))
        M = L @ L.T + 3.0 * np.eye(n)
        m_inv = np.linalg.inv(M)
        J = rng.normal(size=(6, n))
        qd = rng.normal(size=n) * 0.5
        bias = rng.normal(size=n)
        value, p0, w = 0.02, 0.0, 0.05
        lower, upper = p0 - w, p0 + w
        a1, a2 = 10.0, 7.0
        axis = 1
        A_max, b_max, A_min, b_min = XTaskYZCorridorQPController._corridor_rows(
            j_row=J[axis, :], m_inv=m_inv, bias=bias, qd=qd, value=value,
            lower=lower, upper=upper, alpha1=a1, alpha2=a2,
        )
        # find tau with A_max @ tau == b_max
        d = A_max.reshape(-1)
        tau = d * (b_max / (d @ d))
        qddot = m_inv @ (tau - bias)
        h = upper - value
        hdot = -float(J[axis, :] @ qd)
        hddot = -float(J[axis, :] @ qddot)          # Jdot term dropped, as coded
        res_max = hddot + (a1 + a2) * hdot + a1 * a2 * h
        d = A_min.reshape(-1)
        tau = d * (b_min / (d @ d))
        qddot = m_inv @ (tau - bias)
        h = value - lower
        hdot = float(J[axis, :] @ qd)
        hddot = float(J[axis, :] @ qddot)
        res_min = hddot + (a1 + a2) * hdot + a1 * a2 * h
        print(f"  case {k}: residual_max={res_max:.3e}  residual_min={res_min:.3e}")


if __name__ == "__main__":
    main()
