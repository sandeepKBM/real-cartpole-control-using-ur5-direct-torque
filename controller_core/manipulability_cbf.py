"""Manipulability Control Barrier Function (CBF) -- a per-cycle QP safety
filter that makes it structurally hard for the torque law to drive the arm
INTO a kinematic singularity, rather than reacting once it is already there.

Method source: "Safe, Task-Consistent Manipulation with Operational Space
Control Barrier Functions" (OSCBF), arXiv:2503.06736, Section V-B1
(singularity-avoidance CBF) + Eq. 5 (high-order CBF) + Eq. 11-15
(operational-space / rigid-body dynamics). That paper runs on a 7-DOF Franka
Panda; NOTHING robot-specific is borrowed here -- only the method. This module
is the 6-DOF UR5e adaptation.

HOW THIS DIFFERS FROM EVERY EXISTING SINGULARITY MECHANISM IN THIS REPO
----------------------------------------------------------------------
This repo already has three, all of them REACTIVE or OFFLINE:
  - ``jacobian_singular_cond_max`` -- scales the whole wrench down once
    ``cond(J)`` is already bad (isotropic; and known to have frozen the
    controller outright, see AGENTS.md sec 4's 2026-07-30 entry);
  - ``svd_singularity_filtering`` -- per-direction damped-least-squares
    back-off, again applied only once a singular value is already small;
  - offline pose pre-filtering (pick a start pose with good ``cond(J)``).
None of them constrains the arm's MOTION; they only shrink authority after the
fact. A CBF is a different object: it adds an inequality constraint to a QP so
the commanded torque is *required* to keep a barrier function non-negative,
i.e. the singular set is made forward-invariant (an approach-rate limit, not a
magnitude limit). It is complementary to, not a replacement for, the three
above -- in particular it does nothing about the authority actually lost
in a direction that is ALREADY singular, which is what SCI addresses.

Note also that ``mu(q)`` (Yoshikawa manipulability, the PRODUCT of the singular
values) is a genuinely different quantity from ``cond(J)`` (the RATIO of the
extreme ones): ``cond`` is scale-invariant and blows up to +inf at a
singularity, ``mu`` is a smooth-ish volume measure that goes to 0. Measured on
this repo's real model (assets/ur5e_torque/scene.xml) sweeping ``wrist_2``
away from the transport singularity at HEIGHT_ALPHA_0_5_Q:

    wrist_2 (rad)   0.000     0.005     0.020     0.050     0.200    1.200
    mu             1.2e-18   9.4e-05   3.8e-04   9.4e-04   3.7e-03  1.8e-02
    cond(J)        7.3e+16   9.1e+02   2.3e+02   9.1e+01   2.9e+01  2.3e+01

i.e. ``mu`` is very nearly LINEAR in the distance from the singular set here,
which is exactly the property that makes it a usable barrier (``1/cond`` would
work too, but ``mu`` is what the paper uses and it has a bounded, well-scaled
gradient).

DERIVATION (this is the part that must be right; see also the unit tests in
tests/unit/test_manipulability_cbf.py, which check each step numerically)
----------------------------------------------------------------------------
State ``z = (q, qd)``; QP decision variable ``u = tau`` (joint torque, in Nm) --
torque, not task acceleration, because that is the level this repo's plant
interface actually accepts and the level ``hard_constraint_qp.py`` already
established for a genuine linear inequality here.

  1. Barrier (OSCBF Sec V-B1):

         mu(q) = prod_i sigma_i( J(q) ) = sqrt( det( J J^T ) )
         h(z)  = mu(q) - eps                                       (eps > 0)

     ``mu`` is computed from the singular values directly, not from
     ``det(J)``: for a 6x6 Jacobian the two are equal in exact arithmetic, but
     ``prod(svd)`` has no catastrophic cancellation and is non-negative by
     construction.

  2. Relative degree. ``h`` depends on ``q`` only, so

         hdot = grad_mu(q) . qd                                    (no tau)

     -- relative degree 2 w.r.t. a torque input. The paper says exactly this
     (velocity control: relative degree 1; torque control: relative degree 2)
     and resolves it with the high-order CBF of its Eq. 5:

         h2(z) = hdot(z) + alpha1( h(z) )
         Lf h2(z) + Lg h2(z) u  >=  -alpha2( h2(z) )

     With LINEAR class-K functions ``alpha1(s) = a1 s``, ``alpha2(s) = a2 s``
     this expands to the familiar second-order form

         hddot + (a1 + a2) hdot + a1 a2 h  >=  0                   (*)

     whose homogeneous solutions decay as ``exp(-a1 t)``, ``exp(-a2 t)``: the
     set ``{h >= 0}`` is forward-invariant, and from ``h < 0`` (already inside
     the singular region, which is where this repo's real transport start pose
     literally sits: mu = 1.2e-18 at HEIGHT_ALPHA_0_5_Q) the constraint drives
     ``h`` back up rather than merely holding.

  3. Making it affine in tau. Write ``g(q) := grad_mu(q)`` in R^6 and
     ``H_mu(q) := d^2 mu / dq^2``. Then

         hddot = d/dt ( g . qd ) = qd^T H_mu qd + g . qddot

     and the rigid-body dynamics (the same ``M`` as OSCBF Eq. 11-15) give

         M(q) qddot + b(q, qd) = tau     =>    qddot = M^-1 ( tau - b )

     so

         hddot = g^T M^-1 tau  -  g^T M^-1 b  +  qd^T H_mu qd

     -- LINEAR in tau, which is the whole point.

  4. The constraint row. Substituting into (*):

         g^T M^-1 tau  >=  g^T M^-1 b - qd^T H_mu qd
                           - (a1 + a2) (g . qd) - a1 a2 (mu - eps)

     and in the ``A x <= b`` form ``solve_constrained_box_qp`` accepts:

         A = -( g^T M^-1 )                                   shape (1, 6)
         b = -g^T M^-1 b_dyn + qd^T H_mu qd
             + (a1 + a2) (g . qd) + a1 a2 (mu - eps)          scalar

  5. The QP itself (OSCBF's structure: minimum deviation from the nominal
     task-space control law, subject to the CBF and the input limits):

         min_tau  || tau - tau_nominal ||^2
         s.t.     A tau <= b            (the CBF row above)
                  tau_lo <= tau <= tau_hi   (torque headroom box)

     Built with ``box_qp.build_weighted_least_squares_qp`` and solved with
     ``constrained_box_qp.solve_constrained_box_qp`` -- both already in this
     package, both already tested. NO new solver was written and
     ``box_qp.py`` needed NO extension: ``solve_constrained_box_qp`` was added
     2026-08-03 for exactly this class of problem (a small number of general
     linear inequality rows on top of a box) and a CBF row is one more
     instance of it.

WHAT IS APPROXIMATED (stated, not assumed away)
-----------------------------------------------
  - ``b_dyn`` is the caller-supplied dynamics bias. Coriolis is NOT included
    unless the caller puts it there -- the same omission every other torque
    path in this repo already makes (see hard_constraint_qp.py's module
    docstring). In the MuJoCo lane the adapter compensates gravity itself
    (``tau_applied = tau_controller + tau_gravity``), so the correct bias to
    pass alongside a controller output that does NOT contain gravity is zero,
    which is what an absent ``gravity_torque`` state key already yields.
  - ``grad_mu`` and ``qd^T H_mu qd`` are finite-differenced from
    ``jacobian_fn``. There is no way around needing ``J`` at PERTURBED ``q``:
    ``grad_mu`` depends on ``dJ/dq``, and the per-cycle state contract carries
    ``J(q)`` at the current ``q`` only. The full Hessian is never formed -- only
    the directional second derivative along ``qd`` is needed, which costs 2
    extra Jacobian evaluations instead of ~36 (see
    ``manipulability_directional_curvature``).
  - ``mu`` is NOT differentiable where a singular value crosses zero (it
    behaves like ``c|wrist_2|`` near this arm's wrist singularity), so the
    finite-difference curvature is meaningless within ~one FD step of the
    singular set itself. In normal operation ``eps`` keeps the state orders of
    magnitude further away than that; the risk is real only when starting from
    ``h < 0``, and it is bounded by the QP's own box (an absurd curvature can
    make the row infeasible, which is REPORTED, not silently absorbed).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from .box_qp import build_weighted_least_squares_qp
from .constrained_box_qp import solve_constrained_box_qp

#: A callable ``q -> J(q)`` returning the 6x6 world-frame end-effector
#: Jacobian for a joint configuration. Supplied by whoever owns the kinematic
#: model (the MuJoCo adapter, or ``model_dynamics.PinocchioUR5eDynamics``);
#: deliberately NOT part of the per-cycle state dict, which
#: ``state_types.as_impedance_robot_state`` normalizes to plain arrays.
JacobianFn = Callable[[np.ndarray], np.ndarray]

#: Dual-ascent budget for the single-row CBF QP. Fixed here rather than
#: exposed as config knobs: with exactly one constraint row the outer sweep
#: has nothing to coordinate, so one pass of a well-resolved root-find is the
#: whole solve. Matches solve_constrained_box_qp's own documented scoping
#: ("a SMALL number of constraints").
_DUAL_SWEEPS = 2
_DUAL_ROOT_ITERS = 24


@dataclass
class ManipulabilityCBFResult:
    """Everything one CBF cycle produced, for the torque path AND the trace."""

    #: The filtered torque. Exactly ``tau_nominal`` (same object contents,
    #: no QP round-off) whenever the constraint is inactive -- see
    #: ``manipulability_cbf_filter``'s short-circuit.
    tau: np.ndarray
    #: True when the CBF row was actually binding at ``tau_nominal`` and the
    #: QP therefore ran and changed the torque.
    active: bool
    #: mu(q) this cycle.
    manipulability: float
    #: h = mu - eps. Negative means already inside the singular region; the
    #: HOCBF then drives h back up rather than merely holding it.
    h: float
    #: hdot = grad_mu . qd -- negative means moving TOWARD the singularity.
    h_dot: float
    #: ||grad_mu||, the barrier's kinematic authority. Near zero means the QP
    #: has no direction to push in and the row is skipped (reported inactive).
    grad_norm: float
    #: qd^T H_mu qd, the directional curvature term of hddot.
    curvature: float
    #: How much slack the constraint had at ``tau_nominal``: ``b - A @ tau``.
    #: Positive = satisfied without help, negative = the row was violated and
    #: the QP had to intervene.
    slack_at_nominal: float
    #: False when the constraint cannot be met inside the torque box --
    #: reported, never silently clamped (this is the real diagnostic value of
    #: a hard constraint over a soft penalty).
    feasible: bool
    #: ||tau_cbf - tau_nominal||, how far the filter moved the command.
    delta_norm: float


def manipulability(jacobian: np.ndarray) -> float:
    """Yoshikawa manipulability ``mu = prod_i sigma_i`` of a task Jacobian.

    Equal to ``sqrt(det(J J^T))``, and for a square ``J`` to ``|det J|`` -- but
    computed from the singular values directly (see the module docstring for
    why: no cancellation, non-negative by construction).
    """
    jac = np.asarray(jacobian, dtype=np.float64)
    sigma = np.linalg.svd(jac, compute_uv=False)
    return float(np.prod(sigma))


def manipulability_gradient(
    jacobian_fn: JacobianFn,
    q: np.ndarray,
    *,
    step: float = 1.0e-5,
) -> np.ndarray:
    """``grad_mu(q)`` in R^6, by CENTRAL finite differences.

    Central rather than forward: the error is O(step^2) instead of O(step),
    which matters because ``mu`` is small in absolute terms on this arm
    (~1e-3 near the transport poses) while its gradient is not (~1e-2 per rad),
    so a one-sided difference's truncation error is not negligible relative to
    the signal. Costs 12 ``jacobian_fn`` evaluations.
    """
    q = np.asarray(q, dtype=np.float64).reshape(-1)
    n = int(q.shape[0])
    delta = float(step)
    if not np.isfinite(delta) or delta <= 0.0:
        raise ValueError(f"manipulability_gradient requires step > 0; got {step!r}")
    grad = np.zeros(n, dtype=np.float64)
    for i in range(n):
        q_plus = q.copy()
        q_minus = q.copy()
        q_plus[i] += delta
        q_minus[i] -= delta
        grad[i] = (manipulability(jacobian_fn(q_plus)) - manipulability(jacobian_fn(q_minus))) / (
            2.0 * delta
        )
    return grad


def manipulability_directional_curvature(
    jacobian_fn: JacobianFn,
    q: np.ndarray,
    qd: np.ndarray,
    *,
    step: float = 1.0e-4,
) -> float:
    """``qd^T H_mu(q) qd``, the only piece of the Hessian ``hddot`` needs.

    Computed as a directional second derivative along the UNIT vector
    ``u = qd/||qd||`` and then rescaled by ``||qd||^2``:

        qd^T H_mu qd = ||qd||^2 * d^2/ds^2 mu(q + s u) |_{s=0}
                     ~= ||qd||^2 * [ mu(q+du) - 2 mu(q) + mu(q-du) ] / d^2

    Two extra ``jacobian_fn`` evaluations, versus ~36 to build the full 6x6
    Hessian and contract it -- and the full Hessian is never needed, since
    ``hddot``'s second-order term is exactly this one scalar. Normalizing the
    perturbation direction (rather than stepping along raw ``qd``) keeps the
    finite-difference step size independent of how fast the arm happens to be
    moving, which is what keeps the second difference out of the cancellation
    regime at low speed.

    Exactly 0.0 at ``qd == 0`` (returned without any extra evaluation) -- the
    quadratic form is identically zero there, not merely small.
    """
    q = np.asarray(q, dtype=np.float64).reshape(-1)
    qd = np.asarray(qd, dtype=np.float64).reshape(-1)
    speed = float(np.linalg.norm(qd))
    if speed <= 0.0:
        return 0.0
    delta = float(step)
    if not np.isfinite(delta) or delta <= 0.0:
        raise ValueError(
            f"manipulability_directional_curvature requires step > 0; got {step!r}"
        )
    unit = qd / speed
    mu_0 = manipulability(jacobian_fn(q))
    mu_p = manipulability(jacobian_fn(q + delta * unit))
    mu_m = manipulability(jacobian_fn(q - delta * unit))
    second_derivative = (mu_p - 2.0 * mu_0 + mu_m) / (delta * delta)
    return float(speed * speed * second_derivative)


def manipulability_cbf_constraint_row(
    *,
    grad_mu: np.ndarray,
    m_inv: np.ndarray,
    bias: np.ndarray,
    qd: np.ndarray,
    mu: float,
    curvature: float,
    epsilon: float,
    alpha1: float,
    alpha2: float,
) -> tuple[np.ndarray, float]:
    """Build ``(A_row, b_scalar)`` for ``A_row @ tau <= b_scalar``.

    Step 4 of the module docstring's derivation, verbatim::

        A = -( grad_mu^T M^-1 )
        b = -grad_mu^T M^-1 bias + qd^T H_mu qd
            + (alpha1 + alpha2) (grad_mu . qd)
            + alpha1 * alpha2 * (mu - epsilon)

    Kept as a standalone pure function precisely so the sign convention and
    the algebra can be unit-tested against a hand-computed case without
    instantiating a controller or a simulator.
    """
    grad_mu = np.asarray(grad_mu, dtype=np.float64).reshape(-1)
    m_inv = np.asarray(m_inv, dtype=np.float64)
    bias = np.asarray(bias, dtype=np.float64).reshape(-1)
    qd = np.asarray(qd, dtype=np.float64).reshape(-1)
    lie = grad_mu @ m_inv  # g^T M^-1, the row that multiplies tau
    h_val = float(mu) - float(epsilon)
    h_dot = float(grad_mu @ qd)
    a_row = -lie
    b_scalar = (
        -float(lie @ bias)
        + float(curvature)
        + (float(alpha1) + float(alpha2)) * h_dot
        + float(alpha1) * float(alpha2) * h_val
    )
    return a_row.reshape(1, -1), float(b_scalar)


def manipulability_cbf_filter(
    *,
    tau_nominal: np.ndarray,
    jacobian: np.ndarray,
    jacobian_fn: JacobianFn,
    q: np.ndarray,
    qd: np.ndarray,
    m_inv: np.ndarray,
    bias: np.ndarray,
    tau_lower: np.ndarray,
    tau_upper: np.ndarray,
    epsilon: float,
    alpha1: float,
    alpha2: float,
    fd_step: float = 1.0e-5,
    curvature_step: float = 1.0e-4,
) -> ManipulabilityCBFResult:
    """One cycle of the OSCBF-style singularity-avoidance QP filter.

    ``jacobian`` is the already-available current-``q`` Jacobian (reused for
    ``mu`` so the caller does not pay for a 15th evaluation); ``jacobian_fn``
    is only used for the finite differences.

    SHORT-CIRCUIT (this is a correctness property, not just an optimization):
    when the constraint row is already satisfied at ``tau_nominal``, the
    QP's solution IS ``tau_nominal`` (the objective's unconstrained minimizer,
    and it is inside the box by construction of the caller's backtracking), so
    this returns ``tau_nominal`` untouched rather than a solver approximation
    of it. That makes "far from any singularity => exact no-op" an exact
    statement rather than a 1e-9 one.
    """
    tau_nominal = np.asarray(tau_nominal, dtype=np.float64).reshape(-1)
    n = int(tau_nominal.shape[0])
    q = np.asarray(q, dtype=np.float64).reshape(-1)
    qd = np.asarray(qd, dtype=np.float64).reshape(-1)
    m_inv = np.asarray(m_inv, dtype=np.float64).reshape(n, n)
    bias = np.asarray(bias, dtype=np.float64).reshape(n)
    tau_lower = np.asarray(tau_lower, dtype=np.float64).reshape(n)
    tau_upper = np.asarray(tau_upper, dtype=np.float64).reshape(n)

    mu = manipulability(jacobian)
    grad_mu = manipulability_gradient(jacobian_fn, q, step=fd_step)
    grad_norm = float(np.linalg.norm(grad_mu))
    h_val = float(mu) - float(epsilon)
    h_dot = float(grad_mu @ qd)

    if not np.isfinite(grad_norm) or grad_norm <= 0.0:
        # No usable barrier direction (a genuine plateau, or a non-finite
        # finite-difference result). Nothing to constrain against -- report
        # inactive rather than emit a degenerate 0 @ tau <= b row, which is
        # either vacuous or unsatisfiable-by-construction.
        return ManipulabilityCBFResult(
            tau=tau_nominal,
            active=False,
            manipulability=float(mu),
            h=h_val,
            h_dot=h_dot,
            grad_norm=grad_norm,
            curvature=0.0,
            slack_at_nominal=float("inf"),
            feasible=True,
            delta_norm=0.0,
        )

    curvature = manipulability_directional_curvature(
        jacobian_fn, q, qd, step=curvature_step
    )
    a_ineq, b_ineq = manipulability_cbf_constraint_row(
        grad_mu=grad_mu,
        m_inv=m_inv,
        bias=bias,
        qd=qd,
        mu=mu,
        curvature=curvature,
        epsilon=epsilon,
        alpha1=alpha1,
        alpha2=alpha2,
    )
    slack = float(b_ineq - float(a_ineq[0] @ tau_nominal))
    if slack >= 0.0:
        return ManipulabilityCBFResult(
            tau=tau_nominal,
            active=False,
            manipulability=float(mu),
            h=h_val,
            h_dot=h_dot,
            grad_norm=grad_norm,
            curvature=float(curvature),
            slack_at_nominal=slack,
            feasible=True,
            delta_norm=0.0,
        )

    # min ||tau - tau_nominal||^2 -- the shared weighted-least-squares builder,
    # one term, unit weight, no Tikhonov (the identity Hessian is already
    # perfectly conditioned).
    hessian, linear = build_weighted_least_squares_qp(
        [(np.eye(n, dtype=np.float64), tau_nominal, 1.0)], reg=0.0, n=n
    )
    tau_cbf, _dual, feasible = solve_constrained_box_qp(
        hessian,
        linear,
        tau_lower,
        tau_upper,
        a_ineq,
        np.array([b_ineq], dtype=np.float64),
        dual_sweeps=_DUAL_SWEEPS,
        dual_root_iters=_DUAL_ROOT_ITERS,
    )
    return ManipulabilityCBFResult(
        tau=np.asarray(tau_cbf, dtype=np.float64).reshape(n),
        active=True,
        manipulability=float(mu),
        h=h_val,
        h_dot=h_dot,
        grad_norm=grad_norm,
        curvature=float(curvature),
        slack_at_nominal=slack,
        feasible=bool(feasible),
        delta_norm=float(np.linalg.norm(tau_cbf - tau_nominal)),
    )
