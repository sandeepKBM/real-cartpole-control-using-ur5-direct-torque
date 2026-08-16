"""Task-space inertia (Lambda) shaping on the reduced-task QP controller.

Ordered by what breaks worst if wrong:

  1. Flag OFF is bit-identical to before the feature existed. Every shipped config
     leaves it off, and a live job importing this module must be unaffected.
  2. Lambda is built from the UN-EXCLUDED Jacobian. Building it from the
     column-zeroed one inverts a rank-deficient matrix; measured at ARM_Q0 with the
     default task_excluded_joints=(0,) that took cond() from 236 to 184195 and
     Lambda_zz from 2.73 to 9915, turning a sane wrench into numerical garbage.
  3. The Hessian's weights follow the wrench's domain, so H and the linear term
     describe the same task.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from controller_core.x_task_yz_corridor_qp import (  # noqa: E402
    XTaskYZCorridorQPConfig,
    XTaskYZCorridorQPController,
)
# Reuse the sibling suite's fixtures rather than re-inventing them: they already
# encode the required state keys ("time", mass_matrix, ...) and the torque-limit
# block the YAML parser demands.
from tests.unit.test_x_task_yz_corridor_qp import (  # noqa: E402
    make_state,
    make_config,
    _yaml_section,
)


def _cfg(**kw):
    return make_config(**kw)


def _run(cfg):
    c = XTaskYZCorridorQPController(cfg)
    c.reset_from_state(make_state())
    return c.compute(make_state())


# ===================== 1. DEFAULT PATH UNTOUCHED =========================

def test_shaping_defaults_off():
    assert XTaskYZCorridorQPConfig().task_space_inertia_shaping is False


def test_flag_off_is_bit_identical_to_unset():
    a = _run(_cfg())
    b = _run(_cfg(task_space_inertia_shaping=False))
    np.testing.assert_array_equal(a.tau, b.tau)
    np.testing.assert_array_equal(a.wrench_reduced, b.wrench_reduced)


# ============ 2. LAMBDA MUST NOT SEE THE EXCLUDED COLUMNS ================

def test_lambda_uses_unexcluded_jacobian_so_exclusion_cannot_blow_it_up():
    """Excluding a joint must not change the inertia MODEL.

    Lambda is a property of the mechanism (M, J), not of which joints we let act.
    If it were built from the column-zeroed Jacobian the wrench would explode as
    the excluded set grows; here the shaped wrench must stay finite and bounded.
    """
    shaped_none = _run(_cfg(task_space_inertia_shaping=True,
                            lambda_regularization=0.1,
                            task_excluded_joints=()))
    shaped_pan = _run(_cfg(task_space_inertia_shaping=True,
                           lambda_regularization=0.1,
                           task_excluded_joints=(0,)))
    for out in (shaped_none, shaped_pan):
        assert np.all(np.isfinite(out.wrench_reduced))
    # The wrench is the Lambda-shaped task wrench; excluding a joint changes which
    # torques may be applied, NOT the task inertia, so it must be identical.
    np.testing.assert_allclose(
        shaped_none.wrench_reduced, shaped_pan.wrench_reduced, rtol=1e-12, atol=1e-12
    )


def test_shaping_changes_the_wrench_when_on():
    off = _run(_cfg())
    on = _run(_cfg(task_space_inertia_shaping=True, lambda_regularization=0.1))
    assert not np.allclose(off.wrench_reduced, on.wrench_reduced)


# ===================== 3. CONFIG PLUMBING ================================

def test_parser_reads_the_lambda_keys():
    """These fields existed on the dataclass but the parser dropped them, so a
    YAML could never enable the feature at all."""
    c = XTaskYZCorridorQPConfig.from_controller_yaml_section(
        _yaml_section(task_space_inertia_shaping=True,
                      lambda_regularization=0.1,
                      lambda_diagonal_shaping=True)
    )
    assert c.task_space_inertia_shaping is True
    assert c.lambda_diagonal_shaping is True
    assert c.lambda_regularization == pytest.approx(0.1)


def test_parser_defaults_keep_shaping_off():
    c = XTaskYZCorridorQPConfig.from_controller_yaml_section(_yaml_section())
    assert c.task_space_inertia_shaping is False
    assert c.lambda_diagonal_shaping is False


@pytest.mark.parametrize("bad", [0.0, -1.0, float("nan"), float("inf")])
def test_parser_rejects_bad_lambda_regularization(bad):
    with pytest.raises(ValueError, match="lambda_regularization"):
        XTaskYZCorridorQPConfig.from_controller_yaml_section(
            _yaml_section(lambda_regularization=bad)
        )


def test_shaping_without_mass_matrix_raises_loudly():
    cfg = _cfg(task_space_inertia_shaping=True, lambda_regularization=0.1)
    c = XTaskYZCorridorQPController(cfg)
    c.reset_from_state(make_state())
    bad = make_state(mass_matrix=False)
    with pytest.raises(ValueError, match="mass_matrix"):
        c.compute(bad)


def test_diagonal_shaping_removes_cross_axis_terms():
    full = _run(_cfg(task_space_inertia_shaping=True, lambda_regularization=0.1,
                     lambda_diagonal_shaping=False))
    diag = _run(_cfg(task_space_inertia_shaping=True, lambda_regularization=0.1,
                     lambda_diagonal_shaping=True))
    assert not np.allclose(full.wrench_reduced, diag.wrench_reduced)
