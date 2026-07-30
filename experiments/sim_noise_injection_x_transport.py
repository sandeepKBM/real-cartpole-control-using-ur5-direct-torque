"""Sim noise-injection X-transport backtest: reproduce the real-hardware
noise-driven spurious-trip problem entirely in MuJoCo (no robot needed), using
clean, controlled, repeatable synthetic sensor noise instead of the 2 real
data points from the previous rounds. Validates both fixes that landed since
``docs/status/safety_envelope_backtest_2026-07-30.md`` SS1-8:

  1. ``hardware/safety.py::CartesianMoveMonitor``'s new DeadlineMonitor-style
     graduated tolerance (``accel_max_consecutive_violations``/
     ``accel_hard_multiple``/``speed_max_consecutive_violations``/
     ``speed_hard_multiple``, commit 8eccd1d on
     ``feature/ur5e-mujoco-torque-control``, merged into this worktree via
     ``git rebase origin/feature/ur5e-mujoco-torque-control`` before this
     script was written -- the REAL current class is imported and used
     unmodified, never hand-reimplemented).
  2. This project's own ``QdGrowthRateCandidate`` growth-rate metric (added in
     the previous round, ``experiments/safety_envelope_backtest.py``),
     re-tested here against synthetic noise it was never fit to.

Real noise numbers used below (measured, NOT re-derived or guessed) come from
``hardware_captures/2026-07-28_thinkrobot_172.16.71.77/
stationary_noise_capture_154018_stats.json`` (10s / 4730-sample real
stationary RTDE capture, 500 Hz, robot 172.16.71.77):
``tcp_pos_std_m = [8.88e-06, 9.86e-06, 3.25e-06]`` (X, Y, Z),
``q_std_rad_max = 1.695e-05``. **Caveat, explicit, not silently assumed away**:
that capture's ``qd_std_radps`` is exactly 0.0 -- RTDE's ``qd`` estimate has an
internal deadband at rest, so this capture cannot characterize real ``qd``
noise DURING motion, only at rest. This script therefore injects NO synthetic
``qd`` noise anywhere and reports true (physics-computed, unperturbed) ``qd``
into every safety check -- any conclusion below about ``|qd|``-based checks
(the ``qd_max_radps`` guard, and Candidate C's own ``|qd|``-growth metric)
inherits this caveat and is not evidence about how those checks would behave
under real motion-time ``qd`` sensor noise, only position/orientation-derived
noise.

Noise convention (mirrors ``rl_gain_scheduling/gain_scheduling_env.py``'s own
established rule): noise perturbs only a COPY of the state that is READ by a
safety-monitor check, never the true physics state fed to ``mj_step`` or the
real controller. The rollout physics/controller loop below is the same real
one this whole project already runs (``simulation/ur5e_mujoco_torque.py``,
``config/ur5e_mujoco_torque_osc_tuned.yaml``) -- no simplified physics
stand-in, no reimplementation of ``CartesianMoveMonitor``'s math.

Run with the mujoco_ur5e conda env (needs pinocchio, since the tuned config
uses ``gravity_source: pinocchio``):
    /common/users/ss5772/miniforge3/envs/mujoco_ur5e/bin/python3 \
        experiments/sim_noise_injection_x_transport.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import mujoco  # noqa: E402

from hardware.safety import CartesianMoveLimits, CartesianMoveMonitor  # noqa: E402
from simulation.ur5e_mujoco_torque import (  # noqa: E402
    apply_start_q,
    build_initial_state_and_adapter,
    build_mujoco_state,
    load_model,
    resolve_start_q,
    x_profile_target,
)

sys.path.insert(0, str(REPO_ROOT / "experiments"))
from safety_envelope_backtest import QdGrowthRateCandidate  # noqa: E402

# Real measured stationary-noise numbers -- see module docstring. Do not
# re-derive or adjust these; they come verbatim from
# hardware_captures/2026-07-28_thinkrobot_172.16.71.77/
# stationary_noise_capture_154018_stats.json.
TCP_POS_STD_M = np.array([8.88e-06, 9.86e-06, 3.25e-06], dtype=np.float64)
Q_STD_RAD_MAX = 1.695e-05

CONFIG_PATH = REPO_ROOT / "config" / "ur5e_mujoco_torque_osc_tuned.yaml"


# ---------------------------------------------------------------------------
# Physics rollout (deterministic -- no noise anywhere in the physics/controller
# loop). Real controller, real MuJoCo stepping, same as
# tools/ur5e_mujoco_torque_experiments.py's own per-step loop.
# ---------------------------------------------------------------------------


def run_true_trajectory(
    *, target_x_delta: float, move_duration_s: float, hold_duration_s: float,
) -> list[dict[str, Any]]:
    """Runs ONE real, deterministic MuJoCo rollout and returns the true
    per-cycle state trace (q, qd, ee_pos, orientation_error_norm, time_s).
    Noise is injected later, in a separate pass, onto a COPY of this trace --
    never into this rollout itself."""
    cfg = yaml.safe_load(CONFIG_PATH.read_text())
    mujoco_cfg = cfg["mujoco"]
    ctrl_cfg = cfg["controller"]
    scene_xml = REPO_ROOT / mujoco_cfg["scene_xml"]

    model, data, site_id, joint_ids, actuator_ids = load_model(scene_xml)
    gravity_scratch = mujoco.MjData(model)

    # Apply the config's home_qpos (the real active-origin transport start pose,
    # itself deliberately at the wrist_2=0 singularity per this config's own
    # header comments) BEFORE building state -- exactly mirroring
    # tools/ur5e_mujoco_torque_experiments.py's run()'s own init sequence.
    # Without this, MuJoCo's raw default qpos (all zeros) is used instead, an
    # even more degenerate all-joints-zero configuration with no numerical
    # jitter to ever perturb it off exactly singular -- confirmed by hand to
    # freeze the controller at tau[shoulder_pan]=0.0 bit-exact for the entire
    # rollout (zero real motion), the exact "global cond(J)-based singular_scale
    # nulls task authority at the transport start pose" issue AGENTS.md SS3
    # already documents, just triggered at the wrong (unintended) pose.
    start_q, _ = resolve_start_q(mujoco_cfg, None)
    if start_q is not None:
        apply_start_q(model, data, start_q)

    state0, adapter = build_initial_state_and_adapter(
        model, data, site_id, joint_ids,
        controller_cfg=ctrl_cfg,
        transport_axis_index=0,
        target_x_delta=float(target_x_delta),
        controller_kind=str(mujoco_cfg.get("default_controller", "impedance")),
        force_hold_current_pose=False,
        gravity_mode=str(mujoco_cfg.get("gravity_mode", "gravity_comp")),
        gravity_source=str(mujoco_cfg.get("gravity_source", "mujoco_qfrc")),
        coriolis_feedforward=bool(mujoco_cfg.get("coriolis_feedforward", False)),
        torque_limit_scale=1.0,
        gravity_scratch_data=gravity_scratch,
    )

    dt = float(model.opt.timestep)
    duration_s = float(move_duration_s) + float(hold_duration_s)
    steps = max(1, int(np.ceil(duration_s / dt)))
    x0 = float(state0.ee_pos[0])
    y0 = float(state0.ee_pos[1])
    z0 = float(state0.ee_pos[2])

    rows: list[dict[str, Any]] = []
    for _ in range(steps):
        t_s = float(data.time)
        target_x_now, target_x_vel_now = x_profile_target(
            "min_jerk_move_hold", x0, float(target_x_delta), t_s, duration_s,
            move_duration_s=float(move_duration_s),
        )
        target_ee_pos = np.array([target_x_now, y0, z0], dtype=np.float64)
        target_ee_vel = np.array([target_x_vel_now, 0.0, 0.0], dtype=np.float64)
        pre_state = build_mujoco_state(
            model, data, site_id=site_id, joint_ids=joint_ids,
            time_s=t_s, dt_s=dt,
            target_x=float(target_x_now), target_x_vel=float(target_x_vel_now),
            target_axis=float(target_x_now), target_axis_vel=float(target_x_vel_now),
            target_ee_pos=target_ee_pos, target_ee_vel=target_ee_vel,
            reference_quat=state0.reference_quat, hold_current_pose=state0.hold_current_pose,
            transport_axis_index=0,
            gravity_compensation=bool(mujoco_cfg.get("gravity_mode") == "gravity_comp"),
            gravity_scratch_data=gravity_scratch,
        )
        tau, diag = adapter.step(state=pre_state)
        data.ctrl[:6] = np.asarray(tau, dtype=np.float64).reshape(6)
        mujoco.mj_step(model, data)

        rows.append({
            "time_s": t_s,
            "q": np.asarray(pre_state.q, dtype=np.float64).copy(),
            "qd": np.asarray(pre_state.qd, dtype=np.float64).copy(),
            "ee_pos": np.asarray(pre_state.ee_pos, dtype=np.float64).copy(),
            "orientation_error_norm": float(diag.get("orientation_error_norm", 0.0)),
            "target_x": float(target_x_now),
            "sim_safety_ok": bool(diag.get("safety_ok", True)),
            "sim_safety_reason": diag.get("safety_reason"),
        })
    return rows, dt, y0, z0


# ---------------------------------------------------------------------------
# Noise injection + monitor replay (deterministic given a seed -- SAME noise
# draw per cycle is fed to all monitors under comparison, so differences
# between them are never an artifact of different random draws).
# ---------------------------------------------------------------------------


def _build_monitor(*, base_accel: float, base_speed: float, **overrides: Any) -> CartesianMoveMonitor:
    limits = CartesianMoveLimits(max_tcp_accel_mps2=base_accel, max_tcp_speed_mps=base_speed, **overrides)
    return CartesianMoveMonitor(limits)


def replay_with_noise(
    true_rows: list[dict[str, Any]], *, dt: float, y0: float, z0: float,
    move_duration_s: float, base_accel: float, base_speed: float, seed: int,
    pos_std: np.ndarray = TCP_POS_STD_M, q_std: float = Q_STD_RAD_MAX,
) -> dict[str, Any]:
    """Single noise draw per cycle (seeded), replayed through FOUR parallel
    monitors that all see the identical noisy input each cycle:
      - default:   accel/speed_max_consecutive_violations=1 (today's old
                   rigid single-cycle-trip behavior, hard_multiple irrelevant
                   at consecutive=1)
      - graduated: accel_max_consecutive_violations=3, accel_hard_multiple=5.0,
                   speed_max_consecutive_violations=3, speed_hard_multiple=5.0
                   (the new fix, commit 8eccd1d, values matching that
                   commit's own defaults) -- accel_gap_cycles/speed_lowpass_alpha
                   left at their own defaults (1, 1.0), i.e. testing the NEW
                   graduated-tolerance mechanism in isolation, not combined with
                   the older, separate noise-filtering mechanism.
      - graduated_filtered: same graduated settings PLUS accel_gap_cycles=5,
                   speed_lowpass_alpha=0.2 -- the older noise-robust filtering
                   mechanism (predates commit 8eccd1d), added here because the
                   isolated "graduated" result below shows it is NOT sufficient
                   alone at this noise magnitude (see report); this variant
                   checks whether combining both existing mechanisms succeeds
                   where either alone does not.
      - qdgrowth:  default CartesianMoveMonitor limits (consecutive=1, no
                   filtering), but max_tcp_accel_mps2/max_tcp_speed_mps mutated
                   every cycle by QdGrowthRateCandidate.thresholds() (same
                   mechanism as experiments/safety_envelope_backtest.py's
                   replay_candidate). NOTE: QdGrowthRateCandidate only ever
                   SHRINKS the threshold when a growth trend is detected -- for
                   pure IID sensor noise (growth_rate ~0, no real trend) it
                   returns scale=1.0, i.e. the SAME base threshold as default,
                   with the SAME single-cycle-trip rule. It was designed to
                   distinguish static-high-risk poses from actively-diverging
                   ones (see docs/status/safety_envelope_backtest_2026-07-30.md
                   SS8), not to add noise tolerance -- expect it to inherit
                   default's exact noise vulnerability, not fix it.
    """
    rng = np.random.default_rng(seed)

    row0 = true_rows[0]
    start_pose = np.concatenate([row0["ee_pos"], np.zeros(3)])

    monitor_default = _build_monitor(base_accel=base_accel, base_speed=base_speed)
    monitor_graduated = _build_monitor(
        base_accel=base_accel, base_speed=base_speed,
        accel_max_consecutive_violations=3, accel_hard_multiple=5.0,
        speed_max_consecutive_violations=3, speed_hard_multiple=5.0,
    )
    monitor_graduated_filtered = _build_monitor(
        base_accel=base_accel, base_speed=base_speed,
        accel_max_consecutive_violations=3, accel_hard_multiple=5.0,
        speed_max_consecutive_violations=3, speed_hard_multiple=5.0,
        accel_gap_cycles=5, speed_lowpass_alpha=0.2,
    )
    monitor_qdgrowth = _build_monitor(base_accel=base_accel, base_speed=base_speed)
    qdgrowth = QdGrowthRateCandidate()

    for m in (monitor_default, monitor_graduated, monitor_graduated_filtered, monitor_qdgrowth):
        m.set_start(start_pose, move_axis_index=0)

    result: dict[str, Any] = {
        "default": {"tripped": False, "cycle": None, "time_s": None, "reason": None},
        "graduated": {"tripped": False, "cycle": None, "time_s": None, "reason": None},
        "graduated_filtered": {"tripped": False, "cycle": None, "time_s": None, "reason": None},
        "qdgrowth": {"tripped": False, "cycle": None, "time_s": None, "reason": None},
    }

    for i, row in enumerate(true_rows):
        t_s = row["time_s"]
        true_ee_pos = row["ee_pos"]
        noisy_ee_pos = true_ee_pos + rng.normal(0.0, pos_std, size=3)
        noisy_q = row["q"] + rng.normal(0.0, q_std, size=6)
        true_qd = row["qd"]  # NOT noised -- see module docstring caveat

        noisy_tcp_pose = np.concatenate([noisy_ee_pos, np.zeros(3)])
        target_tcp_pose = np.array([row["target_x"], y0, z0, 0.0, 0.0, 0.0], dtype=np.float64)
        axis_target_moving = bool(t_s <= move_duration_s)

        common_kwargs = dict(
            q=noisy_q, qd=true_qd, tcp_pose=noisy_tcp_pose,
            target_tcp_pose=target_tcp_pose,
            orientation_error_rad=row["orientation_error_norm"],
            axis_target_moving=axis_target_moving, dt_s=dt,
        )

        if not result["default"]["tripped"]:
            d = monitor_default.check(**common_kwargs)
            if not d.ok:
                result["default"] = {"tripped": True, "cycle": i, "time_s": t_s, "reason": d.reason}

        if not result["graduated"]["tripped"]:
            d = monitor_graduated.check(**common_kwargs)
            if not d.ok:
                result["graduated"] = {"tripped": True, "cycle": i, "time_s": t_s, "reason": d.reason}

        if not result["graduated_filtered"]["tripped"]:
            d = monitor_graduated_filtered.check(**common_kwargs)
            if not d.ok:
                result["graduated_filtered"] = {"tripped": True, "cycle": i, "time_s": t_s, "reason": d.reason}

        if not result["qdgrowth"]["tripped"]:
            accel_thr, speed_thr, growth_rate = qdgrowth.thresholds(
                qd=true_qd, base_accel=base_accel, base_speed=base_speed,
            )
            monitor_qdgrowth.limits.max_tcp_accel_mps2 = accel_thr
            monitor_qdgrowth.limits.max_tcp_speed_mps = speed_thr
            d = monitor_qdgrowth.check(**common_kwargs)
            if not d.ok:
                result["qdgrowth"] = {"tripped": True, "cycle": i, "time_s": t_s, "reason": d.reason}

    return result


# ---------------------------------------------------------------------------
# Move profiles under test.
# ---------------------------------------------------------------------------

# (a) Canonical, already-validated grid point (config/ur5e_mujoco_torque_osc_tuned.yaml's
# own header: "canonical grid (dx 0.01-0.04m x hold 1/2s, move 1.0s...): 8/8 valid").
# Theoretical KINEMATIC min-jerk peak velocity/accel for dx=0.02m/T=1.0s:
# v_peak=1.875*0.02/1.0=0.0375 m/s, a_peak=5.7735*0.02/1.0^2=0.1155 m/s^2 -- both
# comfortably under base_accel/base_speed below.
#
# REAL, UNPLANNED FINDING (measured directly, not assumed): the ACHIEVED trajectory's
# own real closed-loop controller output has enough real, physical high-frequency
# content (torque-tracking chatter, not sensor noise) that the unfiltered
# (accel_gap_cycles=1) double-differenced accel ESTIMATE reaches up to ~3.6 m/s^2
# even with ZERO injected sensor noise -- ~31x the smooth kinematic theoretical
# peak, and enough to trip base_accel=0.5 on its own. This is a real, separate
# manifestation of the same "double-differencing amplifies high-frequency content
# by ~1/dt^2" mechanism this project's own code already documents for SENSOR
# noise (see CartesianMoveLimits.accel_gap_cycles docstring) -- it turns out to
# apply just as much to real controller output jitter. See the report for the
# full breakdown; this profile keeps base_accel=0.5 (the realistic tight/default
# value) specifically BECAUSE this controller-chatter effect is itself real,
# relevant evidence, not something to threshold away.
CANONICAL_PROFILE = dict(target_x_delta=0.02, move_duration_s=1.0, hold_duration_s=1.0,
                          base_accel=0.5, base_speed=0.05)

# (a-headroom) Same move, but base_accel/base_speed raised to have real, measured
# headroom over the noise-free ACHIEVED peak (accel ~3.62 m/s^2, speed ~0.127 m/s,
# both measured directly -- see report) -- isolates the SENSOR-noise-specific
# question ("does injected real-magnitude RTDE noise alone cause spurious trips,
# separate from real controller chatter?") that CANONICAL_PROFILE's tight default
# threshold cannot answer cleanly on its own, since that one already trips on
# chatter alone before any sensor noise is added.
CANONICAL_HEADROOM_PROFILE = dict(target_x_delta=0.02, move_duration_s=1.0, hold_duration_s=1.0,
                                   base_accel=4.5, base_speed=0.2)

# (b) Large-displacement case with real, intended min-jerk peak acceleration near
# (here, above) the guard threshold -- dx=-0.20m/T=1.0s is within this same config's
# own validated envelope ("large displacements (dx up to 0.20m, hold 1/2s): 16/16
# valid"). Theoretical min-jerk peak: a_peak = 5.7735 * |dx| / T^2 =
# 5.7735*0.20/1.0 = 1.1547 m/s^2 (matches the ~1.15 m/s^2 figure this project's own
# real-hardware notes reference for exactly this move). base_accel/base_speed here
# are picked to put that peak genuinely, moderately above the accel ceiling (as
# several real hardware_transport runs actually used 0.8 as their accel threshold)
# while the speed ceiling is loosened enough (0.5 m/s, matching this move's peak
# velocity of 1.875*0.20/1.0=0.375 m/s with headroom) that the test isolates the
# accel check specifically, not an unrelated speed trip.
LARGE_DISPLACEMENT_PROFILE = dict(target_x_delta=-0.20, move_duration_s=1.0, hold_duration_s=1.0,
                                   base_accel=0.8, base_speed=0.5)

N_SEEDS = 30


def summarize_profile(name: str, profile: dict[str, Any], n_seeds: int = N_SEEDS) -> dict[str, Any]:
    true_rows, dt, y0, z0 = run_true_trajectory(
        target_x_delta=profile["target_x_delta"],
        move_duration_s=profile["move_duration_s"],
        hold_duration_s=profile["hold_duration_s"],
    )
    peak_accel_theory = 5.7735 * abs(profile["target_x_delta"]) / (profile["move_duration_s"] ** 2)
    peak_speed_theory = 1.875 * abs(profile["target_x_delta"]) / profile["move_duration_s"]

    # Real ACHIEVED peak speed/accel (gap=1, unfiltered) from the noise-free true
    # trajectory -- measured directly, not theoretical -- see CANONICAL_PROFILE's
    # comment for why this can differ substantially from the smooth kinematic
    # theory above (real closed-loop controller chatter).
    _pos = np.array([r["ee_pos"] for r in true_rows])
    _speed = np.linalg.norm(np.diff(_pos, axis=0), axis=1) / (float(true_rows[1]["time_s"]) - float(true_rows[0]["time_s"]))
    _accel = np.abs(np.diff(_speed)) / (float(true_rows[1]["time_s"]) - float(true_rows[0]["time_s"]))
    peak_speed_achieved = float(_speed.max())
    peak_accel_achieved = float(_accel.max())

    # Noise-free ground truth: does this move genuinely trip at these
    # thresholds with ZERO injected noise? Establishes what "spurious" means
    # for this profile -- pos_std/q_std explicitly zeroed for this one call,
    # not a global mutation.
    noise_free = replay_with_noise(
        true_rows, dt=dt, y0=y0, z0=z0, move_duration_s=profile["move_duration_s"],
        base_accel=profile["base_accel"], base_speed=profile["base_speed"], seed=0,
        pos_std=np.zeros(3, dtype=np.float64), q_std=0.0,
    )

    per_seed: list[dict[str, Any]] = []
    for seed in range(n_seeds):
        r = replay_with_noise(
            true_rows, dt=dt, y0=y0, z0=z0, move_duration_s=profile["move_duration_s"],
            base_accel=profile["base_accel"], base_speed=profile["base_speed"], seed=seed,
        )
        per_seed.append(r)

    n_default_trip = sum(1 for r in per_seed if r["default"]["tripped"])
    n_graduated_trip = sum(1 for r in per_seed if r["graduated"]["tripped"])
    n_graduated_filtered_trip = sum(1 for r in per_seed if r["graduated_filtered"]["tripped"])
    n_qdgrowth_trip = sum(1 for r in per_seed if r["qdgrowth"]["tripped"])

    return {
        "name": name,
        "profile": profile,
        "peak_accel_theory_mps2": peak_accel_theory,
        "peak_speed_theory_mps": peak_speed_theory,
        "peak_accel_achieved_mps2": peak_accel_achieved,
        "peak_speed_achieved_mps": peak_speed_achieved,
        "n_steps": len(true_rows),
        "dt_s": dt,
        "noise_free_reference": noise_free,
        "n_seeds": n_seeds,
        "n_default_trip": n_default_trip,
        "n_graduated_trip": n_graduated_trip,
        "n_graduated_filtered_trip": n_graduated_filtered_trip,
        "n_qdgrowth_trip": n_qdgrowth_trip,
        "per_seed": per_seed,
    }


def main() -> None:
    results = {}
    for name, profile in (
        ("canonical_dx0.02_T1.0", CANONICAL_PROFILE),
        ("canonical_dx0.02_T1.0_headroom", CANONICAL_HEADROOM_PROFILE),
        ("large_displacement_dx-0.20_T1.0", LARGE_DISPLACEMENT_PROFILE),
    ):
        print(f"\n{'=' * 100}\nProfile: {name} -- {profile}")
        summary = summarize_profile(name, profile)
        results[name] = summary

        print(f"  theoretical (smooth kinematic) peak accel: {summary['peak_accel_theory_mps2']:.4f} m/s^2, "
              f"peak speed: {summary['peak_speed_theory_mps']:.4f} m/s")
        print(f"  ACHIEVED (real controller, gap=1, zero injected noise) peak accel: "
              f"{summary['peak_accel_achieved_mps2']:.4f} m/s^2, peak speed: "
              f"{summary['peak_speed_achieved_mps']:.4f} m/s "
              f"(base_accel={profile['base_accel']}, base_speed={profile['base_speed']})")
        print(f"  steps: {summary['n_steps']}, dt_s={summary['dt_s']}")
        nf = summary["noise_free_reference"]
        print(f"  NOISE-FREE reference (ground truth): "
              f"default={'TRIPS' if nf['default']['tripped'] else 'clean'} "
              f"({nf['default']['reason']})")
        print(f"  Over {summary['n_seeds']} noise seeds (real measured tcp_pos_std_m="
              f"{TCP_POS_STD_M.tolist()}, q_std_rad_max={Q_STD_RAD_MAX}):")
        print(f"    default (consecutive=1, old rigid):        "
              f"{summary['n_default_trip']}/{summary['n_seeds']} tripped")
        print(f"    graduated (consecutive=3, hard=5.0):       "
              f"{summary['n_graduated_trip']}/{summary['n_seeds']} tripped")
        print(f"    graduated_filtered (+gap=5, alpha=0.2):    "
              f"{summary['n_graduated_filtered_trip']}/{summary['n_seeds']} tripped")
        print(f"    qd_growth_rate (Candidate C):               "
              f"{summary['n_qdgrowth_trip']}/{summary['n_seeds']} tripped")
        # Show one example spurious-trip reason (if any) for the record, for
        # both the isolated graduated fix and the combined variant.
        example = next((r["default"] for r in summary["per_seed"] if r["default"]["tripped"]), None)
        if example is not None:
            print(f"  example default-trip reason: {example['reason']} at cycle {example['cycle']}, "
                  f"t={example['time_s']:.4f}s")
        example_g = next((r["graduated"] for r in summary["per_seed"] if r["graduated"]["tripped"]), None)
        if example_g is not None:
            print(f"  example graduated-trip reason: {example_g['reason']} at cycle {example_g['cycle']}, "
                  f"t={example_g['time_s']:.4f}s")

    out_path = REPO_ROOT / "experiments" / "sim_noise_injection_x_transport_results.json"

    def _default(o: Any) -> Any:
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        return str(o)

    out_path.write_text(json.dumps(results, indent=2, default=_default))
    print(f"\nFull results written to {out_path}")


if __name__ == "__main__":
    main()
