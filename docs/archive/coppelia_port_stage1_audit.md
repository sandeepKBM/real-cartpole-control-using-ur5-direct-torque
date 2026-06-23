# Stage 1 — MuJoCo UR5 / cartpole X-axis system (short audit)

No code behavior was changed for this audit. Findings refer to the current `real_Cartpole` tree.

## 1. Where the MuJoCo model / XML is loaded

- **Primary:** `mujoco.MjModel.from_xml_path(...)` in runner scripts under `simulation/`, e.g. `run_x_velocity_transport.py`, `run_x_acceleration_transport.py`, `run_fixed_z_x_transport.py`, `run_x_torque_transport_mujoco.py`.
- **Typical scene path:** `mujoco_menagerie/universal_robots_ur5e/scene_ur5e_cartpole.xml` (if present) else `.../scene.xml`.

## 2. Where `qpos` / `qvel` are read

- Each control / sim step: `data.qpos[: model.nu]` and `data.qvel[: model.nu]` (or full `qpos`/`qvel` when cartpole DoFs exist) in the same runners after `mujoco.mj_step` or before controller call, depending on script.

## 3. Where end-effector pose is computed

- **Forward kinematics from MuJoCo:** `data.site_xpos[tool_site_id]`, `data.site_xmat[tool_site_id]` after `mujoco.mj_forward` (or implicit in step).
- **Jacobians:** `mujoco.mj_jacSite(model, data, jacp, jacr, tool_site_id)`.

## 4. End-effector site / body name

- **Tool site:** `attachment_site` (see `TOOL_SITE_NAME` in runners and `ur5e.xml` site definitions).

## 5. How `target_x` is defined

- **Velocity transport:** explicit `x_goal` or segment waypoints from CLI / workspace JSON.
- **Acceleration transport:** motion implied by integrated velocity from outer-loop `a_x_cmd`; not a single fixed `target_x` in the same form.
- **Fixed-Z transport:** `run_fixed_z_x_transport.py` uses workspace / optimization outputs for X limits and targets.

## 6. Controller output type (position vs torque vs Cartesian)

- **`velocity_x_transport_controller` / `acceleration_x_transport_controller`:** output is **`data.ctrl`** — **joint position setpoints** for MuJoCo `general` actuators (position-servo style). Cartesian velocity/acceleration is solved internally (WLS + guards) then converted to **Δctrl**, not to torques as the final command.
- **Torque estimate only:** `SERVO_KP`, `SERVO_KD`, `SERVO_FORCE_LIMIT` in `simulation/controller.py` estimate implied servo torque for **feasibility scaling** and diagnostics — **not** the actuation command.

## 7. Gravity compensation

- **Legacy transport controllers:** no explicit gravity feedforward in the control law; gravity is handled implicitly by the MuJoCo position actuators + dynamics.
- **`run_x_torque_transport_mujoco.py`:** optional `--gravity-comp` adds `data.qfrc_bias` into the commanded torque for open-loop dynamics compensation in torque mode.

## 8. Where logging happens

- Runners append per-step dicts (time, pose, ctrl, diagnostics) and write **JSON** summaries under `outputs/control_runs/`. Some runners also write **MP4** under `demonstration_videos/ur5e_cartpole/`.

## 9. Where reset happens

- Runners set `data.qpos`, `data.qvel`, and usually `data.ctrl` to a start configuration (`ACTIVE_ORIGIN_Q`, workspace JSON `start_q`, etc.), then `mujoco.mj_forward`.

## 10. What the trace / summary contains

- Typical fields: time series of tool position, `x_goal` / velocity commands, `ctrl`, diagnostics (`tau_estimate_nm`, scales, etc.), and metadata (scene path, dt, CLI args). Exact keys vary by script; JSON summaries are the canonical artifact for analysis.

---

**Conclusion for CoppeliaSim port:** the production MuJoCo “transport” stack is **position-servo based** with Cartesian **task** in velocity form and **torque-like numbers only for guards/logging**. The CoppeliaSim port targets **explicit torque** with a **full Cartesian impedance** layer so Y/Z/orientation are actively regulated (see subsequent stages).
