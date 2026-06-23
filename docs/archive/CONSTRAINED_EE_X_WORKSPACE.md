# Constrained end-effector X workspace (arm only, shoulder-side face upright)

This summarizes **kinematic limits** for **`attachment_site` world X** when:

- the UR5e **tool orientation** matches `TARGET_SITE_ROTATION_WORLD` (same “shoulder side facing upright” law as `split_forearm_origin_face_controller`);
- **`shoulder_pan_joint` is fixed** at `0` (same policy as the active controller);
- **joint limits** come from the loaded MJCF (`ur5e.xml` via `scene_ur5e_cartpole.xml`);
- **no cartpole** is considered.

Numbers below were produced by [`simulation/study_constrained_ee_x_workspace.py`](/common/users/ss5772/real_Cartpole/simulation/study_constrained_ee_x_workspace.py) and saved to [`outputs/control_runs/constrained_ee_x_workspace.json`](/common/users/ss5772/real_Cartpole/outputs/control_runs/constrained_ee_x_workspace.json). Re-run the script after any model or target-orientation change.

**Dependency:** `scipy` (for SLSQP). The `mujoco_ur5e` conda env used in this repo was given `pip install scipy` for this study.

## Global translation range along world X

Under the **exact** orientation equality constraint and joint bounds, multi-start SLSQP gives:

| Quantity | Value (representative run) |
|----------|----------------------------|
| Min world X of `attachment_site` | **−0.890** m |
| Max world X | **+0.917** m |
| Span | **≈ 1.807** m |

So, in principle, **about 1.81 m** of world-X travel remains while keeping the tool frame fixed in orientation and pan locked, subject to this MuJoCo model and **no self-collision constraints** in the optimizer (only joint limits). If you add collision avoidance, the feasible span will shrink.

## “How much X per degree of each limb?” — correct interpretation

There is **no** single answer of the form “joint *i* alone may move X by *k* mm per degree” while orientation stays locked, because **orientation is three scalar constraints** and feasible motion is a **curve in joint space**; several joints must move together.

Two legitimate quantities:

### 1. Raw Jacobian entry (orientation **not** held)

At the canonical pose `ACTIVE_ORIGIN_Q`, **∂x/∂q_i** (world X vs. joint *i*, others instantaneously fixed) is useful for **unconstrained** intuition only. Example magnitudes from a representative run:

| Joint | Approx. mm per degree (raw) |
|-------|-----------------------------|
| shoulder_pan | +4.1 (pan is locked in policy, not in this column) |
| shoulder_lift | **−16.0** |
| elbow | **−8.6** |
| wrist_1 | −1.7 |
| wrist_2 | +1.7 |
| wrist_3 | 0.0 |

Negative sign means **increasing** that joint angle **decreases** world X at this pose, for an infinitesimal move with other joints held.

These numbers **do not** respect the orientation constraint; they answer “which joint most nudges X if nothing else moves.”

### 2. Feasible instantaneous X under orientation lock (first order)

With **angular velocity of the tool held to zero** (linearization: `jacr @ qdot = 0`) and **pan rate zero**, the **largest** world-X velocity for **unit** Euclidean \(\|\dot q_{1:5}\|\) (rad/s on shoulder lift through wrist 3) equals \(\| P \nabla x \|\), where \(P\) projects onto the null space of the site rotational Jacobian.

Representative value: **≈ 0.724 m/s per unit \(\|\dot q_{1:5}\|\)** (i.e. if the five joint speeds have combined magnitude 1 rad/s in the best direction for X, world X changes at ~0.72 m/s at this pose).

The corresponding **unit joint mix** (shoulder lift, elbow, wrist_1, wrist_2, wrist_3) from the same run looked like:

```text
[ -0.746, -0.159, +0.383, ~0, +0.521 ]
```

So at the canonical pose, **shoulder lift**, **elbow**, **wrist_1**, and **wrist_3** participate in the dominant feasible X direction; **wrist_2** is nearly inactive in that particular null-space direction (numerically ~0).

## Relation to control

- The **global span (~1.8 m)** is the **kinematic envelope** for X under this task.
- The **raw mm/deg** table informs which joints are strong **local levers** for X if you briefly ignore orientation (e.g. outer-loop shapers).
- The **projected gradient** informs **coordinated** motion that respects **tool orientation to first order**, consistent with keeping the shoulder-side face upright.

## Further statistical questions

Relating **limb angles** to **X motion** across many poses needs **defined metrics** (Jacobian row, constrained range, etc.) and **stratified or partial** statistics—simple correlation across coupled joint samples is easy to misread. Extend this study by batch-sampling `q`, logging the same quantities, and aggregating offline.

## Reproduce

```bash
cd /common/users/ss5772/real_Cartpole
source /common/users/ss5772/miniforge3/etc/profile.d/conda.sh
conda activate mujoco_ur5e
MUJOCO_GL=egl python simulation/study_constrained_ee_x_workspace.py
```
