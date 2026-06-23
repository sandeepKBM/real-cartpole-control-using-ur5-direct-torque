# Z-sweep: X freedom, IK branches, and Jacobian singularities

## Task definition

- **End-effector:** `attachment_site` in world frame.
- **Fixed:** tool orientation `TARGET_SITE_ROTATION_WORLD`, `shoulder_pan_joint` at configured value, joint limits.
- **Z band:** uniform grid from `z_min` up to `z_max_feasible - z_below_max` (feasible max from same constrained optimization as the envelope probe).
- **X freedom per Z:** `x_span_m = x_max - x_min` from SLSQP boundary solves (Y free).
- **Along X:** for each sample on `[x_min, x_max]`, multi-start IK solves **5 equality constraints** (3 rotvec + world X + world Z) in **5** joint variables (shoulder lift … wrist 3). Distinct postures are **clustered** by joint-space distance.
- **Singularity proxy:** `cond(J_pose)` for the 5×5 Jacobian of (rotvec, x, z) w.r.t. those joints.
- **X dexterity on the slice:** magnitude of world-X row of Jacobian projected onto the **null space of [J_rot; J_z]** (first-order motion with orientation and Z held). Small values mean X is locally poorly conditioned along the constrained manifold.

## Run metadata

{
  "scene_xml": "/common/users/ss5772/real_Cartpole/mujoco_menagerie/universal_robots_ur5e/scene_ur5e_cartpole.xml",
  "tool_site": "attachment_site",
  "wall_time_s": 249.8027982711792,
  "z_floor_world_m": 0.0,
  "z_max_feasible_m": 1.0799999999999996,
  "z_grid_m": [
    0.2,
    0.2709090909090909,
    0.34181818181818174,
    0.4127272727272726,
    0.4836363636363635,
    0.5545454545454543,
    0.6254545454545453,
    0.6963636363636361,
    0.7672727272727269,
    0.8381818181818177,
    0.9090909090909087,
    0.9799999999999996
  ],
  "z_min_requested": 0.2,
  "z_below_max_margin_m": 0.1,
  "shoulder_pan_fixed_rad": 0.0,
  "z_samples": 12,
  "x_samples": 5,
  "slsqp_seeds_boundary": 8,
  "ik_seeds_per_x": 6,
  "ik_merge_min_sep_rad": 0.12,
  "cond_singular_threshold": 10000.0,
  "x_gain_lock_threshold": 0.0001,
  "multi_solution_min": 2,
  "jobs": 1,
  "seed": 0
}

## Global aggregates (feasible IK samples only)

{
  "n_z_slices": 12,
  "n_feasible_ik_samples": 42,
  "fraction_singular_jacobian": 1.0,
  "fraction_x_tangent_lock": 0.0,
  "fraction_multi_branch_ik": 0.7857142857142857,
  "median_cond_J_pose": 1e+16,
  "p90_cond_J_pose": 1e+16,
  "median_x_tangent_gain_m_per_rad": 0.26685708538428443,
  "median_n_distinct_solutions": 3.0,
  "max_n_distinct_solutions": 6,
  "median_x_span_m": 0.9782857906640094,
  "max_x_span_m": 1.557051059082275
}

## Per-Z summary

| Z (m) | slice_ok | x_span (m) | n_X | feas IK | frac singular | frac X-lock | frac multi-branch | max clusters |
|------:|:--------:|-----------:|----:|--------:|---------------:|------------:|------------------:|-------------:|
| 0.2000 | yes | 0.6325 | 5 | 5 | 1.00 | 0.00 | 0.60 | 5 |
| 0.2709 | yes | 0.0410 | 5 | 4 | 1.00 | 0.00 | 0.25 | 3 |
| 0.3418 | yes | 1.5571 | 5 | 4 | 1.00 | 0.00 | 1.00 | 5 |
| 0.4127 | yes | 1.5191 | 5 | 4 | 1.00 | 0.00 | 0.75 | 5 |
| 0.4836 | yes | 1.3977 | 5 | 5 | 1.00 | 0.00 | 0.80 | 4 |
| 0.5545 | yes | 1.0867 | 5 | 4 | 1.00 | 0.00 | 1.00 | 4 |
| 0.6255 | yes | 1.1102 | 5 | 3 | 1.00 | 0.00 | 1.00 | 3 |
| 0.6964 | yes | 0.8698 | 5 | 3 | 1.00 | 0.00 | 1.00 | 5 |
| 0.7673 | no |  | 0 | 0 |  |  |  | 0 |
| 0.8382 | yes | 0.7571 | 5 | 5 | 1.00 | 0.00 | 0.60 | 6 |
| 0.9091 | yes | 0.0001 | 5 | 5 | 1.00 | 0.00 | 1.00 | 4 |
| 0.9800 | no |  | 0 | 0 |  |  |  | 0 |

## Files

- `z_x_singularity_study.json` — full numeric output.
- `figure_span_vs_z.png` — X span vs Z.
- `figure_heatmap_cond.png` — log10(cond J_pose) vs (Z, X).
- `figure_heatmap_clusters.png` — distinct IK cluster count vs (Z, X).
- `figure_heatmap_x_gain.png` — X tangent gain vs (Z, X).

## Interpretation notes

- **IK cluster count > 1** is empirical (depends on seeds / merge threshold); it indicates *likely* multiple disconnected branches, not a guaranteed exhaustive count.
- **High condition number** marks postures where small solver noise can flip behavior; combine with controller tests on hardware.
- Self-collision is **not** modeled; feasible sets shrink with collision constraints.
