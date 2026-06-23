# Z-sweep: X freedom, IK branches, and Jacobian singularities

## Task definition

- **End-effector:** `attachment_site` in world frame.
- **Fixed:** tool orientation `TARGET_SITE_ROTATION_WORLD`, `shoulder_pan_joint` at configured value, joint limits.
- **Z band:** uniform grid from `z_min` up to `z_max_feasible - z_below_max` (feasible max from same constrained optimization as the envelope probe).
- **X freedom per Z:** `x_span_m = x_max - x_min` from SLSQP boundary solves (Y free).
- **Along X:** for each sample on `[x_min, x_max]`, multi-start IK solves **5 equality constraints** (3 rotvec + world X + world Z) in **5** joint variables (shoulder lift … wrist 3). Distinct postures are **clustered** by joint-space distance.
- **Singularity proxy:** SVD of the **row-normalized** 5×5 Jacobian of (rotvec, x, z) w.r.t. shoulder lift…wrist 3 (mixed units otherwise distort `cond`).
- **X dexterity on the slice:** magnitude of world-X row of Jacobian projected onto the **null space of [J_rot; J_z]** (first-order motion with orientation and Z held). Small values mean X is locally poorly conditioned along the constrained manifold.

## Run metadata

{
  "scene_xml": "/common/users/ss5772/real_Cartpole/mujoco_menagerie/universal_robots_ur5e/scene_ur5e_cartpole.xml",
  "tool_site": "attachment_site",
  "wall_time_s": 6.741692304611206,
  "z_floor_world_m": 0.0,
  "z_max_feasible_m": 1.0799999999999996,
  "z_grid_m": [
    0.2,
    0.5899999999999999,
    0.9799999999999996
  ],
  "z_min_requested": 0.2,
  "z_below_max_margin_m": 0.1,
  "shoulder_pan_fixed_rad": 0.0,
  "z_samples": 3,
  "x_samples": 2,
  "slsqp_seeds_boundary": 4,
  "slsqp_maxiter": 280,
  "ik_seeds_per_x": 2,
  "ik_merge_min_sep_rad": 0.12,
  "cond_singular_threshold": 10000.0,
  "sigma_singular_tol": 0.02,
  "x_gain_lock_threshold": 0.0001,
  "multi_solution_min": 2,
  "jobs": 2,
  "seed": 0
}

## Global aggregates (feasible IK samples only)

{
  "n_z_slices": 3,
  "n_feasible_ik_samples": 1,
  "fraction_singular_jacobian": 1.0,
  "fraction_x_tangent_lock": 0.0,
  "fraction_multi_branch_ik": 0.0,
  "median_cond_J_pose_row_normalized": 1e+16,
  "p90_cond_J_pose_row_normalized": 1e+16,
  "median_x_tangent_gain_m_per_rad": 0.17880099754110595,
  "median_n_distinct_solutions": 1.0,
  "max_n_distinct_solutions": 1,
  "median_x_span_m": 0.6144890354866408,
  "max_x_span_m": 0.6144890354866408
}

## Per-Z summary

| Z (m) | slice_ok | x_span (m) | n_X | feas IK | frac singular | frac X-lock | frac multi-branch | max clusters |
|------:|:--------:|-----------:|----:|--------:|---------------:|------------:|------------------:|-------------:|
| 0.2000 | yes | 0.6145 | 2 | 1 | 1.00 | 0.00 | 0.00 | 1 |
| 0.5900 | no |  | 0 | 0 |  |  |  | 0 |
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

## Figures

`matplotlib` was not available; PNG figures were skipped. Install matplotlib in the active environment and re-run to generate plots.
