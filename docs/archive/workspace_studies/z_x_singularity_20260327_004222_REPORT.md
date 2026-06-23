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
  "wall_time_s": 3691.7655029296875,
  "z_floor_world_m": 0.0,
  "z_max_feasible_m": 1.0799999999999996,
  "z_grid_m": [
    0.2,
    0.21098591549295775,
    0.2219718309859155,
    0.23295774647887324,
    0.24394366197183098,
    0.2549295774647887,
    0.26591549295774647,
    0.2769014084507042,
    0.28788732394366195,
    0.2988732394366197,
    0.30985915492957744,
    0.3208450704225352,
    0.33183098591549287,
    0.34281690140845067,
    0.35380281690140836,
    0.36478873239436616,
    0.37577464788732384,
    0.38676056338028164,
    0.39774647887323933,
    0.4087323943661971,
    0.4197183098591548,
    0.4307042253521126,
    0.4416901408450703,
    0.45267605633802804,
    0.4636619718309858,
    0.47464788732394353,
    0.48563380281690127,
    0.496619718309859,
    0.5076056338028168,
    0.5185915492957744,
    0.5295774647887322,
    0.54056338028169,
    0.5515492957746477,
    0.5625352112676054,
    0.5735211267605632,
    0.584507042253521,
    0.5954929577464787,
    0.6064788732394364,
    0.6174647887323942,
    0.628450704225352,
    0.6394366197183097,
    0.6504225352112674,
    0.6614084507042252,
    0.672394366197183,
    0.6833802816901406,
    0.6943661971830983,
    0.705352112676056,
    0.7163380281690139,
    0.7273239436619716,
    0.7383098591549293,
    0.749295774647887,
    0.7602816901408449,
    0.7712676056338026,
    0.7822535211267603,
    0.793239436619718,
    0.8042253521126759,
    0.8152112676056336,
    0.8261971830985912,
    0.8371830985915489,
    0.8481690140845068,
    0.8591549295774645,
    0.8701408450704222,
    0.8811267605633799,
    0.8921126760563378,
    0.9030985915492955,
    0.9140845070422532,
    0.9250704225352109,
    0.9360563380281686,
    0.9470422535211265,
    0.9580281690140842,
    0.9690140845070419,
    0.9799999999999996
  ],
  "z_min_requested": 0.2,
  "z_below_max_margin_m": 0.1,
  "shoulder_pan_fixed_rad": 0.0,
  "z_samples": 72,
  "x_samples": 21,
  "slsqp_seeds_boundary": 18,
  "slsqp_maxiter": 500,
  "ik_seeds_per_x": 14,
  "ik_merge_min_sep_rad": 0.12,
  "cond_singular_threshold": 10000.0,
  "sigma_singular_tol": 0.02,
  "x_gain_lock_threshold": 0.0001,
  "multi_solution_min": 2,
  "jobs": 4,
  "seed": 0,
  "verbose": true,
  "progress_bar": true
}

## Global aggregates (feasible IK samples only)

{
  "n_z_slices": 72,
  "n_feasible_ik_samples": 1497,
  "fraction_singular_jacobian": 1.0,
  "fraction_x_tangent_lock": 0.0,
  "fraction_multi_branch_ik": 0.9652638610554443,
  "median_cond_J_pose_row_normalized": 1e+16,
  "p90_cond_J_pose_row_normalized": 1e+16,
  "median_x_tangent_gain_m_per_rad": 0.2744778827429409,
  "median_n_distinct_solutions": 5.0,
  "max_n_distinct_solutions": 12,
  "median_x_span_m": 1.242205558954273,
  "max_x_span_m": 1.7759170424147852
}

## Per-Z summary

| Z (m) | slice_ok | x_span (m) | n_X | feas IK | frac singular | frac X-lock | frac multi-branch | max clusters |
|------:|:--------:|-----------:|----:|--------:|---------------:|------------:|------------------:|-------------:|
| 0.2000 | yes | 1.5770 | 21 | 21 | 1.00 | 0.00 | 1.00 | 10 |
| 0.2110 | yes | 1.7245 | 21 | 21 | 1.00 | 0.00 | 1.00 | 8 |
| 0.2220 | yes | 1.6579 | 21 | 21 | 1.00 | 0.00 | 1.00 | 7 |
| 0.2330 | yes | 1.0579 | 21 | 21 | 1.00 | 0.00 | 1.00 | 9 |
| 0.2439 | yes | 1.6320 | 21 | 21 | 1.00 | 0.00 | 1.00 | 9 |
| 0.2549 | yes | 1.6572 | 21 | 21 | 1.00 | 0.00 | 1.00 | 8 |
| 0.2659 | yes | 1.5990 | 21 | 20 | 1.00 | 0.00 | 1.00 | 10 |
| 0.2769 | yes | 1.3612 | 21 | 21 | 1.00 | 0.00 | 1.00 | 9 |
| 0.2879 | yes | 1.0246 | 21 | 21 | 1.00 | 0.00 | 1.00 | 11 |
| 0.2989 | yes | 1.4714 | 21 | 21 | 1.00 | 0.00 | 1.00 | 9 |
| 0.3099 | yes | 1.2704 | 21 | 21 | 1.00 | 0.00 | 1.00 | 10 |
| 0.3208 | yes | 0.7488 | 21 | 21 | 1.00 | 0.00 | 0.95 | 12 |
| 0.3318 | yes | 1.1430 | 21 | 21 | 1.00 | 0.00 | 0.95 | 9 |
| 0.3428 | yes | 1.3668 | 21 | 21 | 1.00 | 0.00 | 0.95 | 11 |
| 0.3538 | yes | 1.4667 | 21 | 20 | 1.00 | 0.00 | 1.00 | 9 |
| 0.3648 | yes | 1.7759 | 21 | 21 | 1.00 | 0.00 | 1.00 | 11 |
| 0.3758 | yes | 1.2781 | 21 | 21 | 1.00 | 0.00 | 1.00 | 10 |
| 0.3868 | yes | 1.6125 | 21 | 21 | 1.00 | 0.00 | 1.00 | 10 |
| 0.3977 | yes | 1.6680 | 21 | 21 | 1.00 | 0.00 | 1.00 | 7 |
| 0.4087 | yes | 1.6046 | 21 | 21 | 1.00 | 0.00 | 1.00 | 9 |
| 0.4197 | yes | 1.1108 | 21 | 21 | 1.00 | 0.00 | 1.00 | 11 |
| 0.4307 | yes | 1.4441 | 21 | 21 | 1.00 | 0.00 | 1.00 | 9 |
| 0.4417 | yes | 1.6260 | 21 | 21 | 1.00 | 0.00 | 0.86 | 8 |
| 0.4527 | yes | 1.6161 | 21 | 21 | 1.00 | 0.00 | 0.95 | 9 |
| 0.4637 | yes | 1.6605 | 21 | 20 | 1.00 | 0.00 | 1.00 | 10 |
| 0.4746 | yes | 1.5888 | 21 | 21 | 1.00 | 0.00 | 0.95 | 10 |
| 0.4856 | yes | 1.4631 | 21 | 21 | 1.00 | 0.00 | 0.95 | 9 |
| 0.4966 | yes | 1.2877 | 21 | 21 | 1.00 | 0.00 | 0.95 | 10 |
| 0.5076 | yes | 1.5860 | 21 | 21 | 1.00 | 0.00 | 1.00 | 9 |
| 0.5186 | yes | 1.1098 | 21 | 21 | 1.00 | 0.00 | 1.00 | 10 |
| 0.5296 | yes | 1.3219 | 21 | 21 | 1.00 | 0.00 | 1.00 | 10 |
| 0.5406 | yes | 1.6463 | 21 | 19 | 1.00 | 0.00 | 0.95 | 9 |
| 0.5515 | yes | 1.6259 | 21 | 21 | 1.00 | 0.00 | 0.90 | 8 |
| 0.5625 | yes | 1.5049 | 21 | 21 | 1.00 | 0.00 | 1.00 | 9 |
| 0.5735 | yes | 1.4007 | 21 | 21 | 1.00 | 0.00 | 0.95 | 7 |
| 0.5845 | yes | 1.4800 | 21 | 21 | 1.00 | 0.00 | 0.90 | 8 |
| 0.5955 | yes | 1.0097 | 21 | 21 | 1.00 | 0.00 | 1.00 | 9 |
| 0.6065 | yes | 1.1176 | 21 | 21 | 1.00 | 0.00 | 1.00 | 9 |
| 0.6175 | yes | 1.2972 | 21 | 21 | 1.00 | 0.00 | 1.00 | 9 |
| 0.6285 | yes | 1.0977 | 21 | 21 | 1.00 | 0.00 | 1.00 | 10 |
| 0.6394 | yes | 0.3452 | 21 | 20 | 1.00 | 0.00 | 0.70 | 11 |
| 0.6504 | yes | 1.2751 | 21 | 21 | 1.00 | 0.00 | 1.00 | 10 |
| 0.6614 | yes | 1.3967 | 21 | 21 | 1.00 | 0.00 | 1.00 | 10 |
| 0.6724 | yes | 1.3159 | 21 | 20 | 1.00 | 0.00 | 0.90 | 10 |
| 0.6834 | yes | 1.2317 | 21 | 21 | 1.00 | 0.00 | 0.95 | 10 |
| 0.6944 | yes | 1.0171 | 21 | 21 | 1.00 | 0.00 | 0.95 | 9 |
| 0.7054 | yes | 1.0606 | 21 | 21 | 1.00 | 0.00 | 1.00 | 10 |
| 0.7163 | yes | 1.1942 | 21 | 20 | 1.00 | 0.00 | 1.00 | 9 |
| 0.7273 | yes | 0.8384 | 21 | 20 | 1.00 | 0.00 | 0.95 | 7 |
| 0.7383 | yes | 1.1494 | 21 | 21 | 1.00 | 0.00 | 1.00 | 11 |
| 0.7493 | yes | 1.1481 | 21 | 21 | 1.00 | 0.00 | 0.90 | 9 |
| 0.7603 | yes | 0.9160 | 21 | 21 | 1.00 | 0.00 | 1.00 | 8 |
| 0.7713 | yes | 1.2527 | 21 | 21 | 1.00 | 0.00 | 0.86 | 9 |
| 0.7823 | yes | 1.2647 | 21 | 21 | 1.00 | 0.00 | 1.00 | 12 |
| 0.7932 | yes | 1.0785 | 21 | 21 | 1.00 | 0.00 | 0.95 | 10 |
| 0.8042 | yes | 0.9496 | 21 | 21 | 1.00 | 0.00 | 1.00 | 8 |
| 0.8152 | yes | 1.1644 | 21 | 21 | 1.00 | 0.00 | 0.95 | 8 |
| 0.8262 | yes | 1.1119 | 21 | 20 | 1.00 | 0.00 | 1.00 | 8 |
| 0.8372 | yes | 0.6282 | 21 | 21 | 1.00 | 0.00 | 1.00 | 9 |
| 0.8482 | yes | 0.8379 | 21 | 21 | 1.00 | 0.00 | 1.00 | 9 |
| 0.8592 | yes | 0.7694 | 21 | 21 | 1.00 | 0.00 | 1.00 | 10 |
| 0.8701 | yes | 1.1275 | 21 | 21 | 1.00 | 0.00 | 0.90 | 8 |
| 0.8811 | yes | 1.0234 | 21 | 21 | 1.00 | 0.00 | 0.90 | 7 |
| 0.8921 | yes | 0.8434 | 21 | 21 | 1.00 | 0.00 | 0.86 | 9 |
| 0.9031 | yes | 0.7912 | 21 | 21 | 1.00 | 0.00 | 1.00 | 8 |
| 0.9141 | yes | 0.7487 | 21 | 21 | 1.00 | 0.00 | 0.95 | 8 |
| 0.9251 | yes | 0.4129 | 21 | 20 | 1.00 | 0.00 | 0.85 | 8 |
| 0.9361 | yes | 0.3219 | 21 | 21 | 1.00 | 0.00 | 0.90 | 7 |
| 0.9470 | yes | 0.3206 | 21 | 20 | 1.00 | 0.00 | 0.95 | 9 |
| 0.9580 | yes | 0.4335 | 21 | 21 | 1.00 | 0.00 | 1.00 | 8 |
| 0.9690 | yes | 0.4757 | 21 | 20 | 1.00 | 0.00 | 1.00 | 7 |
| 0.9800 | yes | 0.7787 | 21 | 19 | 1.00 | 0.00 | 0.79 | 7 |

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
