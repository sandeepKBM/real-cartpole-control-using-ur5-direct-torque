# UR5e Control Workflow

This repo no longer uses the older sway-demo or notebook-driven pose workflow as the primary path.

Current active path:

- define the peak-height origin from the MuJoCo model
- start from a seeded disturbed pose
- recover to the origin with the fixed-base reverse-nested controller
- save a rendered video and JSON metrics

## Primary Run

```bash
cd /common/users/ss5772/real_Cartpole
source /common/users/ss5772/miniforge3/etc/profile.d/conda.sh
conda activate mujoco_ur5e
xvfb-run -a python simulation/run_origin_stabilization.py --seed 7
```

## Primary Artifacts

Current reference outputs:

- [`ur5e_origin_stabilization_fixedbase_detailed_seed7.mp4`](/common/users/ss5772/real_Cartpole/demonstration_videos/ur5e_cartpole/ur5e_origin_stabilization_fixedbase_detailed_seed7.mp4)
- [`ur5e_origin_stabilization_fixedbase_detailed_seed7.json`](/common/users/ss5772/real_Cartpole/outputs/control_runs/ur5e_origin_stabilization_fixedbase_detailed_seed7.json)

## Active Files

- [`controller.py`](/common/users/ss5772/real_Cartpole/simulation/controller.py)
- [`run_origin_stabilization.py`](/common/users/ss5772/real_Cartpole/simulation/run_origin_stabilization.py)
- [`CONTROL_DESIGN_NOTEBOOK.md`](/common/users/ss5772/real_Cartpole/CONTROL_DESIGN_NOTEBOOK.md)

## Notes

- `shoulder_pan_joint` is currently locked to the origin orientation.
- The video overlay includes xyz, orientation, residual errors, and limb alignment angles.
- The ROS side is a control scaffold, not the main simulation workflow.
