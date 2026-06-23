# UR5e MJCF — End-to-End Explanation

This document walks through the MuJoCo model (MJCF) for the Universal Robots UR5e and how the scene files extend it.

---

## 1. File layout

| File | Role |
|------|------|
| **ur5e.xml** | Robot model: links, joints, actuators, assets |
| **scene.xml** | Scene: includes ur5e + floor, lights, visuals |
| **scene_ur5e_cartpole.xml** | Extended scene: includes scene.xml + pole at EE |

Include chain: `scene_ur5e_cartpole.xml` → `scene.xml` → `ur5e.xml`.

---

## 2. ur5e.xml — Top-level settings

### 2.1 Compiler

```xml
<compiler angle="radian" meshdir="assets" autolimits="true"/>
```

- **angle="radian"** — All joint angles and defaults in radians.
- **meshdir="assets"** — Meshes loaded from `assets/` (e.g. `base_0.obj`).
- **autolimits="true"** — Joint limits inferred from range if needed.

### 2.2 Option

```xml
<option integrator="implicitfast"/>
```

- **integrator="implicitfast"** — Fast implicit integrator for stable contact and stiff actuation.

---

## 3. Defaults (ur5e.xml)

Defaults define reusable templates so each joint/geom doesn’t repeat the same values.

### 3.1 Class `ur5e` (base)

- **material** — specular, shininess for all links.
- **joint** — default axis `0 1 0`, full range, armature 0.1.
- **general** (actuator) — gaintype fixed, biastype affine, gain/bias, forcerange -150..150.
- **visual** — geom class: mesh, no collision (`contype="0" conaffinity="0"`), group 2.
- **collision** — geom class: capsule, group 3; **eef_collision** subclass: cylinder.
- **site** — small size, group 4.

### 3.2 Class `size3` (big joints)

- Inherits ur5e; used for shoulder_pan, shoulder_lift, elbow.
- **size3_limited** — range ±π (elbow).

### 3.3 Class `size1` (wrist joints)

- **general** — weaker actuators: gainprm 500, forcerange -28..28.

So: base → size3 / size3_limited for arm, size1 for wrist.

---

## 4. Assets (ur5e.xml)

### 4.1 Materials

- **black**, **jointgray**, **linkgray**, **urblue** — used on meshes and geoms.

### 4.2 Meshes

- One or more `.obj` per link: base, shoulder, upperarm, forearm, wrist1, wrist2, wrist3.
- Referenced as e.g. `mesh="base_0"` in `<geom class="visual"/>`.

---

## 5. World and kinematic tree (ur5e.xml)

Everything is under `<worldbody>`. The robot is one chain of bodies.

### 5.1 Light

```xml
<light name="spotlight" mode="targetbodycom" target="wrist_2_link" pos="0 -1 2"/>
```

- Spotlight tracking the COM of `wrist_2_link`.

### 5.2 Body hierarchy (simplified)

```
worldbody
└── base  [quat 0 0 0 -1, childclass ur5e]
    └── shoulder_link     [pos 0 0 0.163]     joint: shoulder_pan_joint  axis Z
        └── upper_arm_link [pos 0 0.138 0, quat 1 0 1 0]  joint: shoulder_lift_joint  axis Y
            └── forearm_link [pos 0 -0.131 0.425]  joint: elbow_joint  axis Y, limited
                └── wrist_1_link [pos 0 0 0.392, quat 1 0 1 0]  joint: wrist_1_joint
                    └── wrist_2_link [pos 0 0.127 0]  joint: wrist_2_joint  axis Z
                        └── wrist_3_link [pos 0 0 0.1]  joint: wrist_3_joint
                            ├── visual geom (wrist3 mesh)
                            ├── eef_collision geom (cylinder)
                            └── site: attachment_site [pos 0 0.1 0, quat -1 1 0 0]
```

- **base** — Fixed to world; `quat="0 0 0 -1"` flips the robot into the desired world orientation.
- Each link has **inertial** (mass, com, inertia), **joint** (name, class, axis), **visual** geoms (meshes), and often **collision** geoms (capsules/cylinders).
- **attachment_site** — Site at the flange tip (0.1 m along wrist_3 +Y) with flange orientation; used as the reference for attaching tools (e.g. pole).

### 5.3 Joint axes (summary)

| Joint | Axis (local) | Notes |
|-------|----------------|------|
| shoulder_pan_joint | 0 0 1 (Z) | Base rotation |
| shoulder_lift_joint | 0 1 0 (Y) | Default class |
| elbow_joint | 0 1 0 (Y) | size3_limited ±π |
| wrist_1_joint | 0 1 0 (Y) | size1 |
| wrist_2_joint | 0 0 1 (Z) | size1 |
| wrist_3_joint | 0 1 0 (Y) | size1 |

### 5.4 End effector and attachment_site

- **wrist_3_link** — Last link (tool flange).
- **eef_collision** — Cylinder at `pos="0 0.08 0"` for the flange.
- **attachment_site** — `pos="0 0.1 0"` (tip), `quat="-1 1 0 0"` (flange frame). This is a **site** (frame only), not a body. To attach a tool, add a **body** under `wrist_3_link` with the same `pos` and `quat`.

---

## 6. Actuators (ur5e.xml)

```xml
<actuator>
  <general class="size3" name="shoulder_pan" joint="shoulder_pan_joint"/>
  <general class="size3" name="shoulder_lift" joint="shoulder_lift_joint"/>
  <general class="size3_limited" name="elbow" joint="elbow_joint"/>
  <general class="size1" name="wrist_1" joint="wrist_1_joint"/>
  <general class="size1" name="wrist_2" joint="wrist_2_joint"/>
  <general class="size1" name="wrist_3" joint="wrist_3_joint"/>
</actuator>
```

- **general** — Position/velocity-style actuators; dynamics come from the **general** defaults (gain/bias/forcerange) of the joint classes (size3, size3_limited, size1).
- Order of actuators matches **ctrl** and **qpos** order: 0=shoulder_pan, 1=shoulder_lift, 2=elbow, 3=wrist_1, 4=wrist_2, 5=wrist_3.

---

## 7. Keyframe (ur5e.xml)

```xml
<keyframe>
  <key name="home" qpos="-1.5708 -1.5708 1.5708 -1.5708 -1.5708 0" ctrl="-1.5708 -1.5708 1.5708 -1.5708 -1.5708 0"/>
</keyframe>
```

- **home** — Named pose: all joints at −π/2 except wrist_3 at 0 (L-shape, EE up). **qpos** sets state; **ctrl** sets default control targets.

---

## 8. scene.xml — Scene wrapper

- **Include** — `<include file="ur5e.xml"/>` pulls in the full robot and its defaults/assets.
- **statistic** — `center="0.3 0 0.4" extent="0.8"` for view/fit.
- **visual** — headlight, haze, global azimuth/elevation.
- **asset** — skybox texture, ground texture, ground material.
- **worldbody** — Adds a **floor** (plane geom) and a **light**. The robot’s worldbody content is merged with this (via include), so the final world has floor + lights + base and its tree.

---

## 9. scene_ur5e_cartpole.xml — Adding the pole

- **Include** — `<include file="scene.xml"/>` so we get scene + ur5e.
- **Body merge** — A second `<body name="wrist_3_link">` does not create a new body; MuJoCo **merges** it with the existing `wrist_3_link` from ur5e.xml. So children of this body are added to the existing wrist_3_link.

```xml
<body name="wrist_3_link">
  <body name="pole_attachment" pos="0 0.1 0" quat="-1 1 0 0">
    <inertial mass="0.3" pos="0 0 -0.25" diaginertia="..."/>
    <geom name="long_pole" type="capsule" fromto="0 0 0 0 0 -0.5" size="0.02" .../>
  </body>
</body>
```

- **pole_attachment** — Same **pos** and **quat** as **attachment_site**, so the pole is aligned with the flange frame.
- **long_pole** — Capsule from (0,0,0) to (0,0,-0.5) in that frame (50 cm along local −Z). No collision (`contype="0" conaffinity="0"`).

Result: a 50 cm pole rigidly attached at the UR5e end effector, without changing ur5e.xml.

---

## 10. End-to-end flow (what gets loaded)

1. **scene_ur5e_cartpole.xml** is loaded.
2. It includes **scene.xml**.
3. scene.xml includes **ur5e.xml** and adds floor, lights, visuals.
4. ur5e.xml defines compiler, option, defaults, assets, worldbody (base → … → wrist_3_link, attachment_site), actuators, keyframe.
5. Back in scene_ur5e_cartpole.xml, the extra `<body name="wrist_3_link">` is merged with the existing one, adding **pole_attachment** and **long_pole** under **wrist_3_link**.

So: **MJCF for the UR5e is the chain ur5e.xml → scene.xml → scene_ur5e_cartpole.xml**, with the pole added by body merge at the end effector.

---

## 11. Quick reference

| Concept | Where | What |
|--------|--------|------|
| Angles | compiler | radian |
| Integrator | option | implicitfast |
| Joint/actuator strength | default class size3 / size1 | gain, bias, forcerange |
| Link shapes | asset meshes + geom visual | OBJ meshes |
| Collision | geom class collision / eef_collision | capsules, cylinder |
| EE frame | site attachment_site | pos 0 0.1 0, quat -1 1 0 0 |
| Attaching a tool | New body under wrist_3_link | Same pos/quat as attachment_site |
| Control order | actuator order | 0..5 = shoulder_pan → wrist_3 |
