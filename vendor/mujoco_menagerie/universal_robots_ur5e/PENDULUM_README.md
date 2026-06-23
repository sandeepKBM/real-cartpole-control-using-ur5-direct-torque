# Pendulum Attachment — How It Works

## What is `attachment_site`?

```xml
<site name="attachment_site" pos="0 0.1 0" quat="-1 1 0 0"/>
```

A **site** in MuJoCo is a named coordinate frame (position + orientation) used for:
- **Attachment points** — where to mount tools, grippers, or payloads
- **Sensors** — force/torque, position references
- **Visualization** — markers, debugging

`attachment_site` is **not** a physical body or geom. It defines:
- **`pos="0 0.1 0"`** — 0.1 m along the local +Y of `wrist_3_link` (the flange tip)
- **`quat="-1 1 0 0"`** — orientation aligned with the tool flange (Z-axis = tool approach direction)

To attach something, create a **body** as a child of `wrist_3_link` with the same `pos` and `quat` so it sits at the site.

---

## Overview

The pendulum is attached to the UR5e end effector (`wrist_3_link`) via a **non-actuated revolute (hinge) joint**. It extends perpendicular to the flange surface and swings freely.

---

## Coordinate System & Orientation

### UR5e wrist_3_link frame

- **Origin:** At the wrist_3 joint
- **Tip:** At `pos="0 0.1 0"` — 0.1 m along local **+Y**
- **Attachment site:** Same location, with `quat="-1 1 0 0"` (aligns with tool flange)

### Why the pendulum was invisible before

1. **Wrong axis:** The pole extended along local **+Y**. When the EE pointed down, +Y pointed into the floor → pole went underground.
2. **Too thin:** Radius 0.015 m (1.5 cm) was hard to see.
3. **No flange alignment:** Without the attachment_site quaternion, the pole didn’t follow the flange orientation.

### Fix: align with flange, extend along tool axis

- **`quat="-1 1 0 0"`** — Matches `attachment_site` so the pendulum frame matches the tool flange.
- **`fromto="0 0 0 0 0 0.4"`** — Pole along local **+Z** (tool axis), 0.4 m long.
- **`size="0.03"`** — Radius 3 cm for visibility.
- **`rgba="1 0.2 0.1 1"`** — Bright red-orange.

---

## MJCF Structure

```
wrist_3_link (EE)
└── pendulum_base [pos 0 0.1 0, quat -1 1 0 0]
    ├── pendulum_hinge [axis 1 0 0, damping 0.01]  ← revolute, no actuator
    ├── inertial [mass 0.08, COM at 0 0 0.2]
    └── pole [capsule, 0→0.4 along Z, radius 0.03]
```

### Key parameters

| Parameter | Value | Meaning |
|-----------|-------|---------|
| `pos="0 0.1 0"` | At EE tip | Attachment point |
| `quat="-1 1 0 0"` | Flange alignment | Pole along tool axis |
| `axis="1 0 0"` | Hinge axis X | Pole swings in YZ plane |
| `fromto="0 0 0 0 0 0.4"` | 0.4 m along Z | Pole length |
| `size="0.03"` | 3 cm radius | Visibility |
| `damping="0.01"` | Light damping | Slight energy loss |

---

## Changing the pendulum

Edit `scene_ur5e_cartpole.xml`:

- **Length:** Change `0.4` in `fromto`
- **Thickness:** Change `size="0.03"`
- **Color:** Change `rgba`
- **Mass:** Change `mass` in `<inertial>`
- **Swing plane:** Change `axis` (e.g. `0 1 0` for XZ plane)
