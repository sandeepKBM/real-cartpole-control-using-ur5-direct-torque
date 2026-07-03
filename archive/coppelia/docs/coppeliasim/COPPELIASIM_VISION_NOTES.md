# CoppeliaSim vision + ZMQ (what the official API says)

This project ships the CoppeliaSim v4.10 **HTML manual** under `third_party/coppelia_runtime/.../manual/en/regularApi/`. The headless controller follows these rules.

## `sim.createVisionSensor` (options bitfield)

From `simCreateVisionSensor.htm` in the bundle:

- **Bit 0 (1):** sensor is **explicitly handled** (you drive when it is sensed).
- **Bit 1 (2):** perspective mode.
- **Bit 2 (4):** view frustum not shown in scene view.
- **Bit 7 (128):** use a specific null-pixel background color (floats 6–8 in `floatParams`); do not confuse “null” grey with an empty buffer.

`run_coppeliasim_x_axis_headless.py` uses **`1 | 2 | 4`** so the sensor is explicit at creation, then `sim.setExplicitHandling(handle, 1)` remains for clarity. This matches the need to call `handleVisionSensor` yourself in stepped control flows.

## `sim.handleVisionSensor` then `sim.getVisionSensorImg`

From `simGetVisionSensorImg.htm`:

> *“Reads the image of a vision sensor. The returned data doesn't make sense if sim.handleVisionSensor wasn't called previously”*

So the order in our runner is always:

1. **Set** vision sensor world pose (camera aim).
2. **Apply** torques and call **`sim.step()`** (with stepping enabled, each step is explicit).
3. **`sim.handleVisionSensor`**
4. **`sim.getVisionSensorImg`**

This matches the pattern in `run_coppeliasim_video_smoke.py` (pose → `step` → `handle` → `get`).

## Stepped simulation

`sim.setStepping(true)` (used by the remote client when you drive from Python) means **each** physics step is issued by `sim.step()`. Vision sensors that are **explicit** are not updated until `handleVisionSensor` runs, which is why the controller must not skip that call.

## ZMQ: image buffer type in Python

The return type of the first value from `getVisionSensorImg` can be **bytes** or, after CBOR decoding, a **list of integers**. The manual’s Lua side often refers to **`sim.unpackUInt8Table`**. In Python, failing to treat a **list** as raw pixels and instead passing it to **`np.frombuffer` incorrectly** can yield all-zero arrays. `decode_vision_image_buffer` in `run_coppeliasim_x_axis_headless.py` handles both `bytes` and `list` shapes.

## References (files in this repo)

| Topic | Path under `third_party/coppelia_runtime/.../manual/en/regularApi/` |
|--------|---------------------------------------------------------------------|
| Create vision sensor | `simCreateVisionSensor.htm` |
| Handle sensor | `simHandleVisionSensor.htm` |
| Read image | `simGetVisionSensorImg.htm` |
| Set stepping | `simSetStepping.htm` |

ZMQ client overview: `third_party/coppelia_runtime/.../programming/zmqRemoteApi/README.md`.
