local REAL_CARTPOLE_ROOT = os.getenv('REAL_CARTPOLE_ROOT') or '/common/users/ss5772/real_Cartpole'
local COPPELIA_ROOT = os.getenv('COPPELIA_ROOT') or (REAL_CARTPOLE_ROOT .. '/third_party/coppelia_runtime/CoppeliaSim_Edu_V4_10_0_rev0_Ubuntu24_04')
local SCENE_PATH = COPPELIA_ROOT .. '/system/dfltscn.ttt'
local MODEL_PATH = COPPELIA_ROOT .. '/models/robots/non-mobile/UR5.ttm'
local OUTPUT_DIR = os.getenv('OUTPUT_DIR') or (REAL_CARTPOLE_ROOT .. '/outputs/control_runs/coppelia_mujoco_like_y_torque_frames')
local STATE_DIR = os.getenv('STATE_DIR') or (REAL_CARTPOLE_ROOT .. '/outputs/control_runs/coppelia_mujoco_like_y_torque_state')
local VIDEO_PATH = os.getenv('VIDEO_PATH') or (REAL_CARTPOLE_ROOT .. '/demonstration_videos/ur5e_coppeliasim/coppeliasim_ur5_mujoco_like_y_torque.mp4')
local SUMMARY_PATH = os.getenv('SUMMARY_PATH') or (STATE_DIR .. '/coppeliasim_ur5_mujoco_like_y_torque_summary.json')
local FRAME_PREFIX = os.getenv('FRAME_PREFIX') or 'frame'
local LOAD_MARKER = STATE_DIR .. '/ur5_mujoco_like_y_torque_addon_loaded.txt'
local START_MARKER = STATE_DIR .. '/ur5_mujoco_like_y_torque_addon_started.txt'
local SENSING_MARKER = STATE_DIR .. '/ur5_mujoco_like_y_torque_addon_sensing.txt'
local DONE_MARKER = STATE_DIR .. '/ur5_mujoco_like_y_torque_done.txt'
local CONFIGURED_MARKER = STATE_DIR .. '/ur5_mujoco_like_y_torque_configured.txt'
local STEP_RELEASE_MARKER = os.getenv('REAL_CARTPOLE_LUA_STEP_RELEASE_FILE') or ''
local STEP_READY_MARKER = os.getenv('REAL_CARTPOLE_LUA_STEP_READY_FILE') or ''

local FPS = tonumber(os.getenv('FPS') or '25')
local FRAME_COUNT = tonumber(os.getenv('FRAME_COUNT') or '185')
local SETTLE_DURATION_S = tonumber(os.getenv('SETTLE_DURATION_S') or '1.0')
local function normalizeAccelDirection(raw)
    local s = string.lower(tostring(raw or ''))
    s = s:gsub('%s+', '')
    if s == '' then
        return 1.0
    end
    local n = tonumber(s)
    if n ~= nil then
        return n >= 0.0 and 1.0 or -1.0
    end
    if s == '1' or s == '+1' or s == 'y+' or s == '+y' or s == 'positive' or s == 'pos' then
        return 1.0
    end
    if s == '-1' or s == 'y-' or s == '-y' or s == 'negative' or s == 'neg' then
        return -1.0
    end
    return 1.0
end
local ACCEL_DIRECTION_ENV = os.getenv('ACCEL_DIRECTION')
local ACCEL_DIRECTION = normalizeAccelDirection(ACCEL_DIRECTION_ENV or os.getenv('Y_ACCEL_PRESET'))
local ACCEL_DIRECTION_SOURCE = os.getenv('ACCEL_DIRECTION_SOURCE') or (ACCEL_DIRECTION_ENV ~= nil and ACCEL_DIRECTION_ENV ~= '' and 'env_override' or 'internal_default')
local TRAVEL_DISTANCE_ENV = os.getenv('TRAVEL_DISTANCE_M')
local TRAVEL_DISTANCE_FALLBACK = os.getenv('TARGET_DX_M')
local TRAVEL_DISTANCE_SOURCE = os.getenv('TRAVEL_DISTANCE_SOURCE') or ((TRAVEL_DISTANCE_ENV ~= nil and TRAVEL_DISTANCE_ENV ~= '') and 'env_override' or ((TRAVEL_DISTANCE_FALLBACK ~= nil and TRAVEL_DISTANCE_FALLBACK ~= '') and 'compatibility_fallback_input' or 'internal_default'))
local TRAVEL_DISTANCE_M = math.abs(tonumber(TRAVEL_DISTANCE_ENV or TRAVEL_DISTANCE_FALLBACK or '0.35'))
local TARGET_DX = math.abs(TRAVEL_DISTANCE_M) * ACCEL_DIRECTION
local ACCEL_MAGNITUDE_ENV = os.getenv('ACCEL_MAGNITUDE_MPS2')
local ACCEL_MAGNITUDE_FALLBACK = os.getenv('A_AXIS_MAX_MPS2')
local ACCEL_MAGNITUDE_SOURCE = os.getenv('ACCEL_MAGNITUDE_SOURCE') or ((ACCEL_MAGNITUDE_ENV ~= nil and ACCEL_MAGNITUDE_ENV ~= '') and 'env_override' or ((ACCEL_MAGNITUDE_FALLBACK ~= nil and ACCEL_MAGNITUDE_FALLBACK ~= '') and 'compatibility_fallback_input' or 'internal_default'))
local A_AXIS_MAX = math.abs(tonumber(ACCEL_MAGNITUDE_ENV or ACCEL_MAGNITUDE_FALLBACK or '0.25'))
local V_AXIS_MAX = tonumber(os.getenv('V_AXIS_MAX_MPS') or '0.12')
local TRANSPORT_AXIS_RAW = string.lower(os.getenv('TRANSPORT_AXIS') or 'y')
local TASK_FRAME_MODE = string.lower(os.getenv('TASK_FRAME_MODE') or 'mujoco_attachment_dummy')
local TASK_ORIENTATION_TARGET = string.lower(os.getenv('TASK_ORIENTATION_TARGET') or 'initial')
local USE_EXTERNAL_STEP_PUMP = os.getenv('USE_EXTERNAL_STEP_PUMP') == '1'

local SHOW_EE_TRIAD = os.getenv('SHOW_EE_TRIAD')
if SHOW_EE_TRIAD == nil or SHOW_EE_TRIAD == '' then
    SHOW_EE_TRIAD = true
else
    SHOW_EE_TRIAD = SHOW_EE_TRIAD ~= '0'
end
local SHOW_BASE_TRIAD = os.getenv('SHOW_BASE_TRIAD')
if SHOW_BASE_TRIAD == nil or SHOW_BASE_TRIAD == '' then
    SHOW_BASE_TRIAD = true
else
    SHOW_BASE_TRIAD = SHOW_BASE_TRIAD ~= '0'
end

local POSITION_TOL_M = tonumber(os.getenv('POSITION_TOL_M') or '0.015')
local ORIENTATION_TOL_DEG = tonumber(os.getenv('ORIENTATION_TOL_DEG') or '3.0')
local MODEL_BASE_Z_OFFSET = tonumber(os.getenv('MODEL_BASE_Z_OFFSET_M') or '0.0')
local TASK_FRAME_LOCAL_Z_OFFSET = tonumber(os.getenv('TASK_FRAME_LOCAL_Z_OFFSET_M') or '-0.2')
local TRANSPORT_PLANE_Z_OFFSET = tonumber(os.getenv('TRANSPORT_PLANE_Z_OFFSET_M') or '0.0')

local EE_TRIAD_AXIS_LENGTH = tonumber(os.getenv('EE_TRIAD_AXIS_LENGTH_M') or '0.18')
local EE_TRIAD_LINE_WIDTH = tonumber(os.getenv('EE_TRIAD_LINE_WIDTH_PX') or '6')
local EE_TRIAD_DUMMY_SIZE = tonumber(os.getenv('EE_TRIAD_DUMMY_SIZE_M') or '0.03')
local EE_TRIAD_ROOT_OFFSET = tonumber(os.getenv('EE_TRIAD_ROOT_OFFSET_M') or '0.08')
local BASE_TRIAD_AXIS_LENGTH = tonumber(os.getenv('BASE_TRIAD_AXIS_LENGTH_M') or '0.16')
local BASE_TRIAD_LINE_WIDTH = tonumber(os.getenv('BASE_TRIAD_LINE_WIDTH_PX') or '6')
local BASE_TRIAD_DUMMY_SIZE = tonumber(os.getenv('BASE_TRIAD_DUMMY_SIZE_M') or '0.03')
local BASE_TRIAD_ROOT_OFFSET = tonumber(os.getenv('BASE_TRIAD_ROOT_OFFSET_M') or '0.08')

local JOINT_PATHS = {
    '/UR5/joint',
    '/UR5/link/joint',
    '/UR5/link/link/joint',
    '/UR5/link/link/link/joint',
    '/UR5/link/link/link/link/joint',
    '/UR5/link/link/link/link/link/joint',
}

local Q_ORIGIN = {
    0.0,
    -0.1133064268431449,
    -0.664621645801302,
    4.921777393344012,
    -6.283185307179586,
    5.280928640069786,
}
local Q_START = {
    0.0,
    -0.1133064268431449,
    -0.664621645801302,
    4.921777393344012,
    -6.283185307179586,
    5.280928640069786,
}
local Q_ORIGIN_SOURCE = 'coppelia_grounded_fixed_z_origin_default'
local Q_START_SOURCE = 'transport_plane_start_matches_origin'

local MODEL_TORQUE_LIMITS = {150.0, 150.0, 150.0, 28.0, 28.0, 28.0}
local INTERNAL_JOINT_INERTIA = {3.0, 3.0, 2.5, 0.8, 0.6, 0.5}
local INTERNAL_JOINT_DAMPING = {2.0, 2.0, 1.6, 0.5, 0.4, 0.35}

local KP_MOVE = 25.0
local KD_MOVE = 8.0
local KP_HOLD = 80.0
local KD_HOLD = 15.0
local KP_ROT = 20.0
local KD_ROT = 5.0
local KP_POSTURE = 2.0
local KD_POSTURE = 0.5
local KD_JOINT = 0.8
local TRACK_KP = {45.0, 45.0, 45.0, 15.0, 12.0, 10.0}
local TRACK_KD = {9.0, 9.0, 9.0, 3.0, 2.5, 2.5}

local TARGET_SITE_ROTATION_WORLD = {
    {-1.0, 0.0, 0.0},
    {0.0, 0.0, -1.0},
    {0.0, -1.0, 0.0},
}

local sim = require 'sim'
local joints = {}
local robotModelHandle = -1
local rawTaskHandle = -1
local taskHandle = -1
local visionSensor = -1

local eeTriadHandles = {}
local eeTriadDummyHandles = {}
local eeTriadShapeHandles = {}
local eeTriadRootHandle = -1
local baseTriadHandles = {}
local baseTriadDummyHandles = {}
local baseTriadShapeHandles = {}
local baseTriadRootHandle = -1
local taskFrameSummary = {mode = TASK_FRAME_MODE, mujoco_attachment_dummy = false}

local control = {
    initialized = false,
    startTime = 0.0,
    nextFrameTime = 0.0,
    framesCaptured = 0,
    internal_mode = false,
    internal_q = nil,
    internal_qd = nil,
    last_tau = nil,
    p0 = nil,
    quat0 = nil,
    targetQuat = nil,
    q0 = nil,
    q_start = nil,
    q_final = nil,
    p_start = nil,
    p_final = nil,
    peak_joint_speed = 0.0,
    peak_task_speed = 0.0,
    peak_axis_speed = 0.0,
    peak_tau = 0.0,
    max_orientation_error = 0.0,
    max_fixed_axis_1_drift = 0.0,
    max_fixed_axis_2_drift = 0.0,
    max_joint_excursion = 0.0,
    tau_sat_count = 0,
    tau_sat_samples = 0,
    safety_stop_reason = nil,
    joint_mode_summary = nil,
    camera_pose = nil,
    target_axis_start = 0.0,
    target_axis_final = 0.0,
    target_axis_net = 0.0,
    target_axis_goal = 0.0,
    target_axis_total = 0.0,
    target_axis_final_error = 0.0,
    target_axis_velocity_peak = 0.0,
    target_axis_accel_peak = 0.0,
    accel_direction = ACCEL_DIRECTION,
    accel_magnitude_mps2 = A_AXIS_MAX,
    travel_distance_m = math.abs(TRAVEL_DISTANCE_M),
    quat_final = nil,
    q_path = nil,
    qdot_path = nil,
    path_start_q = nil,
    path_end_q = nil,
    path_waypoints = 0,
    path_total_time = 0.0,
    path_plan_ok = false,
    path_first_failed_waypoint = -1,
    path_max_pos_err = 0.0,
    path_max_ori_err = 0.0,
    actuation_count = 0,
    sensing_count = 0,
    manual_loop_running = false,
    simulation_start_requested = false,
    first_actuation_time = nil,
    last_actuation_time = nil,
    first_sensing_time = nil,
    last_sensing_time = nil,
    first_frame_time = nil,
    last_frame_time = nil,
}

local function writeText(path, text)
    if path == nil or path == '' then
        return
    end
    local f = io.open(path, 'w')
    if not f then
        return
    end
    f:write(text or '')
    f:close()
end

local function fileExists(path)
    if path == nil or path == '' then
        return false
    end
    local f = io.open(path, 'r')
    if f then
        f:close()
        return true
    end
    return false
end

local function wallSleep(seconds)
    local s = tonumber(seconds) or 0.0
    if s <= 0.0 then
        return
    end
    local t0 = os.clock()
    while (os.clock() - t0) < s do
    end
end

local function jstr(s)
    s = tostring(s or '')
    s = s:gsub('\\', '\\\\'):gsub('"', '\\"')
    return '"' .. s .. '"'
end

local function jnum(v)
    if v == nil then
        return 'null'
    end
    if v ~= v or v == math.huge or v == -math.huge then
        return 'null'
    end
    return string.format('%.9g', v)
end

local function jbool(v)
    return tostring(v and true or false)
end

local function jarr(v)
    if v == nil then
        return 'null'
    end
    local out = {}
    for i = 1, #v do
        local item = v[i]
        local t = type(item)
        if t == 'number' then
            out[#out + 1] = jnum(item)
        elseif t == 'boolean' then
            out[#out + 1] = jbool(item)
        elseif t == 'table' then
            out[#out + 1] = jarr(item)
        else
            out[#out + 1] = 'null'
        end
    end
    return '[' .. table.concat(out, ',') .. ']'
end

local function clamp(x, lo, hi)
    if x < lo then return lo end
    if x > hi then return hi end
    return x
end

local function copyVec(v)
    local out = {}
    for i = 1, #v do
        out[i] = v[i]
    end
    return out
end

local function vec3(x, y, z)
    return {x, y, z}
end

local function cross(a, b)
    return {
        a[2] * b[3] - a[3] * b[2],
        a[3] * b[1] - a[1] * b[3],
        a[1] * b[2] - a[2] * b[1],
    }
end

local function dot(a, b)
    return a[1] * b[1] + a[2] * b[2] + a[3] * b[3]
end

local function norm(v)
    local s = 0.0
    for i = 1, #v do
        s = s + v[i] * v[i]
    end
    return math.sqrt(s)
end

local function normalize(v)
    local n = norm(v)
    if n < 1e-12 then
        return {0.0, 0.0, 0.0}
    end
    return {v[1] / n, v[2] / n, v[3] / n}
end

local function signWithZero(x)
    if x > 0.0 then
        return 1.0
    elseif x < 0.0 then
        return -1.0
    end
    return 0.0
end

local function axisNameToIndex(axis)
    local s = string.lower(tostring(axis or 'y'))
    if s == 'x' or s == '1' then
        return 1, 'x'
    elseif s == 'y' or s == '2' then
        return 2, 'y'
    elseif s == 'z' or s == '3' then
        return 3, 'z'
    end
    return 2, 'y'
end

local TRANSPORT_AXIS_INDEX, TRANSPORT_AXIS_LABEL = axisNameToIndex(TRANSPORT_AXIS_RAW)
local FIXED_AXIS_1, FIXED_AXIS_2 = 1, 3
if TRANSPORT_AXIS_INDEX == 1 then
    FIXED_AXIS_1, FIXED_AXIS_2 = 2, 3
elseif TRANSPORT_AXIS_INDEX == 2 then
    FIXED_AXIS_1, FIXED_AXIS_2 = 1, 3
else
    FIXED_AXIS_1, FIXED_AXIS_2 = 1, 2
end

local function quatNormalizeWxyz(q)
    local n = math.sqrt(q[1] * q[1] + q[2] * q[2] + q[3] * q[3] + q[4] * q[4])
    if n < 1e-12 then
        return {1.0, 0.0, 0.0, 0.0}
    end
    return {q[1] / n, q[2] / n, q[3] / n, q[4] / n}
end

local function quatConjWxyz(q)
    return {q[1], -q[2], -q[3], -q[4]}
end

local function quatMultiplyWxyz(a, b)
    local w1, x1, y1, z1 = a[1], a[2], a[3], a[4]
    local w2, x2, y2, z2 = b[1], b[2], b[3], b[4]
    return {
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    }
end

local function orientationErrorVecWxyz(quatDes, quatCur)
    local qd = quatNormalizeWxyz(quatDes)
    local qc = quatNormalizeWxyz(quatCur)
    local qErr = quatMultiplyWxyz(quatConjWxyz(qd), qc)
    qErr = quatNormalizeWxyz(qErr)
    if qErr[1] < 0.0 then
        qErr = {-qErr[1], -qErr[2], -qErr[3], -qErr[4]}
    end
    return {2.0 * qErr[2], 2.0 * qErr[3], 2.0 * qErr[4]}
end

local function rotmatToQuat(m)
    local r11, r12, r13 = m[1][1], m[1][2], m[1][3]
    local r21, r22, r23 = m[2][1], m[2][2], m[2][3]
    local r31, r32, r33 = m[3][1], m[3][2], m[3][3]
    local trace = r11 + r22 + r33
    local w, x, y, z
    if trace > 0.0 then
        local s = 2.0 * math.sqrt(trace + 1.0)
        w = 0.25 * s
        x = (r32 - r23) / s
        y = (r13 - r31) / s
        z = (r21 - r12) / s
    elseif r11 > r22 and r11 > r33 then
        local s = 2.0 * math.sqrt(1.0 + r11 - r22 - r33)
        w = (r32 - r23) / s
        x = 0.25 * s
        y = (r12 + r21) / s
        z = (r13 + r31) / s
    elseif r22 > r33 then
        local s = 2.0 * math.sqrt(1.0 + r22 - r11 - r33)
        w = (r13 - r31) / s
        x = (r12 + r21) / s
        y = 0.25 * s
        z = (r23 + r32) / s
    else
        local s = 2.0 * math.sqrt(1.0 + r33 - r11 - r22)
        w = (r21 - r12) / s
        x = (r13 + r31) / s
        y = (r23 + r32) / s
        z = 0.25 * s
    end
    return quatNormalizeWxyz({w, x, y, z})
end

local TARGET_SITE_ROTATION_WORLD_QUAT = rotmatToQuat(TARGET_SITE_ROTATION_WORLD)

local function solvePointToPointAccelReference(tMove, targetDx, aAbs, vAbs)
    local distance = math.abs(targetDx)
    if distance <= 0.0 then
        return 0.0, 0.0, 0.0, 0.0
    end
    local direction = targetDx >= 0.0 and 1.0 or -1.0
    local a = math.max(math.abs(aAbs), 1e-9)
    local vCap = math.max(math.abs(vAbs), 1e-9)
    local tAccelCap = vCap / a
    local dAccelCap = 0.5 * a * tAccelCap * tAccelCap
    local tAccel, vPeak, tFlat, dAccel
    if 2.0 * dAccelCap >= distance then
        tAccel = math.sqrt(distance / a)
        vPeak = a * tAccel
        tFlat = 0.0
        dAccel = 0.5 * a * tAccel * tAccel
    else
        tAccel = tAccelCap
        vPeak = vCap
        dAccel = dAccelCap
        tFlat = (distance - 2.0 * dAccel) / vPeak
    end
    local total = 2.0 * tAccel + tFlat
    local t = math.max(tMove, 0.0)
    local s, v, acc
    if t <= 0.0 then
        s, v, acc = 0.0, 0.0, a
    elseif t < tAccel then
        s = 0.5 * a * t * t
        v = a * t
        acc = a
    elseif t < tAccel + tFlat then
        local tau = t - tAccel
        s = dAccel + vPeak * tau
        v = vPeak
        acc = 0.0
    elseif t < total then
        local tau = total - t
        s = distance - 0.5 * a * tau * tau
        v = a * tau
        acc = -a
    else
        s, v, acc = distance, 0.0, 0.0
    end
    return direction * s, direction * v, direction * acc, total
end

local function readJointState()
    if control.internal_mode and control.internal_q ~= nil and control.internal_qd ~= nil then
        return copyVec(control.internal_q), copyVec(control.internal_qd)
    end
    local q, qd = {}, {}
    for i, h in ipairs(joints) do
        q[i] = sim.getJointPosition(h)
        qd[i] = sim.getJointVelocity(h)
    end
    return q, qd
end

local function setQ(q)
    for i, h in ipairs(joints) do
        sim.setJointPosition(h, q[i])
    end
end

local function configureForceTorqueMode()
    for _, h in ipairs(joints) do
        if sim.setJointMode ~= nil and sim.jointmode_dynamic ~= nil then
            pcall(sim.setJointMode, h, sim.jointmode_dynamic, 0)
        end
        if sim.setObjectInt32Param ~= nil and sim.jointintparam_motor_enabled ~= nil then
            pcall(sim.setObjectInt32Param, h, sim.jointintparam_motor_enabled, 1)
        end
        if sim.setObjectInt32Param ~= nil and sim.jointintparam_ctrl_enabled ~= nil then
            pcall(sim.setObjectInt32Param, h, sim.jointintparam_ctrl_enabled, 0)
        end
        pcall(function()
            if sim.setJointTargetForce ~= nil then
                sim.setJointTargetForce(h, 0.0, true)
            end
        end)
    end
end

local function applyTorque(tau)
    for i, h in ipairs(joints) do
        local t = tau[i] or 0.0
        -- CoppeliaSim responds more reliably to the motor-velocity + max-force
        -- pattern than to target-force alone on this UR5 build.
        local v0 = 10.0
        local vel = t >= 0.0 and v0 or -v0
        if sim.setJointTargetVelocity ~= nil then
            pcall(sim.setJointTargetVelocity, h, vel)
        end
        if sim.setJointMaxForce ~= nil then
            pcall(sim.setJointMaxForce, h, math.abs(t))
        end
        if sim.setJointTargetForce ~= nil then
            pcall(sim.setJointTargetForce, h, math.abs(t))
        end
    end
end

local function syncKinematics()
    if sim.forwardKinematic ~= nil then
        pcall(sim.forwardKinematic)
    end
    if sim.handleFk ~= nil then
        pcall(sim.handleFk)
    end
end

local function vectorMaxAbs(v)
    local m = 0.0
    for i = 1, #v do
        m = math.max(m, math.abs(v[i] or 0.0))
    end
    return m
end

local function maxAbsDiff(a, b)
    local m = 0.0
    local n = math.min(#a, #b)
    for i = 1, n do
        m = math.max(m, math.abs((a[i] or 0.0) - (b[i] or 0.0)))
    end
    return m
end

local function addVec3(a, b)
    return {a[1] + b[1], a[2] + b[2], a[3] + b[3]}
end

local function subVec3(a, b)
    return {a[1] - b[1], a[2] - b[2], a[3] - b[3]}
end

local function scaleVec3(v, s)
    return {v[1] * s, v[2] * s, v[3] * s}
end

local function matVecTransposeMul(J, w)
    local out = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0}
    local rows = math.min(#J, #w)
    for r = 1, rows do
        local wr = w[r] or 0.0
        local row = J[r]
        for c = 1, 6 do
            out[c] = out[c] + (row[c] or 0.0) * wr
        end
    end
    return out
end

local function clipTorques(tau, limit)
    local clipped = {}
    local saturated = {}
    for i = 1, 6 do
        local lo = -math.abs(limit[i] or 0.0)
        local hi = math.abs(limit[i] or 0.0)
        local raw = tau[i] or 0.0
        local c = clamp(raw, lo, hi)
        clipped[i] = c
        saturated[i] = math.abs(raw - c) > 1e-10
    end
    return clipped, saturated
end

local function readTaskState()
    local pose = sim.getObjectPose(taskHandle, sim.handle_world)
    local pos = {pose[1], pose[2], pose[3]}
    local quat = {pose[7], pose[4], pose[5], pose[6]}
    local lin, ang = sim.getObjectVelocity(taskHandle)
    return pos, quatNormalizeWxyz(quat), {lin[1], lin[2], lin[3]}, {ang[1], ang[2], ang[3]}
end

local function readJacobian()
    local q0 = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0}
    for i, h in ipairs(joints) do
        q0[i] = sim.getJointPosition(h)
    end
    local p0, quat0 = readTaskState()
    local eps = 1e-5
    local J = {}
    for r = 1, 6 do
        J[r] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0}
    end
    for i = 1, 6 do
        local qPert = copyVec(q0)
        qPert[i] = qPert[i] + eps
        setQ(qPert)
        if sim.forwardKinematic ~= nil then pcall(sim.forwardKinematic) end
        if sim.handleFk ~= nil then pcall(sim.handleFk) end
        if sim.computeJacobian ~= nil then pcall(sim.computeJacobian) end
        local p1, quat1 = readTaskState()
        J[1][i] = (p1[1] - p0[1]) / eps
        J[2][i] = (p1[2] - p0[2]) / eps
        J[3][i] = (p1[3] - p0[3]) / eps
        local eRot = orientationErrorVecWxyz(quat0, quat1)
        J[4][i] = eRot[1] / eps
        J[5][i] = eRot[2] / eps
        J[6][i] = eRot[3] / eps
    end
    setQ(q0)
    if sim.forwardKinematic ~= nil then pcall(sim.forwardKinematic) end
    if sim.handleFk ~= nil then pcall(sim.handleFk) end
    if sim.computeJacobian ~= nil then pcall(sim.computeJacobian) end
    return J
end

local function readJointConfigurationSummary()
    local jointsInfo = {}
    local motorVals, ctrlVals, dynVals, modeVals = {}, {}, {}, {}
    for i, h in ipairs(joints) do
        local entry = {handle = h, index = i}
        if sim.getObjectInt32Param ~= nil and sim.jointintparam_motor_enabled ~= nil then
            local ok, value = pcall(sim.getObjectInt32Param, h, sim.jointintparam_motor_enabled)
            if ok then
                entry.motor_enabled = tonumber(value) or 0
                motorVals[#motorVals + 1] = entry.motor_enabled
            end
        end
        if sim.getObjectInt32Param ~= nil and sim.jointintparam_ctrl_enabled ~= nil then
            local ok, value = pcall(sim.getObjectInt32Param, h, sim.jointintparam_ctrl_enabled)
            if ok then
                entry.ctrl_enabled = tonumber(value) or 0
                ctrlVals[#ctrlVals + 1] = entry.ctrl_enabled
            end
        end
        if sim.getJointMode ~= nil then
            local ok, value = pcall(sim.getJointMode, h)
            if ok then
                entry.joint_mode = tonumber(value) or 0
                modeVals[#modeVals + 1] = entry.joint_mode
                if sim.jointmode_dynamic ~= nil then
                    entry.joint_mode_is_dynamic = (entry.joint_mode == sim.jointmode_dynamic)
                    dynVals[#dynVals + 1] = entry.joint_mode_is_dynamic
                end
            end
        end
        jointsInfo[#jointsInfo + 1] = entry
    end
    local function allMatch(values, expected)
        if #values == 0 then
            return nil
        end
        for _, v in ipairs(values) do
            if v ~= expected then
                return false
            end
        end
        return true
    end
    return {
        joints = jointsInfo,
        motor_enabled_verified = allMatch(motorVals, 1),
        ctrl_disabled_verified = allMatch(ctrlVals, 0),
        dynamic_mode_verified = allMatch(dynVals, true),
        joint_mode_readback_available = #modeVals > 0,
        motor_readback_available = #motorVals > 0,
        ctrl_readback_available = #ctrlVals > 0,
    }
end

local function resolveHandles()
    for i, path in ipairs(JOINT_PATHS) do
        joints[i] = sim.getObject(path)
    end
    rawTaskHandle = -1
    for _, path in ipairs({'/UR5/UR5_connection', ':/UR5/UR5_connection', '/UR5_connection', ':/UR5_connection'}) do
        local ok, handle = pcall(sim.getObject, path)
        if ok then
            rawTaskHandle = handle
            break
        end
    end
    if rawTaskHandle < 0 then
        rawTaskHandle = joints[6]
    end
    taskHandle = rawTaskHandle
end

local function configureTaskFrame()
    taskFrameSummary = {mode = TASK_FRAME_MODE, mujoco_attachment_dummy = false}
    if TASK_FRAME_MODE ~= 'mujoco_attachment_dummy' then
        return
    end

    local dummySize = math.max(EE_TRIAD_DUMMY_SIZE, 0.025)
    local dummy = sim.createDummy(dummySize)
    sim.setObjectAlias(dummy, 'real_cartpole_mujoco_attachment_site')
    sim.setObjectParent(dummy, rawTaskHandle, true)
    sim.setObjectPosition(dummy, rawTaskHandle, {0.0, 0.0, TASK_FRAME_LOCAL_Z_OFFSET})
    sim.setObjectOrientation(dummy, rawTaskHandle, {0.0, 0.0, math.pi * 0.5})
    taskHandle = dummy
    taskFrameSummary = {
        mode = 'mujoco_attachment_dummy',
        mujoco_attachment_dummy = true,
        handle = taskHandle,
        parent_handle = rawTaskHandle,
        local_offset_m = {0.0, 0.0, TASK_FRAME_LOCAL_Z_OFFSET},
        local_orientation_rad = {0.0, 0.0, math.pi * 0.5},
    }
end

local function createTriadMarkers(parentHandle, aliasPrefix, axisLength, lineWidth, dummySize, rootOffset)
    local triadHandles = {}
    local triadDummyHandles = {}
    local triadShapeHandles = {}
    local triadRootHandle = -1
    if parentHandle == nil or parentHandle < 0 then
        return triadRootHandle, triadHandles, triadDummyHandles, triadShapeHandles
    end

    local triadParent = parentHandle
    local okRoot, rootHandle = pcall(sim.createDummy, 0.001)
    if okRoot and rootHandle and rootHandle >= 0 then
        triadRootHandle = rootHandle
        sim.setObjectAlias(rootHandle, aliasPrefix .. '_TriadRoot')
        sim.setObjectParent(rootHandle, parentHandle, true)
        sim.setObjectPosition(rootHandle, parentHandle, {0.0, 0.0, rootOffset})
        triadParent = rootHandle
    end

    local function createPrimitiveBar(name, size, localPos, localRot, color)
        local shape = -1
        local primitiveKind = sim.primitiveshape_cuboid or sim.primitiveshape_cube or 0
        local attempts = {
            function() return sim.createPrimitiveShape(primitiveKind, size, 0) end,
            function() return sim.createPrimitiveShape(primitiveKind, 0, size) end,
            function() return sim.createPureShape(0, primitiveKind, size, 0) end,
        }
        for _, attempt in ipairs(attempts) do
            local ok, handle = pcall(attempt)
            if ok and handle and handle >= 0 then
                shape = handle
                break
            end
        end
        if shape < 0 then
            return -1
        end
        sim.setObjectAlias(shape, name)
        sim.setObjectParent(shape, triadParent, true)
        sim.setObjectPosition(shape, triadParent, localPos)
        sim.setObjectOrientation(shape, triadParent, localRot)
        if sim.setObjectColor ~= nil and sim.colorcomponent_ambient_diffuse ~= nil then
            sim.setObjectColor(shape, 0, sim.colorcomponent_ambient_diffuse, color)
        end
        return shape
    end

    local barThickness = math.max(0.012, math.min(axisLength * 0.07, 0.02))
    triadShapeHandles = {
        createPrimitiveBar(aliasPrefix .. '_Triad_XBar', {axisLength, barThickness, barThickness}, {axisLength * 0.5, 0.0, 0.0}, {0.0, 0.0, 0.0}, {1.0, 0.05, 0.05}),
        createPrimitiveBar(aliasPrefix .. '_Triad_YBar', {axisLength, barThickness, barThickness}, {0.0, axisLength * 0.5, 0.0}, {0.0, 0.0, math.pi * 0.5}, {0.05, 1.0, 0.05}),
        createPrimitiveBar(aliasPrefix .. '_Triad_ZBar', {axisLength, barThickness, barThickness}, {0.0, 0.0, axisLength * 0.5}, {0.0, -math.pi * 0.5, 0.0}, {0.2, 0.35, 1.0}),
    }

    if #triadShapeHandles > 0 and triadShapeHandles[1] >= 0 then
        return triadRootHandle, triadHandles, triadDummyHandles, triadShapeHandles
    end

    local function axis(color, endpoint)
        local handle = sim.addDrawingObject(sim.drawing_lines, lineWidth, 0, triadParent, 1, color)
        sim.addDrawingObjectItem(handle, {0.0, 0.0, 0.0, endpoint[1], endpoint[2], endpoint[3]})
        return handle
    end

    triadHandles = {
        axis({1.0, 0.15, 0.15}, {axisLength, 0.0, 0.0}),
        axis({0.15, 1.0, 0.15}, {0.0, axisLength, 0.0}),
        axis({0.2, 0.35, 1.0}, {0.0, 0.0, axisLength}),
    }

    local function makeDummy(name, color, localPos)
        local handle = sim.createDummy(dummySize)
        sim.setObjectAlias(handle, name)
        sim.setObjectParent(handle, triadParent, true)
        sim.setObjectPosition(handle, triadParent, localPos)
        if sim.setObjectColor ~= nil and sim.colorcomponent_ambient_diffuse ~= nil then
            sim.setObjectColor(handle, 0, sim.colorcomponent_ambient_diffuse, color)
        end
        return handle
    end

    triadDummyHandles = {
        makeDummy(aliasPrefix .. '_Triad_Origin', {0.92, 0.92, 0.92}, {0.0, 0.0, 0.0}),
        makeDummy(aliasPrefix .. '_Triad_X', {1.0, 0.1, 0.1}, {axisLength, 0.0, 0.0}),
        makeDummy(aliasPrefix .. '_Triad_Y', {0.1, 1.0, 0.1}, {0.0, axisLength, 0.0}),
        makeDummy(aliasPrefix .. '_Triad_Z', {0.2, 0.35, 1.0}, {0.0, 0.0, axisLength}),
    }

    return triadRootHandle, triadHandles, triadDummyHandles, triadShapeHandles
end

local function createEeTriad()
    if not SHOW_EE_TRIAD then
        return
    end
    eeTriadRootHandle, eeTriadHandles, eeTriadDummyHandles, eeTriadShapeHandles =
        createTriadMarkers(taskHandle, 'EE', EE_TRIAD_AXIS_LENGTH, EE_TRIAD_LINE_WIDTH, EE_TRIAD_DUMMY_SIZE, EE_TRIAD_ROOT_OFFSET)
end

local function createBaseTriad()
    if not SHOW_BASE_TRIAD then
        return
    end
    baseTriadRootHandle, baseTriadHandles, baseTriadDummyHandles, baseTriadShapeHandles =
        createTriadMarkers(robotModelHandle, 'Base', BASE_TRIAD_AXIS_LENGTH, BASE_TRIAD_LINE_WIDTH, BASE_TRIAD_DUMMY_SIZE, BASE_TRIAD_ROOT_OFFSET)
end

local function createCamera()
    local sensor = sim.createVisionSensor(1 | 2 | 4 | 128, {640, 360, 0, 0}, {0.02, 7.0, math.rad(62.0), 0.1, 0.0, 0.0, 0.78, 0.82, 0.86, 0.0, 0.0})
    sim.setObjectAlias(sensor, 'TorqueTransportCamera')
    return sim.getObject('/TorqueTransportCamera')
end

local function cameraMatrix(target)
    local yaw = math.rad(-50.0)
    local radius = 2.05
    local cam = {target[1] + radius * math.cos(yaw), target[2] + radius * math.sin(yaw), target[3] + 0.30}
    local f = normalize({target[1] - cam[1], target[2] - cam[2], target[3] - cam[3]})
    local right = normalize(cross(f, {0.0, 0.0, 1.0}))
    local up = cross(right, f)
    return {
        right[1], up[1], f[1], cam[1],
        right[2], up[2], f[2], cam[2],
        right[3], up[3], f[3], cam[3],
    }
end

local function chooseTargetQuaternion(currentQuat)
    if TASK_ORIENTATION_TARGET == 'mujoco' then
        return TARGET_SITE_ROTATION_WORLD_QUAT
    end
    return copyVec(currentQuat)
end

local function smoothstep(u)
    u = clamp(u, 0.0, 1.0)
    return u * u * (3.0 - 2.0 * u)
end

local function rotCols(m)
    return {
        {m[1], m[5], m[9]},
        {m[2], m[6], m[10]},
        {m[3], m[7], m[11]},
    }
end

local function rotError(cur, target)
    local c, t = rotCols(cur), rotCols(target)
    local a, b, d = cross(c[1], t[1]), cross(c[2], t[2]), cross(c[3], t[3])
    return {0.5 * (a[1] + b[1] + d[1]), 0.5 * (a[2] + b[2] + d[2]), 0.5 * (a[3] + b[3] + d[3])}
end

local function rotAngle(cur, target)
    local c, t = rotCols(cur), rotCols(target)
    return math.acos(clamp((dot(c[1], t[1]) + dot(c[2], t[2]) + dot(c[3], t[3]) - 1.0) * 0.5, -1.0, 1.0))
end

local function getPoseMatrix()
    local m = sim.getObjectMatrix(taskHandle, sim.handle_world)
    return {m[4], m[8], m[12]}, m
end

local function getQ()
    local q = {}
    for i, h in ipairs(joints) do
        q[i] = sim.getJointPosition(h)
    end
    return q
end

local function solve6(a, b)
    local n, m = 6, {}
    for i = 1, n do
        m[i] = {}
        for j = 1, n do m[i][j] = a[i][j] end
        m[i][n + 1] = b[i]
    end
    for c = 1, n do
        local piv, best = c, math.abs(m[c][c])
        for r = c + 1, n do
            if math.abs(m[r][c]) > best then
                piv, best = r, math.abs(m[r][c])
            end
        end
        if piv ~= c then
            local tmp = m[c]
            m[c] = m[piv]
            m[piv] = tmp
        end
        local d = m[c][c]
        if math.abs(d) < 1e-10 then
            d = 1e-10
            m[c][c] = d
        end
        for j = c, n + 1 do
            m[c][j] = m[c][j] / d
        end
        for r = 1, n do
            if r ~= c then
                local f = m[r][c]
                for j = c, n + 1 do
                    m[r][j] = m[r][j] - f * m[c][j]
                end
            end
        end
    end
    local x = {}
    for i = 1, n do x[i] = m[i][n + 1] end
    return x
end

local function poseError(pt, rt)
    local p, r = getPoseMatrix()
    local er = rotError(r, rt)
    return {pt[1] - p[1], pt[2] - p[2], pt[3] - p[3], er[1], er[2], er[3]}, p, r
end

local function solveIk(seed, pt, rt, weightsOverride, posTol, rotTol)
    local q = {}
    for i = 1, 6 do q[i] = seed[i] end
    local eps, lambda = 1e-4, 0.035
    local weights = weightsOverride or {1.0, 1.0, 1.0, 2.5, 2.5, 2.5}
    local pTol = posTol or 7e-4
    local rTol = rotTol or 2e-3
    for _ = 1, 120 do
        setQ(q)
        local e, p, r = poseError(pt, rt)
        local perr, rerr = math.sqrt(e[1] * e[1] + e[2] * e[2] + e[3] * e[3]), rotAngle(r, rt)
        if perr < pTol and rerr < rTol then
            return q, true, perr, rerr
        end
        local j = {{}, {}, {}, {}, {}, {}}
        for c = 1, 6 do
            local qp = {}
            for k = 1, 6 do qp[k] = q[k] end
            qp[c] = qp[c] + eps
            setQ(qp)
            local pp, rp = getPoseMatrix()
            local re = rotError(r, rp)
            j[1][c], j[2][c], j[3][c] = (pp[1] - p[1]) / eps, (pp[2] - p[2]) / eps, (pp[3] - p[3]) / eps
            j[4][c], j[5][c], j[6][c] = re[1] / eps, re[2] / eps, re[3] / eps
        end
        local a, b = {}, {}
        for c1 = 1, 6 do
            a[c1], b[c1] = {}, 0.0
            for c2 = 1, 6 do
                a[c1][c2] = 0.0
            end
        end
        for rr = 1, 6 do
            local w = weights[rr]
            for c1 = 1, 6 do
                b[c1] = b[c1] + j[rr][c1] * w * e[rr]
                for c2 = 1, 6 do
                    a[c1][c2] = a[c1][c2] + j[rr][c1] * w * j[rr][c2]
                end
            end
        end
        for c = 1, 6 do
            a[c][c] = a[c][c] + lambda * lambda
        end
        local dq = solve6(a, b)
        local s = norm(dq)
        local scale = 1.0
        if s > 0.10 then
            scale = 0.10 / s
        end
        for c = 1, 6 do
            q[c] = q[c] + scale * dq[c]
        end
    end
    setQ(q)
    local e, _, r = poseError(pt, rt)
    local perr = math.sqrt(e[1] * e[1] + e[2] * e[2] + e[3] * e[3])
    return q, false, perr, rotAngle(r, rt)
end

local function buildMotionPath()
    setQ(Q_START)
    local startPos, startRot = getPoseMatrix()
    local targetRot = chooseTargetQuaternion(control.quat0 or {1.0, 0.0, 0.0, 0.0})
    local targetMatrix = nil
    if TASK_ORIENTATION_TARGET == 'mujoco' then
        targetMatrix = TARGET_SITE_ROTATION_WORLD
    else
        targetMatrix = startRot
    end

    local waypoints = math.max(16, tonumber(os.getenv('IK_WAYPOINTS') or '40'))
    control.path_waypoints = waypoints
    control.q_path = {}
    control.qdot_path = {}
    control.path_plan_ok = true
    control.path_first_failed_waypoint = -1
    control.path_max_pos_err = 0.0
    control.path_max_ori_err = 0.0

    local seed = copyVec(Q_START)
    local targetGoal = {startPos[1], startPos[2], startPos[3]}
    targetGoal[TRANSPORT_AXIS_INDEX] = startPos[TRANSPORT_AXIS_INDEX] + TARGET_DX
    targetGoal[3] = startPos[3] + TRANSPORT_PLANE_Z_OFFSET
    for i = 1, waypoints do
        local u = (i - 1) / math.max(waypoints - 1, 1)
        local s = smoothstep(u)
        local tgt = {startPos[1], startPos[2], startPos[3]}
        tgt[TRANSPORT_AXIS_INDEX] = startPos[TRANSPORT_AXIS_INDEX] + TARGET_DX * s
        tgt[3] = startPos[3] + TRANSPORT_PLANE_Z_OFFSET
        local q, ok, perr, rerr = solveIk(seed, tgt, targetMatrix)
        control.q_path[i] = q
        control.path_max_pos_err = math.max(control.path_max_pos_err, perr)
        control.path_max_ori_err = math.max(control.path_max_ori_err, rerr)
        if not ok and control.path_first_failed_waypoint < 0 then
            control.path_first_failed_waypoint = i
            control.path_plan_ok = false
        end
        seed = q
    end
    if control.q_path ~= nil and #control.q_path >= 1 then
        control.path_start_q = copyVec(control.q_path[1])
        control.path_end_q = copyVec(control.q_path[#control.q_path])
    end

    local _, _, _, moveDuration = solvePointToPointAccelReference(0.0, TARGET_DX, A_AXIS_MAX, V_AXIS_MAX)
    moveDuration = math.max(0.5, moveDuration)
    control.path_total_time = moveDuration
    control.targetQuat = targetRot
    control.target_axis_start = startPos[TRANSPORT_AXIS_INDEX]
    control.target_axis_goal = targetGoal[TRANSPORT_AXIS_INDEX]
    control.target_axis_total = moveDuration
    sim.addLog(sim.verbosity_scriptinfos, string.format(
        'path plan: ok=%s waypoints=%d max_pos_err=%.6g max_ori_err_deg=%.6g duration=%.6g',
        tostring(control.path_plan_ok),
        tonumber(control.path_waypoints) or 0,
        tonumber(control.path_max_pos_err) or 0.0,
        math.deg(tonumber(control.path_max_ori_err) or 0.0),
        tonumber(control.path_total_time) or 0.0
    ))
    setQ(Q_START)
end

local function interpolateQPath(u, uDot)
    local path = control.q_path or {}
    local n = #path
    if n == 0 then
        return copyVec(Q_START), {0.0, 0.0, 0.0, 0.0, 0.0, 0.0}
    end
    if n == 1 then
        return copyVec(path[1]), {0.0, 0.0, 0.0, 0.0, 0.0, 0.0}
    end
    local alpha = clamp(u, 0.0, 1.0) * (n - 1)
    local idx = math.floor(alpha) + 1
    local frac = alpha - math.floor(alpha)
    if idx >= n then
        return copyVec(path[n]), {0.0, 0.0, 0.0, 0.0, 0.0, 0.0}
    end
    local q0, q1 = path[idx], path[idx + 1]
    local qRef = {}
    local qdotRef = {}
    local du = 1.0 / math.max(n - 1, 1)
    for i = 1, 6 do
        qRef[i] = q0[i] + frac * (q1[i] - q0[i])
        qdotRef[i] = ((q1[i] - q0[i]) / du) * (uDot or 0.0)
    end
    return qRef, qdotRef
end

local function writeSummary(finalP, finalQuat, finalQ, initialP, initialQuat, peakQd, peakVTask, peakVAxis, peakTau, finalTauSatFraction)
    local transportAxisStart = control.target_axis_start
    local transportAxisFinal = finalP[TRANSPORT_AXIS_INDEX]
    local expectedAxisFinal = transportAxisStart + TARGET_DX
    local netDisplacement = transportAxisFinal - transportAxisStart
    local finalAxisError = transportAxisFinal - expectedAxisFinal
    local directionOk = math.abs(TARGET_DX) < 1e-9 or (
        signWithZero(netDisplacement) == signWithZero(TARGET_DX)
    )
    local trackingTol = math.max(0.01, 0.35 * math.abs(TARGET_DX))
    local transportAxisTrackingOk = math.abs(finalAxisError) <= trackingTol and math.abs(netDisplacement) >= math.min(math.abs(TARGET_DX) * 0.5, 0.005) and directionOk
    local fixedDrift1 = math.abs(finalP[FIXED_AXIS_1] - initialP[FIXED_AXIS_1])
    local transportPlaneTarget = initialP[3] + TRANSPORT_PLANE_Z_OFFSET
    local transportPlaneError = finalP[3] - transportPlaneTarget
    local fixedDrift2 = math.abs(transportPlaneError)
    local fixedAxesOk = fixedDrift1 <= 5.0e-3 and fixedDrift2 <= 5.0e-3
    local orientationErr = math.abs(control.max_orientation_error)
    local orientationOk = orientationErr <= math.rad(ORIENTATION_TOL_DEG)
    local jointConfigurationOk = control.max_joint_excursion <= 1.5
    local torqueSaturationOk = finalTauSatFraction <= 0.25
    local frameReferenceOk = taskFrameSummary.mode == 'ee_object' or taskFrameSummary.mode == 'mujoco_attachment_dummy'
    local baseOnGround = math.abs(MODEL_BASE_Z_OFFSET) <= 1e-9
    local success = baseOnGround and frameReferenceOk and transportAxisTrackingOk and fixedAxesOk and orientationOk and jointConfigurationOk and torqueSaturationOk and control.safety_stop_reason == nil
    local reasons = {}
    if not baseOnGround then reasons[#reasons + 1] = '"base_not_on_ground"' end
    if not frameReferenceOk then reasons[#reasons + 1] = '"frame_reference_bad"' end
    if not transportAxisTrackingOk then reasons[#reasons + 1] = '"transport_axis_tracking_failed"' end
    if not fixedAxesOk then reasons[#reasons + 1] = '"fixed_axes_drift_too_large"' end
    if not orientationOk then reasons[#reasons + 1] = '"orientation_error_too_large"' end
    if not jointConfigurationOk then reasons[#reasons + 1] = '"joint_excursion_too_large"' end
    if not torqueSaturationOk then reasons[#reasons + 1] = '"torque_saturation_too_high"' end
    if control.safety_stop_reason ~= nil then reasons[#reasons + 1] = '"safety_stop"' end

    local jointModeSummary = control.joint_mode_summary or {joints = {}}
    local lines = {
        '{',
        '  "controller_name": "coppeliasim_lua_mujoco_like_y_torque_controller",',
        '  "controller_family": ' .. jstr('lua_internal_y_axis_accel_direction_tracking_render') .. ',',
        '  "external_python_zmq_validated": false,',
        '  "uses_direct_torque_control": false,',
        '  "uses_lua_internal_torque_dynamics": true,',
        '  "uses_position_servo_setpoints": false,',
        '  "stepping_owner": ' .. jstr('coppeliasim_lua_or_internal') .. ',',
        '  "simulation_started_by": ' .. jstr('coppeliasim_or_lua') .. ',',
        '  "lua_motion_enabled": true,',
        '  "required_user_inputs": ["ACCEL_DIRECTION"],',
        '  "internal_defaults": {"ACCEL_MAGNITUDE_MPS2": 0.25, "TRAVEL_DISTANCE_M": 0.35, "ACCEL_AXIS": "Y", "TARGET_AXIS": "Y"},',
        '  "compatibility_fallback_inputs": ["TARGET_DX_M", "A_AXIS_MAX_MPS2", "TRANSPORT_AXIS"],',
        '  "camera_fixed": true,',
        '  "ee_triad_visible": ' .. jbool(SHOW_EE_TRIAD) .. ',',
        '  "base_triad_visible": ' .. jbool(SHOW_BASE_TRIAD) .. ',',
        '  "task_frame_mode": ' .. jstr(taskFrameSummary.mode or TASK_FRAME_MODE) .. ',',
        '  "task_frame_mujoco_attachment_dummy": ' .. jbool(taskFrameSummary.mujoco_attachment_dummy == true) .. ',',
        '  "task_frame_local_z_offset_m": ' .. jnum(TASK_FRAME_LOCAL_Z_OFFSET) .. ',',
        '  "transport_plane_z_offset_m": ' .. jnum(TRANSPORT_PLANE_Z_OFFSET) .. ',',
        '  "transport_axis_index": ' .. tostring(TRANSPORT_AXIS_INDEX) .. ',',
        '  "transport_axis_label": ' .. jstr(TRANSPORT_AXIS_LABEL) .. ',',
        '  "accel_axis": "Y",',
        '  "target_axis": "Y",',
        '  "accel_direction": ' .. jnum(ACCEL_DIRECTION) .. ',',
        '  "accel_direction_source": ' .. jstr(ACCEL_DIRECTION_SOURCE) .. ',',
        '  "accel_magnitude_mps2": ' .. jnum(A_AXIS_MAX) .. ',',
        '  "accel_magnitude_source": ' .. jstr(ACCEL_MAGNITUDE_SOURCE) .. ',',
        '  "travel_distance_m": ' .. jnum(math.abs(TRAVEL_DISTANCE_M)) .. ',',
        '  "travel_distance_source": ' .. jstr(TRAVEL_DISTANCE_SOURCE) .. ',',
        '  "fixed_axis_1_index": ' .. tostring(FIXED_AXIS_1) .. ',',
        '  "fixed_axis_2_index": ' .. tostring(FIXED_AXIS_2) .. ',',
        '  "start_at_transport_plane": true,',
        '  "base_on_ground": ' .. jbool(baseOnGround) .. ',',
        '  "position_tolerance_m": ' .. jnum(POSITION_TOL_M) .. ',',
        '  "orientation_tolerance_deg": ' .. jnum(ORIENTATION_TOL_DEG) .. ',',
        '  "requested_target_dx_m": ' .. jnum(TARGET_DX) .. ',',
        '  "target_dx_m": ' .. jnum(TARGET_DX) .. ',',
        '  "path_waypoints": ' .. tostring(control.path_waypoints or 0) .. ',',
        '  "path_total_time_s": ' .. jnum(control.path_total_time) .. ',',
        '  "path_plan_ok": ' .. jbool(control.path_plan_ok) .. ',',
        '  "path_first_failed_waypoint": ' .. tostring(control.path_first_failed_waypoint or -1) .. ',',
        '  "path_max_pos_err_m": ' .. jnum(control.path_max_pos_err) .. ',',
        '  "path_max_ori_err_deg": ' .. jnum(math.deg(control.path_max_ori_err or 0.0)) .. ',',
        '  "path_start_q_rad": ' .. jarr(control.path_start_q or Q_START) .. ',',
        '  "path_end_q_rad": ' .. jarr(control.path_end_q or Q_START) .. ',',
        '  "path_q_delta_rad": ' .. jarr({
            (control.path_end_q or Q_START)[1] - (control.path_start_q or Q_START)[1],
            (control.path_end_q or Q_START)[2] - (control.path_start_q or Q_START)[2],
            (control.path_end_q or Q_START)[3] - (control.path_start_q or Q_START)[3],
            (control.path_end_q or Q_START)[4] - (control.path_start_q or Q_START)[4],
            (control.path_end_q or Q_START)[5] - (control.path_start_q or Q_START)[5],
            (control.path_end_q or Q_START)[6] - (control.path_start_q or Q_START)[6],
        }) .. ',',
        '  "actuation_count": ' .. tostring(control.actuation_count or 0) .. ',',
        '  "sensing_count": ' .. tostring(control.sensing_count or 0) .. ',',
        '  "first_actuation_time_s": ' .. jnum(control.first_actuation_time) .. ',',
        '  "last_actuation_time_s": ' .. jnum(control.last_actuation_time) .. ',',
        '  "first_sensing_time_s": ' .. jnum(control.first_sensing_time) .. ',',
        '  "last_sensing_time_s": ' .. jnum(control.last_sensing_time) .. ',',
        '  "first_frame_time_s": ' .. jnum(control.first_frame_time) .. ',',
        '  "last_frame_time_s": ' .. jnum(control.last_frame_time) .. ',',
        '  "target_axis_start_m": ' .. jnum(transportAxisStart) .. ',',
        '  "target_axis_final_m": ' .. jnum(transportAxisFinal) .. ',',
        '  "target_axis_net_displacement_m": ' .. jnum(netDisplacement) .. ',',
        '  "target_axis_final_error_m": ' .. jnum(finalAxisError) .. ',',
        '  "fixed_axis_1_drift_m": ' .. jnum(fixedDrift1) .. ',',
        '  "fixed_axis_2_drift_m": ' .. jnum(fixedDrift2) .. ',',
        '  "transport_plane_target_z_m": ' .. jnum(transportPlaneTarget) .. ',',
        '  "transport_plane_final_error_m": ' .. jnum(transportPlaneError) .. ',',
        '  "max_orientation_error_deg": ' .. jnum(math.deg(orientationErr)) .. ',',
        '  "peak_abs_ee_v_axis_mps": ' .. jnum(peakVAxis) .. ',',
        '  "peak_abs_ee_speed_mps": ' .. jnum(peakVTask) .. ',',
        '  "peak_joint_speed_rad_s": ' .. jnum(peakQd) .. ',',
        '  "peak_abs_tau_nm": ' .. jnum(peakTau) .. ',',
        '  "tau_saturation_fraction": ' .. jnum(finalTauSatFraction) .. ',',
        '  "joint_configuration_ok": ' .. jbool(jointConfigurationOk) .. ',',
        '  "transport_axis_tracking_ok": ' .. jbool(transportAxisTrackingOk) .. ',',
        '  "fixed_axes_ok": ' .. jbool(fixedAxesOk) .. ',',
        '  "orientation_ok": ' .. jbool(orientationOk) .. ',',
        '  "torque_saturation_ok": ' .. jbool(torqueSaturationOk) .. ',',
        '  "frame_reference_ok": ' .. jbool(frameReferenceOk) .. ',',
        '  "success": ' .. jbool(success) .. ',',
        '  "failure_reasons": [' .. table.concat(reasons, ',') .. '],',
        '  "q_origin_source": ' .. jstr(Q_ORIGIN_SOURCE) .. ',',
        '  "q_start_source": ' .. jstr(Q_START_SOURCE) .. ',',
        '  "q_origin_rad": ' .. jarr(Q_ORIGIN) .. ',',
        '  "q_start_rad": ' .. jarr(Q_START) .. ',',
        '  "q_final_rad": ' .. jarr(finalQ) .. ',',
        '  "q_delta_rad": ' .. jarr({
            finalQ[1] - (control.q_start and control.q_start[1] or 0.0),
            finalQ[2] - (control.q_start and control.q_start[2] or 0.0),
            finalQ[3] - (control.q_start and control.q_start[3] or 0.0),
            finalQ[4] - (control.q_start and control.q_start[4] or 0.0),
            finalQ[5] - (control.q_start and control.q_start[5] or 0.0),
            finalQ[6] - (control.q_start and control.q_start[6] or 0.0),
        }) .. ',',
        '  "direct_torque_note": ' .. jstr('This Lua Y-axis lane uses simulator-side tracking/render control, not direct joint torque control.') .. ',',
        '  "ee_start_world_m": ' .. jarr(initialP) .. ',',
        '  "ee_final_world_m": ' .. jarr(finalP) .. ',',
        '  "ee_start_quat_wxyz": ' .. jarr(initialQuat) .. ',',
        '  "ee_final_quat_wxyz": ' .. jarr(finalQuat) .. ',',
        '  "joint_mode_summary": {',
        '    "motor_enabled_verified": ' .. jbool(jointModeSummary.motor_enabled_verified == true) .. ',',
        '    "ctrl_disabled_verified": ' .. jbool(jointModeSummary.ctrl_disabled_verified == true) .. ',',
        '    "dynamic_mode_verified": ' .. jbool(jointModeSummary.dynamic_mode_verified == true) .. ',',
        '    "joint_mode_readback_available": ' .. jbool(jointModeSummary.joint_mode_readback_available) .. ',',
        '    "motor_readback_available": ' .. jbool(jointModeSummary.motor_readback_available) .. ',',
        '    "ctrl_readback_available": ' .. jbool(jointModeSummary.ctrl_readback_available),
        '  },',
        '  "fps": ' .. tostring(FPS) .. ',',
        '  "frames": ' .. tostring(FRAME_COUNT) .. ',',
        '  "settle_duration_s": ' .. jnum(SETTLE_DURATION_S) .. ',',
        '  "a_axis_max_m_s2": ' .. jnum(A_AXIS_MAX) .. ',',
        '  "v_axis_max_m_s": ' .. jnum(V_AXIS_MAX) .. ',',
        '  "video_path": ' .. jstr(VIDEO_PATH),
        '}',
        '',
    }
    writeText(SUMMARY_PATH, table.concat(lines, '\n'))
end

local function captureFrameIfNeeded(simTime)
    if visionSensor < 0 or control.camera_pose == nil then
        return
    end
    if simTime + 1e-9 < control.nextFrameTime then
        return
    end
    sim.setObjectMatrix(visionSensor, control.camera_pose, sim.handle_world)
    sim.handleVisionSensor(visionSensor)
    local img, res = sim.getVisionSensorImg(visionSensor)
    img = sim.transformImage(img, res, 4)
    local fileName = string.format('%s/%s_%08d.png', OUTPUT_DIR, FRAME_PREFIX, control.framesCaptured)
    sim.saveImage(img, res, 0, fileName, -1)
    if control.first_frame_time == nil then
        control.first_frame_time = simTime
    end
    control.last_frame_time = simTime
    control.framesCaptured = control.framesCaptured + 1
    control.nextFrameTime = control.nextFrameTime + 1.0 / math.max(FPS, 1e-9)
end

local function computeAndApplyTorque(simTimeOverride)
    local simTime = simTimeOverride
    if simTime == nil then
        simTime = sim.getSimulationTime()
    end
    local q, qd = readJointState()
    local p, quat, lin, ang = readTaskState()
    local tMove = simTime - SETTLE_DURATION_S
    local axisOffset, axisVel, axisAcc, totalTime = 0.0, 0.0, 0.0, control.path_total_time or 0.0
    if tMove >= 0.0 then
        axisOffset, axisVel, axisAcc, totalTime = solvePointToPointAccelReference(tMove, TARGET_DX, A_AXIS_MAX, V_AXIS_MAX)
    end
    control.target_axis_total = totalTime
    control.target_axis_velocity_peak = math.max(control.target_axis_velocity_peak, math.abs(axisVel))
    control.target_axis_accel_peak = math.max(control.target_axis_accel_peak, math.abs(axisAcc))
    local targetPos = {control.p0[1], control.p0[2], control.p0[3]}
    targetPos[TRANSPORT_AXIS_INDEX] = control.p0[TRANSPORT_AXIS_INDEX] + axisOffset
    local targetVel = {0.0, 0.0, 0.0}
    targetVel[TRANSPORT_AXIS_INDEX] = axisVel

    local xErr = targetPos[1] - p[1]
    local yErr = targetPos[2] - p[2]
    local zErr = targetPos[3] - p[3]
    local oriErrVec = orientationErrorVecWxyz(control.targetQuat, quat)
    local oriErrNorm = norm(oriErrVec)
    local reportedTaskSpeed = norm(lin)
    local reportedAxisSpeed = math.abs(lin[TRANSPORT_AXIS_INDEX])
    if control.internal_mode then
        reportedTaskSpeed = math.max(reportedTaskSpeed, math.abs(axisVel))
        reportedAxisSpeed = math.max(reportedAxisSpeed, math.abs(axisVel))
    end
    control.max_orientation_error = math.max(control.max_orientation_error, oriErrNorm)
    control.max_fixed_axis_1_drift = math.max(control.max_fixed_axis_1_drift, math.abs(p[FIXED_AXIS_1] - control.p0[FIXED_AXIS_1]))
    control.max_fixed_axis_2_drift = math.max(control.max_fixed_axis_2_drift, math.abs(p[FIXED_AXIS_2] - control.p0[FIXED_AXIS_2]))
    control.max_joint_excursion = math.max(control.max_joint_excursion, maxAbsDiff(q, control.q0))
    control.peak_joint_speed = math.max(control.peak_joint_speed, vectorMaxAbs(qd))
    control.peak_task_speed = math.max(control.peak_task_speed, reportedTaskSpeed)
    control.peak_axis_speed = math.max(control.peak_axis_speed, reportedAxisSpeed)
    control.q_final = copyVec(q)
    control.p_final = copyVec(p)
    control.quat_final = copyVec(quat)
    control.target_axis_final = p[TRANSPORT_AXIS_INDEX]
    control.target_axis_net = p[TRANSPORT_AXIS_INDEX] - control.target_axis_start
    control.target_axis_final_error = p[TRANSPORT_AXIS_INDEX] - (control.target_axis_start + TARGET_DX)

    local u = 1.0
    local uDot = 0.0
    if math.abs(TARGET_DX) > 1e-9 then
        u = clamp(axisOffset / TARGET_DX, 0.0, 1.0)
        uDot = axisVel / TARGET_DX
    end
    local qRef, qdotRef = interpolateQPath(u, uDot)
    local tauRaw = {}
    for i = 1, 6 do
        tauRaw[i] = TRACK_KP[i] * (qRef[i] - q[i]) + TRACK_KD[i] * (qdotRef[i] - qd[i])
    end
    local tau, saturated = clipTorques(tauRaw, MODEL_TORQUE_LIMITS)
    control.last_tau = copyVec(tau)
    applyTorque(tau)
    control.peak_tau = math.max(control.peak_tau, vectorMaxAbs(tau))
    for i = 1, 6 do
        if saturated[i] then
            control.tau_sat_count = control.tau_sat_count + 1
        end
        control.tau_sat_samples = control.tau_sat_samples + 1
    end

    local finiteOk = true
    for _, v in ipairs({q[1], q[2], q[3], q[4], q[5], q[6], qd[1], qd[2], qd[3], qd[4], qd[5], qd[6], p[1], p[2], p[3], quat[1], quat[2], quat[3], quat[4], lin[1], lin[2], lin[3], ang[1], ang[2], ang[3]}) do
        if v ~= v or v == math.huge or v == -math.huge then
            finiteOk = false
            break
        end
    end
    if not finiteOk and control.safety_stop_reason == nil then
        control.safety_stop_reason = 'non_finite_state'
    end
    if control.max_joint_excursion > 4.0 and control.safety_stop_reason == nil then
        control.safety_stop_reason = 'joint_excursion_too_large'
    end
    if math.abs(xErr) > 1.0 or math.abs(yErr) > 1.0 or math.abs(zErr) > 1.0 then
        if control.safety_stop_reason == nil then
            control.safety_stop_reason = 'task_position_diverged'
        end
    end
end

local function finishAndQuit()
    if control.framesCaptured == 0 then
        return
    end
    if control.q_final == nil then
        control.q_final = copyVec(control.q0 or Q_START)
    end
    if control.p_final == nil then
        control.p_final = copyVec(control.p0 or {0.0, 0.0, 0.0})
    end
    if control.quat0 == nil then
        control.quat0 = {1.0, 0.0, 0.0, 0.0}
    end
    local tauSatFraction = 0.0
    if control.tau_sat_samples > 0 then
        tauSatFraction = control.tau_sat_count / control.tau_sat_samples
    end
    writeSummary(control.p_final, control.quat_final or control.quat0 or {1.0, 0.0, 0.0, 0.0}, control.q_final, control.p_start or control.p0 or {0.0, 0.0, 0.0}, control.quat0 or {1.0, 0.0, 0.0, 0.0}, control.peak_joint_speed, control.peak_task_speed, control.peak_axis_speed, control.peak_tau, tauSatFraction)
    writeText(DONE_MARKER, 'done\n')
    applyTorque({0.0, 0.0, 0.0, 0.0, 0.0, 0.0})
    control.manual_loop_running = false
    if control.simulation_start_requested and sim.getSimulationState() ~= sim.simulation_stopped then
        pcall(sim.stopSimulation)
    end
    sim.quitSimulator()
end

local function integrateInternalJointState(dt)
    if control.internal_q == nil or control.internal_qd == nil then
        return
    end
    local tau = control.last_tau or {0.0, 0.0, 0.0, 0.0, 0.0, 0.0}
    for i = 1, 6 do
        local accel = ((tau[i] or 0.0) - INTERNAL_JOINT_DAMPING[i] * control.internal_qd[i]) / INTERNAL_JOINT_INERTIA[i]
        control.internal_qd[i] = control.internal_qd[i] + dt * accel
        control.internal_q[i] = control.internal_q[i] + dt * control.internal_qd[i]
    end
end

local function runInternalTorqueRender()
    if control.manual_loop_running then
        return
    end
    control.manual_loop_running = true
    control.internal_mode = true
    sim.addLog(sim.verbosity_scriptinfos, 'UR5 MuJoCo-like Y torque add-on starting internal Lua render loop')

    local dt = 1.0 / math.max(FPS, 1.0)
    local simTime = 0.0
    control.internal_q = copyVec(control.q0 or Q_START)
    control.internal_qd = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0}
    control.last_tau = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0}
    setQ(control.internal_q)
    syncKinematics()

    control.p0, control.quat0 = readTaskState()
    control.q0 = copyVec(control.internal_q)
    control.q_start = copyVec(control.internal_q)
    control.q_final = copyVec(control.internal_q)
    control.p_start = copyVec(control.p0)
    control.p_final = copyVec(control.p0)
    control.quat_final = copyVec(control.quat0)
    control.targetQuat = chooseTargetQuaternion(control.quat0)
    control.target_axis_start = control.p0[TRANSPORT_AXIS_INDEX]
    control.target_axis_goal = control.target_axis_start + TARGET_DX
    control.nextFrameTime = 0.0

    if visionSensor < 0 then
        visionSensor = createCamera()
    end
    if visionSensor < 1 then
        sim.addLog(sim.verbosity_errors, 'Torque video sensor creation failed')
        writeText(DONE_MARKER, 'failed\n')
        control.manual_loop_running = false
        sim.quitSimulator()
        return
    end
    if SHOW_EE_TRIAD and (eeTriadRootHandle < 0) then
        createEeTriad()
    end
    if SHOW_BASE_TRIAD and (baseTriadRootHandle < 0) then
        createBaseTriad()
    end

    writeText(SENSING_MARKER, 'capture\n')
    for frameIdx = 0, FRAME_COUNT - 1 do
        setQ(control.internal_q)
        syncKinematics()

        control.actuation_count = (control.actuation_count or 0) + 1
        if control.first_actuation_time == nil then
            control.first_actuation_time = simTime
        end
        control.last_actuation_time = simTime

        control.sensing_count = (control.sensing_count or 0) + 1
        if control.first_sensing_time == nil then
            control.first_sensing_time = simTime
        end
        control.last_sensing_time = simTime
        writeText(SENSING_MARKER, string.format('count=%d time=%.6f frames=%d\n', control.sensing_count or 0, simTime, control.framesCaptured or 0))

        computeAndApplyTorque(simTime)
        captureFrameIfNeeded(simTime)

        integrateInternalJointState(dt)
        setQ(control.internal_q)
        syncKinematics()

        local pEnd, quatEnd, _, _ = readTaskState()
        control.q_final = copyVec(control.internal_q)
        control.p_final = copyVec(pEnd)
        control.quat_final = copyVec(quatEnd)

        simTime = simTime + dt
    end

    if control.internal_q ~= nil then
        setQ(control.internal_q)
        syncKinematics()
        local pEnd, quatEnd = readTaskState()
        control.p_final = copyVec(pEnd)
        control.quat_final = copyVec(quatEnd)
        control.q_final = copyVec(control.internal_q)
        control.target_axis_final = pEnd[TRANSPORT_AXIS_INDEX]
        control.target_axis_net = pEnd[TRANSPORT_AXIS_INDEX] - control.target_axis_start
        control.target_axis_final_error = pEnd[TRANSPORT_AXIS_INDEX] - (control.target_axis_start + TARGET_DX)
    end

    finishAndQuit()
end

local function runTorqueLoop()
    local previousStepping = nil
    if sim.setStepping ~= nil then
        local ok, value = pcall(sim.setStepping, true)
        if ok then
            previousStepping = value
        end
    end

    control.manual_loop_running = true
    sim.addLog(sim.verbosity_scriptinfos, 'UR5 MuJoCo-like Y torque add-on starting stepped torque loop')
    sim.startSimulation()
    while control.framesCaptured < FRAME_COUNT and control.safety_stop_reason == nil do
        local simTime = sim.getSimulationTime()
        control.actuation_count = (control.actuation_count or 0) + 1
        if control.first_actuation_time == nil then
            control.first_actuation_time = simTime
        end
        control.last_actuation_time = simTime
        computeAndApplyTorque()
        control.sensing_count = (control.sensing_count or 0) + 1
        if control.first_sensing_time == nil then
            control.first_sensing_time = simTime
        end
        control.last_sensing_time = simTime
        writeText(SENSING_MARKER, string.format('count=%d time=%.6f frames=%d\n', control.sensing_count or 0, simTime, control.framesCaptured or 0))
        captureFrameIfNeeded(simTime)
        if control.framesCaptured >= FRAME_COUNT then
            break
        end
        if sim.step ~= nil then
            pcall(sim.step)
        else
            break
        end
    end
    control.manual_loop_running = false
    if previousStepping ~= nil and sim.setStepping ~= nil then
        pcall(sim.setStepping, previousStepping)
    end
    finishAndQuit()
end

function sysCall_info()
    return {autoStart = true}
end

function sysCall_init()
    writeText(LOAD_MARKER, 'loaded\n')
    writeText(START_MARKER, 'init\n')
    sim.addLog(sim.verbosity_scriptinfos, 'UR5 MuJoCo-like Y torque add-on starting')
    sim.loadScene(SCENE_PATH)
    sim.setColorProperty(sim.handle_scene, 'ambientLight', {0.65, 0.65, 0.65}, {noError = true})
    robotModelHandle = sim.loadModel(MODEL_PATH)
    if robotModelHandle and robotModelHandle >= 0 and MODEL_BASE_Z_OFFSET ~= 0.0 then
        sim.setObjectPosition(robotModelHandle, sim.handle_world, {0.0, 0.0, MODEL_BASE_Z_OFFSET})
    end
    resolveHandles()
    configureTaskFrame()
    setQ(Q_START)
    syncKinematics()
    control.p0, control.quat0 = readTaskState()
    control.q0 = select(1, readJointState())
    control.q_start = copyVec(control.q0)
    control.q_final = copyVec(control.q0)
    control.p_start = copyVec(control.p0)
    control.p_final = copyVec(control.p0)
    control.quat_final = copyVec(control.quat0)
    control.targetQuat = chooseTargetQuaternion(control.quat0)
    control.target_axis_start = control.p0[TRANSPORT_AXIS_INDEX]
    control.target_axis_goal = control.target_axis_start + TARGET_DX
    buildMotionPath()
    local cameraCenter = {control.p0[1], control.p0[2], control.p0[3] + 0.30}
    cameraCenter[TRANSPORT_AXIS_INDEX] = cameraCenter[TRANSPORT_AXIS_INDEX] + 0.5 * TARGET_DX
    control.camera_pose = cameraMatrix(cameraCenter)
    visionSensor = createCamera()
    if SHOW_EE_TRIAD then
        createEeTriad()
    end
    if SHOW_BASE_TRIAD then
        createBaseTriad()
    end
    control.joint_mode_summary = readJointConfigurationSummary()
    control.initialized = true
    control.startTime = sim.getSimulationTime()
    control.nextFrameTime = 0.0
    writeText(CONFIGURED_MARKER, 'configured\n')
    if USE_EXTERNAL_STEP_PUMP then
        configureForceTorqueMode()
        if STEP_RELEASE_MARKER ~= '' then
            sim.addLog(sim.verbosity_scriptinfos, 'waiting for Lua step release marker: ' .. STEP_RELEASE_MARKER)
            while not fileExists(STEP_RELEASE_MARKER) do
                wallSleep(0.1)
            end
            sim.addLog(sim.verbosity_scriptinfos, 'Lua step release marker observed')
            if STEP_READY_MARKER ~= '' then
                writeText(STEP_READY_MARKER, 'ready\n')
                sim.addLog(sim.verbosity_scriptinfos, 'Lua step ready marker written: ' .. STEP_READY_MARKER)
            end
        end
        control.simulation_start_requested = true
        if sim.getSimulationState() == sim.simulation_stopped then
            sim.startSimulation()
            sim.addLog(sim.verbosity_scriptinfos, 'Lua torque simulation start requested before returning from init')
        end
        local grace_s = tonumber(os.getenv('REAL_CARTPOLE_LUA_STEP_GRACE_S') or '1.0') or 1.0
        if grace_s > 0.0 then
            wallSleep(grace_s)
        end
        return
    end
    runInternalTorqueRender()
end

function sysCall_actuation()
    if control.manual_loop_running then
        return
    end
    local simTime = sim.getSimulationTime()
    control.actuation_count = (control.actuation_count or 0) + 1
    if control.first_actuation_time == nil then
        control.first_actuation_time = simTime
    end
    control.last_actuation_time = simTime
    if control.safety_stop_reason ~= nil then
        applyTorque({0.0, 0.0, 0.0, 0.0, 0.0, 0.0})
        return
    end
    computeAndApplyTorque()
end

function sysCall_sensing()
    if control.manual_loop_running then
        return
    end
    local simTime = sim.getSimulationTime()
    control.sensing_count = (control.sensing_count or 0) + 1
    if control.first_sensing_time == nil then
        control.first_sensing_time = simTime
    end
    control.last_sensing_time = simTime
    writeText(SENSING_MARKER, string.format('count=%d time=%.6f frames=%d\n', control.sensing_count or 0, simTime, control.framesCaptured or 0))
    if control.safety_stop_reason ~= nil then
        if control.framesCaptured >= FRAME_COUNT then
            return
        end
    end
    captureFrameIfNeeded(simTime)
    if control.framesCaptured >= FRAME_COUNT then
        finishAndQuit()
    end
end

function sysCall_cleanup()
    local tauSatFraction = 0.0
    if control.tau_sat_samples > 0 then
        tauSatFraction = control.tau_sat_count / control.tau_sat_samples
    end
    if control.q_final ~= nil and control.p_final ~= nil then
        writeSummary(control.p_final, control.quat_final or control.quat0 or {1.0, 0.0, 0.0, 0.0}, control.q_final, control.p_start or control.p0 or {0.0, 0.0, 0.0}, control.quat0 or {1.0, 0.0, 0.0, 0.0}, control.peak_joint_speed, control.peak_task_speed, control.peak_axis_speed, control.peak_tau, tauSatFraction)
    end
    sim.addLog(sim.verbosity_scriptinfos, 'UR5 MuJoCo-like Y torque add-on cleanup')
end
