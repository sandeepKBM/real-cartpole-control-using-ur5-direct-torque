local COPPELIA_ROOT = os.getenv('COPPELIA_ROOT') or '/common/users/ss5772/real_Cartpole/third_party/coppelia_runtime/CoppeliaSim_Edu_V4_10_0_rev0_Ubuntu24_04'
local ROOT = os.getenv('REAL_CARTPOLE_ROOT') or '/common/users/ss5772/real_Cartpole'
local SCENE_PATH = COPPELIA_ROOT .. '/system/dfltscn.ttt'
local MODEL_PATH = COPPELIA_ROOT .. '/models/robots/non-mobile/UR5.ttm'
local OUTPUT_DIR = os.getenv('OUTPUT_DIR') or (ROOT .. '/outputs/control_runs/coppelia_origin_acquisition_frames')
local STATE_DIR = os.getenv('STATE_DIR') or (ROOT .. '/outputs/control_runs/coppelia_origin_acquisition_state')
local VIDEO_PATH = os.getenv('VIDEO_PATH') or (ROOT .. '/demonstration_videos/ur5e_coppeliasim/coppeliasim_ur5_origin_acquisition.mp4')
local SUMMARY_PATH = os.getenv('SUMMARY_PATH') or (STATE_DIR .. '/coppeliasim_ur5_origin_acquisition_summary.json')
local FRAME_PREFIX = os.getenv('FRAME_PREFIX') or 'frame'
local LOAD_MARKER = STATE_DIR .. '/ur5_origin_acquisition_addon_loaded.txt'
local START_MARKER = STATE_DIR .. '/ur5_origin_acquisition_addon_started.txt'
local SENSING_MARKER = STATE_DIR .. '/ur5_origin_acquisition_addon_sensing.txt'
local DONE_MARKER = STATE_DIR .. '/ur5_origin_acquisition_done.txt'

local FPS = tonumber(os.getenv('FPS') or '25')
local ORIGIN_MOVE_FRAMES = tonumber(os.getenv('ORIGIN_MOVE_FRAMES') or '80')
local GAP_FRAMES = tonumber(os.getenv('GAP_FRAMES') or tostring(math.floor(FPS + 0.5)))
local ACCEL_FRAMES = tonumber(os.getenv('ACCEL_FRAMES') or '80')
local FRAME_COUNT = tonumber(os.getenv('FRAME_COUNT') or tostring(ORIGIN_MOVE_FRAMES + GAP_FRAMES + ACCEL_FRAMES))
local DURATION_S = tonumber(os.getenv('MOVE_DURATION_S') or tostring(FRAME_COUNT / FPS))
local TARGET_DX = tonumber(os.getenv('TARGET_DX_M') or '0.06')
local V_X_MAX = tonumber(os.getenv('V_X_MAX_MPS') or '0.25')
local A_X_MAX = tonumber(os.getenv('A_X_MAX_MPS2') or '1.2')
local IK_WAYPOINTS = tonumber(os.getenv('IK_WAYPOINTS') or '40')
local EE_TARGET_Z = tonumber(os.getenv('EE_TARGET_Z_M') or '0.65')
local ACCEL_ROT_WEIGHT = tonumber(os.getenv('ACCEL_ROT_WEIGHT') or '0.05')
local SHOW_EE_TRIAD = os.getenv('SHOW_EE_TRIAD')
if SHOW_EE_TRIAD == nil or SHOW_EE_TRIAD == '' then
    SHOW_EE_TRIAD = true
else
    SHOW_EE_TRIAD = SHOW_EE_TRIAD ~= '0'
end
local TASK_FRAME_MODE = string.lower(os.getenv('TASK_FRAME_MODE') or 'ee_object')
local USE_MUJOCO_TARGET_ORIENTATION_RAW = os.getenv('USE_MUJOCO_TARGET_ORIENTATION')
local USE_MUJOCO_TARGET_ORIENTATION = false
if USE_MUJOCO_TARGET_ORIENTATION_RAW == nil or USE_MUJOCO_TARGET_ORIENTATION_RAW == '' then
    USE_MUJOCO_TARGET_ORIENTATION = (TASK_FRAME_MODE == 'mujoco_attachment_dummy')
else
    USE_MUJOCO_TARGET_ORIENTATION = USE_MUJOCO_TARGET_ORIENTATION_RAW ~= '0'
end
-- MuJoCo attachment_site axes in world (cart-pole transport frame).
local TARGET_SITE_ROTATION_WORLD = {
    -1.0, 0.0, 0.0, 0.0,
    0.0, 0.0, -1.0, 0.0,
    0.0, -1.0, 0.0, 0.0,
}
-- Attachment dummy world rotation (Coppelia UR5.ttm proxy rotated +90 deg about local Z).
local TARGET_ATTACHMENT_ROTATION_WORLD = {
    0.0, 1.0, 0.0, 0.0,
    0.0, 0.0, -1.0, 0.0,
    -1.0, 0.0, 0.0, 0.0,
}
local TASK_FRAME_ATTACHMENT_OFFSET = {0.0, 0.0, -0.2}
local TASK_FRAME_ATTACHMENT_QUAT_WXYZ = {
    -1.0,
    1.0,
    0.0,
    0.0,
}
local LOCK_SHOULDER_PAN_RAW = os.getenv('LOCK_SHOULDER_PAN')
local LOCK_SHOULDER_PAN = false
if LOCK_SHOULDER_PAN_RAW == nil or LOCK_SHOULDER_PAN_RAW == '' then
    LOCK_SHOULDER_PAN = USE_MUJOCO_TARGET_ORIENTATION
else
    LOCK_SHOULDER_PAN = LOCK_SHOULDER_PAN_RAW ~= '0'
end
local SHOULDER_PAN_JOINT_INDEX = 1
local shoulderPanLockedRad = 0.0
local START_AT_TRANSPORT_PLANE = os.getenv('START_AT_TRANSPORT_PLANE') == '1'
local MUJOCO_LIKE_X_SWEEP = os.getenv('MUJOCO_LIKE_X_SWEEP') == '1'
local MUJOCO_LIKE_SWEEP_LEGS = math.max(1, tonumber(os.getenv('MUJOCO_LIKE_SWEEP_LEGS') or '3'))
local function parseSweepAxis(raw)
    raw = string.lower(raw or 'x')
    if raw == 'x' or raw == '1' then
        return 1, 'x'
    elseif raw == 'y' or raw == '2' then
        return 2, 'y'
    elseif raw == 'z' or raw == '3' then
        return 3, 'z'
    end
    return 1, 'x'
end
local SWEEP_AXIS_INDEX, SWEEP_AXIS_LABEL = parseSweepAxis(os.getenv('MUJOCO_LIKE_SWEEP_AXIS'))
local SWEEP_FIXED_AXIS_1 = 2
local SWEEP_FIXED_AXIS_2 = 3
if SWEEP_AXIS_INDEX == 1 then
    SWEEP_FIXED_AXIS_1, SWEEP_FIXED_AXIS_2 = 2, 3
elseif SWEEP_AXIS_INDEX == 2 then
    SWEEP_FIXED_AXIS_1, SWEEP_FIXED_AXIS_2 = 1, 3
else
    SWEEP_FIXED_AXIS_1, SWEEP_FIXED_AXIS_2 = 1, 2
end
local EE_TRIAD_AXIS_LENGTH = tonumber(os.getenv('EE_TRIAD_AXIS_LENGTH_M') or '0.18')
local EE_TRIAD_LINE_WIDTH = tonumber(os.getenv('EE_TRIAD_LINE_WIDTH_PX') or '6')
local EE_TRIAD_DUMMY_SIZE = tonumber(os.getenv('EE_TRIAD_DUMMY_SIZE_M') or '0.03')
local EE_TRIAD_ROOT_OFFSET = tonumber(os.getenv('EE_TRIAD_ROOT_OFFSET_M') or '0.08')
local SHOW_BASE_TRIAD = os.getenv('SHOW_BASE_TRIAD')
if SHOW_BASE_TRIAD == nil or SHOW_BASE_TRIAD == '' then
    SHOW_BASE_TRIAD = true
else
    SHOW_BASE_TRIAD = SHOW_BASE_TRIAD ~= '0'
end
local BASE_TRIAD_AXIS_LENGTH = tonumber(os.getenv('BASE_TRIAD_AXIS_LENGTH_M') or '0.16')
local BASE_TRIAD_LINE_WIDTH = tonumber(os.getenv('BASE_TRIAD_LINE_WIDTH_PX') or '6')
local BASE_TRIAD_DUMMY_SIZE = tonumber(os.getenv('BASE_TRIAD_DUMMY_SIZE_M') or '0.03')
local BASE_TRIAD_ROOT_OFFSET = tonumber(os.getenv('BASE_TRIAD_ROOT_OFFSET_M') or '0.08')
local RANGE_SCAN_MAX_M = tonumber(os.getenv('RANGE_SCAN_MAX_M') or '0.35')
local RANGE_SCAN_STEP_M = tonumber(os.getenv('RANGE_SCAN_STEP_M') or '0.0025')
local RANGE_POSITION_TOL_M = tonumber(os.getenv('RANGE_POSITION_TOL_M') or '0.005')
local RANGE_SWEEP_ONLY = os.getenv('RANGE_SWEEP_ONLY') == '1'
local SWEEP_Z_MIN = tonumber(os.getenv('SWEEP_Z_MIN_M') or '0.35')
local SWEEP_Z_MAX = tonumber(os.getenv('SWEEP_Z_MAX_M') or '0.85')
local SWEEP_Z_STEP = tonumber(os.getenv('SWEEP_Z_STEP_M') or '0.025')
local MODEL_BASE_Z_OFFSET = tonumber(os.getenv('MODEL_BASE_Z_OFFSET_M') or '0.0')
local POSITION_TOL_M = tonumber(os.getenv('ORIGIN_POSITION_TOL_M') or '0.015')
local ORIENTATION_TOL_DEG = tonumber(os.getenv('ORIGIN_ORIENTATION_TOL_DEG') or '3.0')

local JOINT_PATHS = {
    '/UR5/joint',
    '/UR5/link/joint',
    '/UR5/link/link/joint',
    '/UR5/link/link/link/joint',
    '/UR5/link/link/link/link/joint',
    '/UR5/link/link/link/link/link/joint',
}

-- Grounded Coppelia-side reference pose. Override with ORIGIN_Q_RAD once the
-- exact MuJoCo reference frame has been mapped into this UR5 model.
local Q_ORIGIN = {
    0.0,
    -0.1133064268431449,
    -0.664621645801302,
    4.921777393344012,
    -6.283185307179586,
    5.280928640069786,
}
local Q_START = {
    0.35,
    -0.5633064268431449,
    -0.114621645801302,
    4.471777393344012,
    -5.933185307179586,
    4.980928640069786,
}
local Q_ORIGIN_SOURCE = 'coppelia_grounded_fixed_z_origin_default'
local Q_START_SOURCE = 'deterministic_offset_from_origin'

local sim = require 'sim'
local joints = {}
local eeHandle = -1
local visionSensor = -1
local targetPos = nil
local targetRot = nil
local effectiveTargetDx = TARGET_DX
local ikSolution = nil
local ikSuccess = false
local ikFinalPosErr = 0.0
local ikFinalRotErr = 0.0
local accelPath = {}
local accelPathLength = 0.0
local accelIkSuccess = true
local accelIkMaxPosErr = 0.0
local accelIkMaxRotErr = 0.0
local accelFirstFailedWaypoint = -1
local accelMonitor = {u={}, target_x={}, target_y={}, target_z={}, actual_x={}, actual_y={}, actual_z={}, pos_err={}, ori_err_deg={}, ik_ok={}}
local rangeSummary = {
    axis_index=SWEEP_AXIS_INDEX, axis_label=SWEEP_AXIS_LABEL,
    axis_min=0.0, axis_max=0.0,
    x_min=0.0, x_max=0.0,
    dx_neg=0.0, dx_pos=0.0, selected_dx=0.0,
    clamped=0, direction=1, first_failed_pos_step=-1, first_failed_neg_step=-1,
    pos_u={}, pos_x={}, pos_err={}, pos_ori_err_deg={}, pos_ok={},
    neg_u={}, neg_x={}, neg_err={}, neg_ori_err_deg={}, neg_ok={},
}
local traces = {t={}, phase={}, pos_err={}, ori_err={}, x={}, y={}, z={}, vx={}, speed={}, accel_progress={}, qdmax={}}
local eeTriadHandles = {}
local eeTriadDummyHandles = {}
local eeTriadShapeHandles = {}
local eeTriadRootHandle = -1
local baseTriadHandles = {}
local baseTriadDummyHandles = {}
local baseTriadShapeHandles = {}
local baseTriadRootHandle = -1
local robotModelHandle = -1
local taskFrameSummary = {mode = TASK_FRAME_MODE, mujoco_attachment_dummy = false}
local sweepPath = {}
local sweepPathLength = 0.0
local sweepLegTargets = {}
local sweepLegFrameCounts = {}
local sweepLegStartXs = {}

function sysCall_info()
    return {autoStart = true}
end

local function writeText(pathName, text)
    local f = io.open(pathName, 'w')
    if f then f:write(text or ''); f:close() end
end

local function parseJointVectorEnv(name, fallback)
    local raw = os.getenv(name)
    if raw == nil or raw == '' then return fallback, nil end
    local q = {}
    for token in string.gmatch(raw, '[^,%s]+') do
        local value = tonumber(token)
        if value == nil then return fallback, 'invalid_number:' .. token end
        q[#q+1] = value
    end
    if #q ~= 6 then return fallback, 'expected_6_values_got_' .. tostring(#q) end
    return q, nil
end

local function copyJointVector(v)
    local out = {}
    for i = 1, #v do
        out[i] = v[i]
    end
    return out
end

local function applyEnvOverrides()
    local qOrigin, originErr = parseJointVectorEnv('ORIGIN_Q_RAD', Q_ORIGIN)
    Q_ORIGIN = qOrigin
    if originErr == nil and os.getenv('ORIGIN_Q_RAD') ~= nil and os.getenv('ORIGIN_Q_RAD') ~= '' then
        Q_ORIGIN_SOURCE = 'env_ORIGIN_Q_RAD'
    elseif originErr ~= nil then
        Q_ORIGIN_SOURCE = Q_ORIGIN_SOURCE .. '_env_parse_error_' .. originErr
    end

    local qStart, startErr = parseJointVectorEnv('START_Q_RAD', Q_START)
    Q_START = qStart
    if startErr == nil and os.getenv('START_Q_RAD') ~= nil and os.getenv('START_Q_RAD') ~= '' then
        Q_START_SOURCE = 'env_START_Q_RAD'
    elseif startErr ~= nil then
        Q_START_SOURCE = Q_START_SOURCE .. '_env_parse_error_' .. startErr
    end

    if (os.getenv('START_Q_RAD') == nil or os.getenv('START_Q_RAD') == '') and START_AT_TRANSPORT_PLANE then
        Q_START = copyJointVector(Q_ORIGIN)
        Q_START_SOURCE = 'transport_plane_start_matches_origin'
    end

    local panLocked = tonumber(os.getenv('SHOULDER_PAN_LOCKED_RAD'))
    if panLocked ~= nil then
        shoulderPanLockedRad = panLocked
    else
        shoulderPanLockedRad = Q_ORIGIN[SHOULDER_PAN_JOINT_INDEX]
    end
end

local function clamp(x, lo, hi)
    if x < lo then return lo end
    if x > hi then return hi end
    return x
end

local function cross(a, b)
    return {a[2]*b[3]-a[3]*b[2], a[3]*b[1]-a[1]*b[3], a[1]*b[2]-a[2]*b[1]}
end

local function dot(a, b)
    return a[1]*b[1] + a[2]*b[2] + a[3]*b[3]
end

local function norm(v)
    local s = 0.0
    for i = 1, #v do s = s + v[i]*v[i] end
    return math.sqrt(s)
end

local function dist(a, b)
    return math.sqrt((a[1]-b[1])^2 + (a[2]-b[2])^2 + (a[3]-b[3])^2)
end

local function normalize(v)
    local n = norm(v)
    if n < 1e-12 then return {0, 0, 0} end
    return {v[1]/n, v[2]/n, v[3]/n}
end

local function smoothstep(u)
    u = clamp(u, 0.0, 1.0)
    return u*u*(3.0 - 2.0*u)
end

local function rotCols(m)
    return {{m[1],m[5],m[9]}, {m[2],m[6],m[10]}, {m[3],m[7],m[11]}}
end

local function rotError(cur, target)
    local c, t = rotCols(cur), rotCols(target)
    local a, b, d = cross(c[1], t[1]), cross(c[2], t[2]), cross(c[3], t[3])
    return {0.5*(a[1]+b[1]+d[1]), 0.5*(a[2]+b[2]+d[2]), 0.5*(a[3]+b[3]+d[3])}
end

local function rotAngle(cur, target)
    local c, t = rotCols(cur), rotCols(target)
    return math.acos(clamp((dot(c[1],t[1]) + dot(c[2],t[2]) + dot(c[3],t[3]) - 1.0) * 0.5, -1.0, 1.0))
end

local function getPose()
    local p = sim.getObjectPose(eeHandle, sim.handle_world)
    return {p[1], p[2], p[3]}, sim.getObjectMatrix(eeHandle, sim.handle_world)
end

local function enforceShoulderPanLock(q)
    if LOCK_SHOULDER_PAN then
        q[SHOULDER_PAN_JOINT_INDEX] = shoulderPanLockedRad
    end
    return q
end

local function setQ(q)
    enforceShoulderPanLock(q)
    for i, h in ipairs(joints) do sim.setJointPosition(h, q[i]) end
end

local function getQ()
    local q = {}
    for i, h in ipairs(joints) do q[i] = sim.getJointPosition(h) end
    return q
end

local function refreshTargetPoseFromOrigin()
    setQ(Q_ORIGIN)
    local p, r = getPose()
    targetPos = {p[1], p[2], EE_TARGET_Z}
    if USE_MUJOCO_TARGET_ORIENTATION then
        if TASK_FRAME_MODE == 'mujoco_attachment_dummy' then
            targetRot = TARGET_ATTACHMENT_ROTATION_WORLD
        else
            targetRot = TARGET_SITE_ROTATION_WORLD
        end
    else
        targetRot = r
    end
end

local function solve6(a, b)
    local n, m = 6, {}
    for i = 1, n do
        m[i] = {}
        for j = 1, n do m[i][j] = a[i][j] end
        m[i][n+1] = b[i]
    end
    for c = 1, n do
        local piv, best = c, math.abs(m[c][c])
        for r = c+1, n do
            if math.abs(m[r][c]) > best then piv, best = r, math.abs(m[r][c]) end
        end
        if piv ~= c then local tmp = m[c]; m[c] = m[piv]; m[piv] = tmp end
        local d = m[c][c]
        if math.abs(d) < 1e-10 then d = 1e-10; m[c][c] = d end
        for j = c, n+1 do m[c][j] = m[c][j] / d end
        for r = 1, n do
            if r ~= c then
                local f = m[r][c]
                for j = c, n+1 do m[r][j] = m[r][j] - f*m[c][j] end
            end
        end
    end
    local x = {}
    for i = 1, n do x[i] = m[i][n+1] end
    return x
end

local function poseError(pt, rt)
    local p, r = getPose()
    local er = rotError(r, rt)
    return {pt[1]-p[1], pt[2]-p[2], pt[3]-p[3], er[1], er[2], er[3]}, p, r
end

local function solveIk(seed, pt, rt, weightsOverride, posTol, rotTol)
    local q = {}
    for i = 1, 6 do q[i] = seed[i] end
    local eps, lambda = 1e-4, 0.035
    local weights = weightsOverride or {1.0, 1.0, 1.0, 2.5, 2.5, 2.5}
    local pTol = posTol or 7e-4
    local rTol = rotTol or 2e-3
    enforceShoulderPanLock(q)
    for _ = 1, 120 do
        setQ(q)
        local e, p, r = poseError(pt, rt)
        local perr, rerr = math.sqrt(e[1]^2 + e[2]^2 + e[3]^2), rotAngle(r, rt)
        if perr < pTol and rerr < rTol then return enforceShoulderPanLock(q), true, perr, rerr end
        local j = {{},{},{},{},{},{}}
        for c = 1, 6 do
            if LOCK_SHOULDER_PAN and c == SHOULDER_PAN_JOINT_INDEX then
                for rr = 1, 6 do j[rr][c] = 0.0 end
            else
                local qp = {}
                for k = 1, 6 do qp[k] = q[k] end
                qp[c] = qp[c] + eps
                setQ(qp)
                local pp, rp = getPose()
                local re = rotError(r, rp)
                j[1][c], j[2][c], j[3][c] = (pp[1]-p[1])/eps, (pp[2]-p[2])/eps, (pp[3]-p[3])/eps
                j[4][c], j[5][c], j[6][c] = re[1]/eps, re[2]/eps, re[3]/eps
            end
        end
        local a, b = {}, {}
        for c1 = 1, 6 do
            a[c1], b[c1] = {}, 0.0
            for c2 = 1, 6 do a[c1][c2] = 0.0 end
        end
        for rr = 1, 6 do
            local w = weights[rr]
            for c1 = 1, 6 do
                b[c1] = b[c1] + j[rr][c1] * w * e[rr]
                for c2 = 1, 6 do a[c1][c2] = a[c1][c2] + j[rr][c1] * w * j[rr][c2] end
            end
        end
        for c = 1, 6 do a[c][c] = a[c][c] + lambda*lambda end
        local dq = solve6(a, b)
        local s = norm(dq)
        local scale = 1.0
        if s > 0.10 then scale = 0.10 / s end
        for c = 1, 6 do q[c] = q[c] + scale*dq[c] end
        enforceShoulderPanLock(q)
    end
    setQ(q)
    local e, _, r = poseError(pt, rt)
    local perr = math.sqrt(e[1]^2 + e[2]^2 + e[3]^2)
    return enforceShoulderPanLock(q), false, perr, rotAngle(r, rt)
end

local function resolveHandles()
    for i, p in ipairs(JOINT_PATHS) do joints[i] = sim.getObject(p) end
    for _, p in ipairs({'/UR5/UR5_connection', ':/UR5/UR5_connection', '/UR5_connection', ':/UR5_connection'}) do
        local ok, h = pcall(sim.getObject, p)
        if ok then eeHandle = h; return end
    end
    eeHandle = joints[6]
end

local function configureTaskFrame()
    taskFrameSummary = {mode = TASK_FRAME_MODE, mujoco_attachment_dummy = false}
    if TASK_FRAME_MODE ~= 'mujoco_attachment_dummy' then
        return
    end

    local parentHandle = eeHandle
    local dummySize = math.max(EE_TRIAD_DUMMY_SIZE, 0.025)
    local dummy = sim.createDummy(dummySize)
    sim.setObjectAlias(dummy, 'real_cartpole_mujoco_attachment_site')
    local quat = TASK_FRAME_ATTACHMENT_QUAT_WXYZ
    local n = math.sqrt(quat[1]^2 + quat[2]^2 + quat[3]^2 + quat[4]^2)
    if n > 1e-12 then
        quat = {quat[1]/n, quat[2]/n, quat[3]/n, quat[4]/n}
    end
    local poseSet = false
    if sim.setObjectPose ~= nil and sim.handleflag_wxyzquat ~= nil then
        poseSet = pcall(function()
            sim.setObjectParent(dummy, parentHandle, true)
            sim.setObjectPose(
                dummy + sim.handleflag_wxyzquat,
                {
                    TASK_FRAME_ATTACHMENT_OFFSET[1],
                    TASK_FRAME_ATTACHMENT_OFFSET[2],
                    TASK_FRAME_ATTACHMENT_OFFSET[3],
                    quat[1], quat[2], quat[3], quat[4],
                },
                parentHandle
            )
        end)
    end
    if not poseSet then
        sim.setObjectParent(dummy, parentHandle, true)
        sim.setObjectPosition(dummy, parentHandle, TASK_FRAME_ATTACHMENT_OFFSET)
        sim.setObjectOrientation(dummy, parentHandle, {0.0, 0.0, math.pi * 0.5})
    end
    eeHandle = dummy
    taskFrameSummary = {
        mode = 'mujoco_attachment_dummy',
        mujoco_attachment_dummy = true,
        handle = eeHandle,
        parent_handle = parentHandle,
        local_offset_m = TASK_FRAME_ATTACHMENT_OFFSET,
        local_orientation_quat_wxyz = quat,
        use_mujoco_target_orientation = USE_MUJOCO_TARGET_ORIENTATION,
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
        createTriadMarkers(eeHandle, 'EE', EE_TRIAD_AXIS_LENGTH, EE_TRIAD_LINE_WIDTH, EE_TRIAD_DUMMY_SIZE, EE_TRIAD_ROOT_OFFSET)
end

local function createBaseTriad()
    if not SHOW_BASE_TRIAD then
        return
    end

    baseTriadRootHandle, baseTriadHandles, baseTriadDummyHandles, baseTriadShapeHandles =
        createTriadMarkers(robotModelHandle, 'Base', BASE_TRIAD_AXIS_LENGTH, BASE_TRIAD_LINE_WIDTH, BASE_TRIAD_DUMMY_SIZE, BASE_TRIAD_ROOT_OFFSET)
end

local function createCamera()
    local sensor = sim.createVisionSensor(1|2|4|128, {640,360,0,0}, {0.02,7.0,math.rad(62),0.1,0,0,0.78,0.82,0.86,0,0})
    sim.setObjectAlias(sensor, 'OriginAcquisitionCamera')
    return sim.getObject('/OriginAcquisitionCamera')
end

local function cameraMatrix(i, n, target)
    local yaw, radius = math.rad(-50), 2.05
    local cam = {target[1] + radius*math.cos(yaw), target[2] + radius*math.sin(yaw), target[3] + 0.40}
    local f = normalize({target[1]-cam[1], target[2]-cam[2], target[3]-cam[3]})
    local right = normalize(cross(f, {0,0,1}))
    local up = cross(right, f)
    return {right[1],up[1],f[1],cam[1], right[2],up[2],f[2],cam[2], right[3],up[3],f[3],cam[3]}
end

local function profile(t, duration, length)
    local vmax, amax = math.max(V_X_MAX, 1e-6), math.max(A_X_MAX, 1e-6)
    local tr = vmax / amax
    local dr = 0.5 * amax * tr * tr
    local vp, tf = vmax, 0.0
    if 2*dr >= length then
        tr = math.sqrt(length / amax)
        vp = amax * tr
    else
        tf = (length - 2*dr) / vmax
    end
    local natural = 2*tr + tf
    local scale = 1.0
    if natural > duration then scale = natural / duration end
    local a, v = amax*scale*scale, vp*scale
    tr, tf = tr/scale, tf/scale
    local d, vel = 0.0, 0.0
    if t < tr then
        d, vel = 0.5*a*t*t, a*t
    elseif t < tr+tf then
        d, vel = 0.5*a*tr*tr + v*(t-tr), v
    elseif t < 2*tr+tf then
        local td = t-tr-tf
        d, vel = 0.5*a*tr*tr + v*tf + v*td - 0.5*a*td*td, math.max(0, v-a*td)
    else
        d, vel = length, 0.0
    end
    return math.min(d / math.max(length, 1e-9), 1.0), vel, v, a
end

local function jarr(v)
    local out = {}
    for i, x in ipairs(v) do out[i] = string.format('%.9g', x) end
    return '[' .. table.concat(out, ',') .. ']'
end

local function j3(v)
    return string.format('[%.9g,%.9g,%.9g]', v[1], v[2], v[3])
end

local function solveTransportIk(seed, pt)
    return solveIk(
        seed,
        pt,
        targetRot,
        {12.0, 12.0, 12.0, ACCEL_ROT_WEIGHT, ACCEL_ROT_WEIGHT, ACCEL_ROT_WEIGHT},
        0.002,
        math.rad(ORIENTATION_TOL_DEG)
    )
end

local function scanDirection(dir)
    local qseed = ikSolution
    local lastGoodDx = 0.0
    local firstFailed = -1
    local mon = {u={}, x={}, err={}, ori={}, ok={}}
    setQ(ikSolution)
    local steps = math.max(1, math.floor(RANGE_SCAN_MAX_M / math.max(RANGE_SCAN_STEP_M, 1e-6)))
    for step = 1, steps do
        local dx = dir * RANGE_SCAN_STEP_M * step
        local pt = {targetPos[1], targetPos[2], targetPos[3]}
        pt[SWEEP_AXIS_INDEX] = pt[SWEEP_AXIS_INDEX] + dx
        local q, ok, pe, re = solveTransportIk(qseed, pt)
        setQ(q)
        local p = getPose()
        local criterionOk = ok and pe <= RANGE_POSITION_TOL_M and re <= math.rad(ORIENTATION_TOL_DEG)
        mon.u[#mon.u+1] = math.abs(dx)
        mon.x[#mon.x+1] = p[1]
        mon.err[#mon.err+1] = pe
        mon.ori[#mon.ori+1] = re * 180 / math.pi
        mon.ok[#mon.ok+1] = criterionOk and 1 or 0
        if criterionOk then
            lastGoodDx = dx
            qseed = q
        else
            firstFailed = step
            break
        end
    end
    return lastGoodDx, firstFailed, mon
end

local function discoverXRange()
    local posDx, posFail, posMon = scanDirection(1)
    local negDx, negFail, negMon = scanDirection(-1)
    local requestedDir = TARGET_DX >= 0 and 1 or -1
    local requestedAbs = math.abs(TARGET_DX)
    local availableAbs = requestedDir > 0 and math.abs(posDx) or math.abs(negDx)
    local selectedAbs = math.min(requestedAbs, availableAbs)
    effectiveTargetDx = requestedDir * selectedAbs
    if selectedAbs + 1e-12 < requestedAbs then rangeSummary.clamped = 1 else rangeSummary.clamped = 0 end
    rangeSummary.axis_index = SWEEP_AXIS_INDEX
    rangeSummary.axis_label = SWEEP_AXIS_LABEL
    rangeSummary.axis_min = targetPos[SWEEP_AXIS_INDEX] + negDx
    rangeSummary.axis_max = targetPos[SWEEP_AXIS_INDEX] + posDx
    rangeSummary.x_min = rangeSummary.axis_min
    rangeSummary.x_max = rangeSummary.axis_max
    rangeSummary.dx_neg = negDx
    rangeSummary.dx_pos = posDx
    rangeSummary.selected_dx = effectiveTargetDx
    rangeSummary.direction = requestedDir
    rangeSummary.first_failed_pos_step = posFail
    rangeSummary.first_failed_neg_step = negFail
    rangeSummary.pos_u, rangeSummary.pos_x, rangeSummary.pos_err = posMon.u, posMon.x, posMon.err
    rangeSummary.pos_ori_err_deg, rangeSummary.pos_ok = posMon.ori, posMon.ok
    rangeSummary.neg_u, rangeSummary.neg_x, rangeSummary.neg_err = negMon.u, negMon.x, negMon.err
    rangeSummary.neg_ori_err_deg, rangeSummary.neg_ok = negMon.ori, negMon.ok
end

local function buildAccelPath()
    accelPath, accelPathLength = {}, 0.0
    accelIkSuccess = true
    accelIkMaxPosErr, accelIkMaxRotErr = 0.0, 0.0
    accelFirstFailedWaypoint = -1
    accelMonitor = {u={}, target_x={}, target_y={}, target_z={}, actual_x={}, actual_y={}, actual_z={}, target_axis={}, actual_axis={}, pos_err={}, ori_err_deg={}, ik_ok={}}
    local qseed = ikSolution
    local prev = nil
    for i = 0, IK_WAYPOINTS-1 do
        local u = i / math.max(IK_WAYPOINTS-1, 1)
        local pt = {targetPos[1], targetPos[2], targetPos[3]}
        pt[SWEEP_AXIS_INDEX] = targetPos[SWEEP_AXIS_INDEX] + effectiveTargetDx*u
        local q, ok, pe, re = solveTransportIk(qseed, pt)
        accelIkSuccess = accelIkSuccess and ok
        if not ok and accelFirstFailedWaypoint < 0 then accelFirstFailedWaypoint = i end
        accelIkMaxPosErr = math.max(accelIkMaxPosErr, pe)
        accelIkMaxRotErr = math.max(accelIkMaxRotErr, re)
        setQ(q)
        local p = getPose()
        accelMonitor.u[#accelMonitor.u+1] = u
        accelMonitor.target_x[#accelMonitor.target_x+1] = pt[1]
        accelMonitor.target_y[#accelMonitor.target_y+1] = pt[2]
        accelMonitor.target_z[#accelMonitor.target_z+1] = pt[3]
        accelMonitor.actual_x[#accelMonitor.actual_x+1] = p[1]
        accelMonitor.actual_y[#accelMonitor.actual_y+1] = p[2]
        accelMonitor.actual_z[#accelMonitor.actual_z+1] = p[3]
        accelMonitor.target_axis[#accelMonitor.target_axis+1] = pt[SWEEP_AXIS_INDEX]
        accelMonitor.actual_axis[#accelMonitor.actual_axis+1] = p[SWEEP_AXIS_INDEX]
        accelMonitor.pos_err[#accelMonitor.pos_err+1] = pe
        accelMonitor.ori_err_deg[#accelMonitor.ori_err_deg+1] = re*180/math.pi
        accelMonitor.ik_ok[#accelMonitor.ik_ok+1] = ok and 1 or 0
        if prev then accelPathLength = accelPathLength + dist(prev, p) end
        accelPath[#accelPath+1] = {q=q, p=p, u=u, target_axis=pt[SWEEP_AXIS_INDEX]}
        qseed, prev = q, p
    end
end

local function buildMujocoLikeSweepPath()
    sweepPath, sweepPathLength = {}, 0.0
    sweepLegTargets, sweepLegFrameCounts, sweepLegStartXs = {}, {}, {}
    accelPath, accelPathLength = {}, 0.0
    accelIkSuccess = true
    accelIkMaxPosErr, accelIkMaxRotErr = 0.0, 0.0
    accelFirstFailedWaypoint = -1
    accelMonitor = {u={}, target_x={}, target_y={}, target_z={}, actual_x={}, actual_y={}, actual_z={}, target_axis={}, actual_axis={}, pos_err={}, ori_err_deg={}, ik_ok={}}

    local legCount = math.max(1, MUJOCO_LIKE_SWEEP_LEGS)
    local baseFrames = math.floor(FRAME_COUNT / legCount)
    local remainder = FRAME_COUNT - baseFrames * legCount
    local qseed = ikSolution
    local currentAxis = targetPos[SWEEP_AXIS_INDEX]
    local prevP = nil
    local globalFrame = 0

    for leg = 1, legCount do
        local goalAxis = targetPos[SWEEP_AXIS_INDEX] + ((leg % 2 == 1) and rangeSummary.dx_pos or rangeSummary.dx_neg)
        local legFrames = baseFrames + (leg <= remainder and 1 or 0)
        if legFrames < 1 then
            legFrames = 1
        end
        sweepLegTargets[#sweepLegTargets+1] = goalAxis
        sweepLegFrameCounts[#sweepLegFrameCounts+1] = legFrames
        sweepLegStartXs[#sweepLegStartXs+1] = currentAxis
        for frame = 0, legFrames - 1 do
            local u = frame / math.max(legFrames - 1, 1)
            local s = smoothstep(u)
            local axisValue = currentAxis + (goalAxis - currentAxis) * s
            local pt = {targetPos[1], targetPos[2], targetPos[3]}
            pt[SWEEP_AXIS_INDEX] = axisValue
            local q, ok, pe, re = solveTransportIk(qseed, pt)
            accelIkSuccess = accelIkSuccess and ok
            if not ok and accelFirstFailedWaypoint < 0 then
                accelFirstFailedWaypoint = globalFrame
            end
            accelIkMaxPosErr = math.max(accelIkMaxPosErr, pe)
            accelIkMaxRotErr = math.max(accelIkMaxRotErr, re)
            setQ(q)
            local p, r = getPose()
            local expected = {targetPos[1], targetPos[2], targetPos[3]}
            expected[SWEEP_AXIS_INDEX] = axisValue
            local pathErr = dist(p, expected)
            accelMonitor.u[#accelMonitor.u+1] = globalFrame / math.max(FRAME_COUNT - 1, 1)
            accelMonitor.target_x[#accelMonitor.target_x+1] = pt[1]
            accelMonitor.target_y[#accelMonitor.target_y+1] = pt[2]
            accelMonitor.target_z[#accelMonitor.target_z+1] = pt[3]
            accelMonitor.actual_x[#accelMonitor.actual_x+1] = p[1]
            accelMonitor.actual_y[#accelMonitor.actual_y+1] = p[2]
            accelMonitor.actual_z[#accelMonitor.actual_z+1] = p[3]
            accelMonitor.target_axis[#accelMonitor.target_axis+1] = axisValue
            accelMonitor.actual_axis[#accelMonitor.actual_axis+1] = p[SWEEP_AXIS_INDEX]
            accelMonitor.pos_err[#accelMonitor.pos_err+1] = pathErr
            accelMonitor.ori_err_deg[#accelMonitor.ori_err_deg+1] = re * 180 / math.pi
            accelMonitor.ik_ok[#accelMonitor.ik_ok+1] = ok and 1 or 0
            if prevP then
                sweepPathLength = sweepPathLength + dist(prevP, p)
            end
            local entry = {
                q = q,
                p = p,
                leg = leg,
                frame = frame,
                target_x = pt[1],
                target_y = targetPos[2],
                target_z = targetPos[3],
                target_axis = axisValue,
                ik_ok = ok and 1 or 0,
                pos_err = pathErr,
                ori_err_deg = re * 180 / math.pi,
            }
            sweepPath[#sweepPath+1] = entry
            accelPath[#accelPath+1] = entry
            prevP = p
            qseed = q
            globalFrame = globalFrame + 1
        end
        currentAxis = goalAxis
    end
    accelPathLength = sweepPathLength
end

local function qAccelAt(u)
    if u <= 0 then return accelPath[1].q end
    if u >= 1 then return accelPath[#accelPath].q end
    local f = u * (#accelPath-1)
    local i = math.floor(f) + 1
    local a = f - math.floor(f)
    local q, q0, q1 = {}, accelPath[i].q, accelPath[i+1].q
    for j = 1, 6 do q[j] = q0[j] + a * (q1[j] - q0[j]) end
    return q
end

local function axisLabel(index)
    if index == 1 then return 'x' end
    if index == 2 then return 'y' end
    return 'z'
end

local function writeSummary(finalP, finalR, finalQ, initialP, initialR, originReachedP, originReachedR, peakQd, peakVx, peakSpeed, plannedPeakV, plannedAccel)
    local posErr = dist(originReachedP, targetPos)
    local oriErr = rotAngle(originReachedR, targetRot)
    local targetAccelerationFinal = {targetPos[1], targetPos[2], targetPos[3]}
    targetAccelerationFinal[SWEEP_AXIS_INDEX] = targetPos[SWEEP_AXIS_INDEX] + effectiveTargetDx
    local sweepAxisName = axisLabel(SWEEP_AXIS_INDEX)
    local fixedAxis1Name = axisLabel(SWEEP_FIXED_AXIS_1)
    local fixedAxis2Name = axisLabel(SWEEP_FIXED_AXIS_2)
    local finalPosErr = dist(finalP, targetAccelerationFinal)
    local finalOriErr = rotAngle(finalR, targetRot)
    local startPosErr = dist(initialP, targetPos)
    local startOriErr = rotAngle(initialR, targetRot)
    local baseOnGround = math.abs(MODEL_BASE_Z_OFFSET) <= 1e-9
    local posOk = posErr <= POSITION_TOL_M
    local oriOk = oriErr <= math.rad(ORIENTATION_TOL_DEG)
    local accelPosOk = finalPosErr <= POSITION_TOL_M
    local accelOriOk = finalOriErr <= math.rad(ORIENTATION_TOL_DEG)
    local xNet = finalP[1] - originReachedP[1]
    local xErr = xNet - effectiveTargetDx
    local xOk = math.abs(xErr) <= math.max(0.005, 0.10 * math.abs(effectiveTargetDx))
    local yDrift = math.abs(finalP[2] - originReachedP[2])
    local zDrift = math.abs(finalP[3] - originReachedP[3])
    local sweepNet = finalP[SWEEP_AXIS_INDEX] - originReachedP[SWEEP_AXIS_INDEX]
    local sweepErr = sweepNet - effectiveTargetDx
    local sweepOk = math.abs(sweepErr) <= math.max(0.005, 0.10 * math.abs(effectiveTargetDx))
    local fixedDrift1 = math.abs(finalP[SWEEP_FIXED_AXIS_1] - originReachedP[SWEEP_FIXED_AXIS_1])
    local fixedDrift2 = math.abs(finalP[SWEEP_FIXED_AXIS_2] - originReachedP[SWEEP_FIXED_AXIS_2])
    local fixedTol1 = SWEEP_FIXED_AXIS_1 == 3 and 0.001 or 0.005
    local fixedTol2 = SWEEP_FIXED_AXIS_2 == 3 and 0.001 or 0.005
    local fixedOk1 = fixedDrift1 <= fixedTol1
    local fixedOk2 = fixedDrift2 <= fixedTol2
    local success = ikSuccess and accelIkSuccess and baseOnGround and posOk and oriOk and accelPosOk and accelOriOk and sweepOk and fixedOk1 and fixedOk2
    local reasons = {}
    if not ikSuccess then reasons[#reasons+1] = '"ik_failed"' end
    if not accelIkSuccess then reasons[#reasons+1] = '"acceleration_ik_failed"' end
    if not baseOnGround then reasons[#reasons+1] = '"base_not_on_ground"' end
    if not posOk then reasons[#reasons+1] = '"origin_position_error_too_large"' end
    if not oriOk then reasons[#reasons+1] = '"orientation_error_too_large"' end
    if not accelPosOk then reasons[#reasons+1] = '"accel_final_position_error_too_large"' end
    if not accelOriOk then reasons[#reasons+1] = '"accel_orientation_error_too_large"' end
    if not sweepOk then reasons[#reasons+1] = '"sweep_axis_tracking_error"' end
    if not fixedOk1 then reasons[#reasons+1] = '"sweep_fixed_axis_1_drift_too_large"' end
    if not fixedOk2 then reasons[#reasons+1] = '"sweep_fixed_axis_2_drift_too_large"' end
    local lines = {
        '{',
        '  "controller_name": "coppeliasim_origin_then_acceleration_controller",',
        '  "controller_family": "mujoco_origin_stabilization_then_acceleration_transport_port_position_servo_ik",',
        '  "stage": "origin_acquisition_gap_then_one_direction_acceleration",',
        '  "uses_position_servo_setpoints": true,',
        '  "uses_direct_torque_control": false,',
        '  "camera_fixed": true,',
        '  "ee_triad_visible": ' .. tostring(SHOW_EE_TRIAD) .. ',',
        '  "base_triad_visible": ' .. tostring(SHOW_BASE_TRIAD) .. ',',
        '  "sweep_axis_index": ' .. tostring(SWEEP_AXIS_INDEX) .. ',',
        '  "sweep_axis_label": "' .. sweepAxisName .. '",',
        '  "sweep_fixed_axis_1_index": ' .. tostring(SWEEP_FIXED_AXIS_1) .. ',',
        '  "sweep_fixed_axis_1_label": "' .. fixedAxis1Name .. '",',
        '  "sweep_fixed_axis_2_index": ' .. tostring(SWEEP_FIXED_AXIS_2) .. ',',
        '  "sweep_fixed_axis_2_label": "' .. fixedAxis2Name .. '",',
        '  "task_frame_mode": "' .. tostring(taskFrameSummary.mode or TASK_FRAME_MODE) .. '",',
        '  "task_frame_mujoco_attachment_dummy": ' .. tostring(taskFrameSummary.mujoco_attachment_dummy == true) .. ',',
        '  "use_mujoco_target_orientation": ' .. tostring(USE_MUJOCO_TARGET_ORIENTATION) .. ',',
        '  "lock_shoulder_pan": ' .. tostring(LOCK_SHOULDER_PAN) .. ',',
        '  "shoulder_pan_locked_rad": ' .. string.format('%.9g', shoulderPanLockedRad) .. ',',
        '  "mujoco_like_sweep": ' .. tostring(MUJOCO_LIKE_X_SWEEP) .. ',',
        '  "start_at_transport_plane": ' .. tostring(START_AT_TRANSPORT_PLANE) .. ',',
        '  "base_on_ground": ' .. tostring(baseOnGround) .. ',',
        '  "model_base_z_offset_m": ' .. string.format('%.9g', MODEL_BASE_Z_OFFSET) .. ',',
        '  "ee_target_z_m": ' .. string.format('%.9g', EE_TARGET_Z) .. ',',
        '  "ik_success": ' .. tostring(ikSuccess) .. ',',
        '  "acceleration_ik_success": ' .. tostring(accelIkSuccess) .. ',',
        '  "position_tolerance_m": ' .. string.format('%.9g', POSITION_TOL_M) .. ',',
        '  "orientation_tolerance_deg": ' .. string.format('%.9g', ORIENTATION_TOL_DEG) .. ',',
        '  "origin_position_ok": ' .. tostring(posOk) .. ',',
        '  "orientation_ok": ' .. tostring(oriOk) .. ',',
        '  "sweep_axis_tracking_ok": ' .. tostring(sweepOk) .. ',',
        '  "sweep_fixed_axis_1_ok": ' .. tostring(fixedOk1) .. ',',
        '  "sweep_fixed_axis_2_ok": ' .. tostring(fixedOk2) .. ',',
        '  "acceleration_orientation_ok": ' .. tostring(accelOriOk) .. ',',
        '  "success": ' .. tostring(success) .. ',',
        '  "failure_reasons": [' .. table.concat(reasons, ',') .. '],',
        '  "q_origin_source": "' .. Q_ORIGIN_SOURCE .. '",',
        '  "q_start_source": "' .. Q_START_SOURCE .. '",',
        '  "q_origin_rad": ' .. jarr(Q_ORIGIN) .. ',',
        '  "q_start_rad": ' .. jarr(Q_START) .. ',',
        '  "q_final_rad": ' .. jarr(finalQ) .. ',',
        '  "target_origin_world_m": ' .. j3(targetPos) .. ',',
        '  "target_acceleration_final_world_m": ' .. j3(targetAccelerationFinal) .. ',',
        '  "initial_ee_world_m": ' .. j3(initialP) .. ',',
        '  "origin_reached_ee_world_m": ' .. j3(originReachedP) .. ',',
        '  "final_ee_world_m": ' .. j3(finalP) .. ',',
        '  "initial_origin_position_error_m": ' .. string.format('%.9g', startPosErr) .. ',',
        '  "final_origin_position_error_m": ' .. string.format('%.9g', posErr) .. ',',
        '  "initial_orientation_error_deg": ' .. string.format('%.9g', startOriErr*180/math.pi) .. ',',
        '  "final_orientation_error_deg": ' .. string.format('%.9g', oriErr*180/math.pi) .. ',',
        '  "ik_final_position_error_m": ' .. string.format('%.9g', ikFinalPosErr) .. ',',
        '  "ik_final_orientation_error_deg": ' .. string.format('%.9g', ikFinalRotErr*180/math.pi) .. ',',
        '  "requested_target_axis_m": ' .. string.format('%.9g', TARGET_DX) .. ',',
        '  "target_axis_m": ' .. string.format('%.9g', effectiveTargetDx) .. ',',
        '  "target_axis_was_clamped_by_range_scan": ' .. tostring(rangeSummary.clamped == 1) .. ',',
        '  "range_scan_max_m": ' .. string.format('%.9g', RANGE_SCAN_MAX_M) .. ',',
        '  "range_scan_step_m": ' .. string.format('%.9g', RANGE_SCAN_STEP_M) .. ',',
        '  "range_scan_position_tolerance_m": ' .. string.format('%.9g', RANGE_POSITION_TOL_M) .. ',',
        '  "axis_reachable_min_m": ' .. string.format('%.9g', rangeSummary.axis_min) .. ',',
        '  "axis_reachable_max_m": ' .. string.format('%.9g', rangeSummary.axis_max) .. ',',
        '  "reachable_axis_negative_m": ' .. string.format('%.9g', rangeSummary.dx_neg) .. ',',
        '  "reachable_axis_positive_m": ' .. string.format('%.9g', rangeSummary.dx_pos) .. ',',
        '  "range_scan_first_failed_positive_step": ' .. tostring(rangeSummary.first_failed_pos_step) .. ',',
        '  "range_scan_first_failed_negative_step": ' .. tostring(rangeSummary.first_failed_neg_step) .. ',',
        '  "axis_net_displacement_m": ' .. string.format('%.9g', sweepNet) .. ',',
        '  "axis_tracking_error_m": ' .. string.format('%.9g', sweepErr) .. ',',
        '  "acceleration_final_position_error_m": ' .. string.format('%.9g', finalPosErr) .. ',',
        '  "acceleration_final_orientation_error_deg": ' .. string.format('%.9g', finalOriErr*180/math.pi) .. ',',
        '  "sweep_fixed_axis_1_drift_m": ' .. string.format('%.9g', fixedDrift1) .. ',',
        '  "sweep_fixed_axis_2_drift_m": ' .. string.format('%.9g', fixedDrift2) .. ',',
        '  "a_x_max_m_s2": ' .. string.format('%.9g', A_X_MAX) .. ',',
        '  "v_x_max_m_s": ' .. string.format('%.9g', V_X_MAX) .. ',',
        '  "acceleration_rotation_weight": ' .. string.format('%.9g', ACCEL_ROT_WEIGHT) .. ',',
        '  "planned_peak_speed_mps": ' .. string.format('%.9g', plannedPeakV) .. ',',
        '  "planned_accel_mps2": ' .. string.format('%.9g', plannedAccel) .. ',',
        '  "peak_abs_ee_vx_mps": ' .. string.format('%.9g', peakVx) .. ',',
        '  "peak_path_speed_mps": ' .. string.format('%.9g', peakSpeed) .. ',',
        '  "acceleration_path_length_m": ' .. string.format('%.9g', accelPathLength) .. ',',
        '  "acceleration_ik_waypoints": ' .. tostring(IK_WAYPOINTS) .. ',',
        '  "acceleration_first_failed_waypoint": ' .. tostring(accelFirstFailedWaypoint) .. ',',
        '  "acceleration_ik_max_waypoint_position_error_m": ' .. string.format('%.9g', accelIkMaxPosErr) .. ',',
        '  "acceleration_ik_max_waypoint_orientation_error_deg": ' .. string.format('%.9g', accelIkMaxRotErr*180/math.pi) .. ',',
        '  "sweep_leg_count": ' .. tostring(#sweepLegTargets) .. ',',
        '  "sweep_leg_target_axis_world_m": ' .. jarr(sweepLegTargets) .. ',',
        '  "sweep_leg_start_axis_world_m": ' .. jarr(sweepLegStartXs) .. ',',
        '  "sweep_leg_frame_count": ' .. jarr(sweepLegFrameCounts) .. ',',
        '  "sweep_path_length_m": ' .. string.format('%.9g', sweepPathLength) .. ',',
        '  "duration_s": ' .. string.format('%.9g', DURATION_S) .. ',',
        '  "fps": ' .. tostring(FPS) .. ',',
        '  "frames": ' .. tostring(FRAME_COUNT) .. ',',
        '  "origin_move_frames": ' .. tostring(ORIGIN_MOVE_FRAMES) .. ',',
        '  "gap_frames": ' .. tostring(GAP_FRAMES) .. ',',
        '  "gap_duration_s": ' .. string.format('%.9g', GAP_FRAMES / math.max(FPS, 1e-9)) .. ',',
        '  "accel_frames": ' .. tostring(ACCEL_FRAMES) .. ',',
        '  "peak_joint_speed_rad_s": ' .. string.format('%.9g', peakQd) .. ',',
        '  "time_s_trace": ' .. jarr(traces.t) .. ',',
        '  "phase_trace": ' .. jarr(traces.phase) .. ',',
        '  "origin_position_error_trace_m": ' .. jarr(traces.pos_err) .. ',',
        '  "orientation_error_trace_deg": ' .. jarr(traces.ori_err) .. ',',
        '  "ee_x_trace": ' .. jarr(traces.x) .. ',',
        '  "ee_y_trace": ' .. jarr(traces.y) .. ',',
        '  "ee_z_trace": ' .. jarr(traces.z) .. ',',
        '  "ee_vx_trace_mps": ' .. jarr(traces.vx) .. ',',
        '  "path_speed_trace_mps": ' .. jarr(traces.speed) .. ',',
        '  "acceleration_progress_trace": ' .. jarr(traces.accel_progress) .. ',',
        '  "joint_speed_max_trace_rad_s": ' .. jarr(traces.qdmax) .. ',',
        '  "range_scan_positive_u": ' .. jarr(rangeSummary.pos_u) .. ',',
        '  "range_scan_positive_actual_x": ' .. jarr(rangeSummary.pos_x) .. ',',
        '  "range_scan_positive_position_error_m": ' .. jarr(rangeSummary.pos_err) .. ',',
        '  "range_scan_positive_orientation_error_deg": ' .. jarr(rangeSummary.pos_ori_err_deg) .. ',',
        '  "range_scan_positive_ok": ' .. jarr(rangeSummary.pos_ok) .. ',',
        '  "range_scan_negative_u": ' .. jarr(rangeSummary.neg_u) .. ',',
        '  "range_scan_negative_actual_x": ' .. jarr(rangeSummary.neg_x) .. ',',
        '  "range_scan_negative_position_error_m": ' .. jarr(rangeSummary.neg_err) .. ',',
        '  "range_scan_negative_orientation_error_deg": ' .. jarr(rangeSummary.neg_ori_err_deg) .. ',',
        '  "range_scan_negative_ok": ' .. jarr(rangeSummary.neg_ok) .. ',',
        '  "accel_waypoint_u": ' .. jarr(accelMonitor.u) .. ',',
        '  "accel_waypoint_target_axis": ' .. jarr(accelMonitor.target_axis) .. ',',
        '  "accel_waypoint_actual_axis": ' .. jarr(accelMonitor.actual_axis) .. ',',
        '  "accel_waypoint_target_x": ' .. jarr(accelMonitor.target_x) .. ',',
        '  "accel_waypoint_target_y": ' .. jarr(accelMonitor.target_y) .. ',',
        '  "accel_waypoint_target_z": ' .. jarr(accelMonitor.target_z) .. ',',
        '  "accel_waypoint_actual_x": ' .. jarr(accelMonitor.actual_x) .. ',',
        '  "accel_waypoint_actual_y": ' .. jarr(accelMonitor.actual_y) .. ',',
        '  "accel_waypoint_actual_z": ' .. jarr(accelMonitor.actual_z) .. ',',
        '  "accel_waypoint_position_error_m": ' .. jarr(accelMonitor.pos_err) .. ',',
        '  "accel_waypoint_orientation_error_deg": ' .. jarr(accelMonitor.ori_err_deg) .. ',',
        '  "accel_waypoint_ik_ok": ' .. jarr(accelMonitor.ik_ok) .. ',',
        '  "video_path": "' .. VIDEO_PATH .. '"',
        '}',
        '',
    }
    writeText(SUMMARY_PATH, table.concat(lines, '\n'))
end

local function writeSweepSummary(rows, bestIndex)
    local z, ok, dxNeg, dxPos, span = {}, {}, {}, {}, {}
    local xMin, xMax, posFail, negFail = {}, {}, {}, {}
    local originPe, originRe = {}, {}
    for i, r in ipairs(rows) do
        z[#z+1] = r.z
        ok[#ok+1] = r.ok and 1 or 0
        dxNeg[#dxNeg+1] = r.dxNeg
        dxPos[#dxPos+1] = r.dxPos
        span[#span+1] = r.span
        xMin[#xMin+1] = r.xMin
        xMax[#xMax+1] = r.xMax
        posFail[#posFail+1] = r.posFail
        negFail[#negFail+1] = r.negFail
        originPe[#originPe+1] = r.originPe
        originRe[#originRe+1] = r.originReDeg
    end
    local best = rows[bestIndex]
    local lines = {
        '{',
        '  "controller_name": "coppeliasim_x_range_height_sweep",',
        '  "controller_family": "mujoco_fixed_z_reachable_x_range_scan_port",',
        '  "base_on_ground": ' .. tostring(math.abs(MODEL_BASE_Z_OFFSET) <= 1e-9) .. ',',
        '  "model_base_z_offset_m": ' .. string.format('%.9g', MODEL_BASE_Z_OFFSET) .. ',',
        '  "position_tolerance_m": ' .. string.format('%.9g', POSITION_TOL_M) .. ',',
        '  "orientation_tolerance_deg": ' .. string.format('%.9g', ORIENTATION_TOL_DEG) .. ',',
        '  "range_scan_position_tolerance_m": ' .. string.format('%.9g', RANGE_POSITION_TOL_M) .. ',',
        '  "range_scan_step_m": ' .. string.format('%.9g', RANGE_SCAN_STEP_M) .. ',',
        '  "range_scan_max_m": ' .. string.format('%.9g', RANGE_SCAN_MAX_M) .. ',',
        '  "sweep_z_min_m": ' .. string.format('%.9g', SWEEP_Z_MIN) .. ',',
        '  "sweep_z_max_m": ' .. string.format('%.9g', SWEEP_Z_MAX) .. ',',
        '  "sweep_z_step_m": ' .. string.format('%.9g', SWEEP_Z_STEP) .. ',',
        '  "best_height_m": ' .. string.format('%.9g', best and best.z or 0.0) .. ',',
        '  "best_x_span_m": ' .. string.format('%.9g', best and best.span or 0.0) .. ',',
        '  "best_x_min_m": ' .. string.format('%.9g', best and best.xMin or 0.0) .. ',',
        '  "best_x_max_m": ' .. string.format('%.9g', best and best.xMax or 0.0) .. ',',
        '  "best_reachable_dx_negative_m": ' .. string.format('%.9g', best and best.dxNeg or 0.0) .. ',',
        '  "best_reachable_dx_positive_m": ' .. string.format('%.9g', best and best.dxPos or 0.0) .. ',',
        '  "height_m": ' .. jarr(z) .. ',',
        '  "origin_ik_ok": ' .. jarr(ok) .. ',',
        '  "origin_ik_position_error_m": ' .. jarr(originPe) .. ',',
        '  "origin_ik_orientation_error_deg": ' .. jarr(originRe) .. ',',
        '  "reachable_dx_negative_m": ' .. jarr(dxNeg) .. ',',
        '  "reachable_dx_positive_m": ' .. jarr(dxPos) .. ',',
        '  "x_span_m": ' .. jarr(span) .. ',',
        '  "x_min_m": ' .. jarr(xMin) .. ',',
        '  "x_max_m": ' .. jarr(xMax) .. ',',
        '  "first_failed_positive_step": ' .. jarr(posFail) .. ',',
        '  "first_failed_negative_step": ' .. jarr(negFail) .. '',
        '}',
        '',
    }
    writeText(SUMMARY_PATH, table.concat(lines, '\n'))
end

local function runRangeSweep()
    refreshTargetPoseFromOrigin()
    local originPos = {targetPos[1], targetPos[2], targetPos[3]}
    local rows, bestIndex, bestSpan = {}, 1, -1.0
    local count = math.floor((SWEEP_Z_MAX - SWEEP_Z_MIN) / math.max(SWEEP_Z_STEP, 1e-6) + 0.5)
    for i = 0, count do
        local z = SWEEP_Z_MIN + i * SWEEP_Z_STEP
        if z > SWEEP_Z_MAX + 1e-9 then z = SWEEP_Z_MAX end
        targetPos = {originPos[1], originPos[2], z}
        setQ(Q_START)
        local q, ok, pe, re = solveIk(Q_START, targetPos, targetRot)
        ikSolution = q
        setQ(q)
        local originOk = ok and pe <= POSITION_TOL_M and re <= math.rad(ORIENTATION_TOL_DEG)
        local row = {
            z=z, ok=originOk, dxNeg=0.0, dxPos=0.0, span=0.0,
            xMin=targetPos[1], xMax=targetPos[1], posFail=-1, negFail=-1,
            originPe=pe, originReDeg=re*180/math.pi,
        }
        if originOk then
            discoverXRange()
            row.dxNeg = rangeSummary.dx_neg
            row.dxPos = rangeSummary.dx_pos
            row.xMin = rangeSummary.x_min
            row.xMax = rangeSummary.x_max
            row.posFail = rangeSummary.first_failed_pos_step
            row.negFail = rangeSummary.first_failed_neg_step
            row.span = row.xMax - row.xMin
        end
        rows[#rows+1] = row
        if row.span > bestSpan then
            bestSpan = row.span
            bestIndex = #rows
        end
    end
    writeSweepSummary(rows, bestIndex)
    writeText(DONE_MARKER, 'done\n')
    sim.quitSimulator()
end

local function qAt(u)
    local q = {}
    local s = smoothstep(u)
    for i = 1, 6 do q[i] = Q_START[i] + s * (ikSolution[i] - Q_START[i]) end
    return q
end

local function captureMujocoLikeSweep()
    refreshTargetPoseFromOrigin()
    if START_AT_TRANSPORT_PLANE then
        local planeQ, planeOk, planePosErr, planeRotErr = solveIk(Q_ORIGIN, targetPos, targetRot)
        Q_START = planeQ
        if planeOk and planePosErr <= POSITION_TOL_M and planeRotErr <= math.rad(ORIENTATION_TOL_DEG) then
            Q_START_SOURCE = 'transport_plane_start_from_origin_ik_solution'
        else
            Q_START_SOURCE = 'transport_plane_start_from_origin_ik_solution_approx'
        end
    end
    setQ(Q_START)
    local initialP, initialR = getPose()
    ikSolution, ikSuccess, ikFinalPosErr, ikFinalRotErr = solveIk(Q_START, targetPos, targetRot)
    discoverXRange()
    effectiveTargetDx = rangeSummary.dx_pos
    rangeSummary.selected_dx = effectiveTargetDx
    buildMujocoLikeSweepPath()
    setQ(Q_START)

    visionSensor = createCamera()
    writeText(SENSING_MARKER, 'capture\n')
    local center = {
        targetPos[1],
        targetPos[2],
        targetPos[3] + 0.03,
    }
    center[SWEEP_AXIS_INDEX] = targetPos[SWEEP_AXIS_INDEX] + 0.5 * (rangeSummary.dx_neg + rangeSummary.dx_pos)
    local cameraPose = cameraMatrix(0, FRAME_COUNT, center)
    local prevQ, prevP, prevT = nil, nil, nil
    local peakQd = 0.0
    local peakVx, peakSpeed = 0.0, 0.0
    local plannedPeakV, plannedAccel = 0.0, 0.0
    local finalP, finalR, finalQ = initialP, initialR, Q_START

    for i = 1, #accelPath do
        local entry = accelPath[i]
        local t = DURATION_S * (i - 1) / math.max(FRAME_COUNT - 1, 1)
        local q = entry.q
        setQ(q)
        local p, r = getPose()
        local targetNow = {entry.target_x, entry.target_y, entry.target_z}
        local posErr = dist(p, targetNow)
        local oriErr = rotAngle(r, targetRot)
        local qdmax, vx, speed = 0.0, 0.0, 0.0
        if prevQ and prevP then
            local dt = math.max(t - prevT, 1e-9)
            for j = 1, 6 do qdmax = math.max(qdmax, math.abs((q[j] - prevQ[j]) / dt)) end
            vx = (p[1] - prevP[1]) / dt
            speed = dist(p, prevP) / dt
        end
        peakQd = math.max(peakQd, qdmax)
        peakVx = math.max(peakVx, math.abs(vx))
        peakSpeed = math.max(peakSpeed, speed)
        plannedPeakV = math.max(plannedPeakV, math.abs(vx))
        if i == 1 then
            initialP, initialR = p, r
        end
        traces.t[#traces.t+1], traces.pos_err[#traces.pos_err+1], traces.ori_err[#traces.ori_err+1] = t, posErr, oriErr * 180 / math.pi
        traces.phase[#traces.phase+1] = entry.leg
        traces.x[#traces.x+1], traces.y[#traces.y+1], traces.z[#traces.z+1], traces.qdmax[#traces.qdmax+1] = p[1], p[2], p[3], qdmax
        traces.vx[#traces.vx+1], traces.speed[#traces.speed+1], traces.accel_progress[#traces.accel_progress+1] = vx, speed, (i - 1) / math.max(FRAME_COUNT - 1, 1)
        sim.setObjectMatrix(visionSensor, cameraPose, sim.handle_world)
        sim.handleVisionSensor(visionSensor)
        local img, res = sim.getVisionSensorImg(visionSensor)
        img = sim.transformImage(img, res, 4)
        sim.saveImage(img, res, 0, string.format('%s/%s_%08d.png', OUTPUT_DIR, FRAME_PREFIX, i - 1), -1)
        prevQ, prevP, prevT = q, p, t
        finalP, finalR, finalQ = p, r, q
    end
    plannedAccel = plannedPeakV
    writeSummary(finalP, finalR, finalQ, initialP, initialR, initialP, initialR, peakQd, peakVx, peakSpeed, plannedPeakV, plannedAccel)
    writeText(DONE_MARKER, 'done\n')
    sim.quitSimulator()
end

local function capture()
    if MUJOCO_LIKE_X_SWEEP then
        captureMujocoLikeSweep()
        return
    end
    refreshTargetPoseFromOrigin()
    if START_AT_TRANSPORT_PLANE then
        local planeQ, planeOk, planePosErr, planeRotErr = solveIk(Q_ORIGIN, targetPos, targetRot)
        Q_START = planeQ
        if planeOk and planePosErr <= POSITION_TOL_M and planeRotErr <= math.rad(ORIENTATION_TOL_DEG) then
            Q_START_SOURCE = 'transport_plane_start_from_origin_ik_solution'
        else
            Q_START_SOURCE = 'transport_plane_start_from_origin_ik_solution_approx'
        end
    end
    setQ(Q_START)
    local initialP, initialR = getPose()
    ikSolution, ikSuccess, ikFinalPosErr, ikFinalRotErr = solveIk(Q_START, targetPos, targetRot)
    discoverXRange()
    buildAccelPath()
    setQ(Q_START)

    visionSensor = createCamera()
    writeText(SENSING_MARKER, 'capture\n')
    local center = {
        0.5*(initialP[1] + targetPos[1] + effectiveTargetDx),
        0.5*(initialP[2] + targetPos[2]),
        0.5*(initialP[3] + targetPos[3]) + 0.03,
    }
    local cameraPose = cameraMatrix(0, FRAME_COUNT, center)
    local prevQ, prevP, prevT = nil, nil, nil
    local peakQd = 0.0
    local peakVx, peakSpeed = 0.0, 0.0
    local plannedPeakV, plannedAccel = 0.0, 0.0
    local finalP, finalR, finalQ = initialP, initialR, Q_START
    local originReachedP, originReachedR = initialP, initialR
    for i = 0, FRAME_COUNT-1 do
        local t = DURATION_S * i / math.max(FRAME_COUNT-1, 1)
        local phase, accelProgress = 0, 0.0
        local q = nil
        if i < ORIGIN_MOVE_FRAMES then
            phase = 0
            local u = clamp(i / math.max(ORIGIN_MOVE_FRAMES-1, 1), 0.0, 1.0)
            q = qAt(u)
        elseif i < ORIGIN_MOVE_FRAMES + GAP_FRAMES then
            phase = 1
            q = ikSolution
        else
            phase = 2
            local accelIdx = i - ORIGIN_MOVE_FRAMES - GAP_FRAMES
            local accelT = accelIdx / math.max(FPS, 1e-9)
            local accelDuration = math.max((ACCEL_FRAMES - 1) / math.max(FPS, 1e-9), 1e-9)
            local u, _, pv, pa = profile(accelT, accelDuration, accelPathLength)
            plannedPeakV, plannedAccel = math.max(plannedPeakV, pv), math.max(plannedAccel, pa)
            accelProgress = u
            q = qAccelAt(u)
        end
        setQ(q)
        local p, r = getPose()
        local posErr = dist(p, targetPos)
        local oriErr = rotAngle(r, targetRot)
        local qdmax, vx, speed = 0.0, 0.0, 0.0
        if prevQ and prevP then
            local dt = math.max(t - prevT, 1e-9)
            for j = 1, 6 do qdmax = math.max(qdmax, math.abs((q[j] - prevQ[j]) / dt)) end
            vx = (p[1] - prevP[1]) / dt
            speed = dist(p, prevP) / dt
        end
        peakQd = math.max(peakQd, qdmax)
        peakVx = math.max(peakVx, math.abs(vx))
        peakSpeed = math.max(peakSpeed, speed)
        if i == ORIGIN_MOVE_FRAMES - 1 then
            originReachedP, originReachedR = p, r
        end
        traces.t[#traces.t+1], traces.pos_err[#traces.pos_err+1], traces.ori_err[#traces.ori_err+1] = t, posErr, oriErr*180/math.pi
        traces.phase[#traces.phase+1] = phase
        traces.x[#traces.x+1], traces.y[#traces.y+1], traces.z[#traces.z+1], traces.qdmax[#traces.qdmax+1] = p[1], p[2], p[3], qdmax
        traces.vx[#traces.vx+1], traces.speed[#traces.speed+1], traces.accel_progress[#traces.accel_progress+1] = vx, speed, accelProgress
        sim.setObjectMatrix(visionSensor, cameraPose, sim.handle_world)
        sim.handleVisionSensor(visionSensor)
        local img, res = sim.getVisionSensorImg(visionSensor)
        img = sim.transformImage(img, res, 4)
        sim.saveImage(img, res, 0, string.format('%s/%s_%08d.png', OUTPUT_DIR, FRAME_PREFIX, i), -1)
        prevQ, prevP, prevT = q, p, t
        finalP, finalR, finalQ = p, r, q
    end
    writeSummary(finalP, finalR, finalQ, initialP, initialR, originReachedP, originReachedR, peakQd, peakVx, peakSpeed, plannedPeakV, plannedAccel)
    writeText(DONE_MARKER, 'done\n')
    sim.quitSimulator()
end

function sysCall_init()
    applyEnvOverrides()
    writeText(LOAD_MARKER, 'loaded\n')
    writeText(START_MARKER, 'init\n')
    sim.addLog(sim.verbosity_scriptinfos, 'UR5 origin acquisition add-on starting')
    sim.loadScene(SCENE_PATH)
    sim.setColorProperty(sim.handle_scene, 'ambientLight', {0.65, 0.65, 0.65}, {noError = true})
    robotModelHandle = sim.loadModel(MODEL_PATH)
    if robotModelHandle and robotModelHandle >= 0 and MODEL_BASE_Z_OFFSET ~= 0.0 then
        sim.setObjectPosition(robotModelHandle, sim.handle_world, {0.0, 0.0, MODEL_BASE_Z_OFFSET})
    end
    resolveHandles()
    configureTaskFrame()
    createEeTriad()
    createBaseTriad()
    if RANGE_SWEEP_ONLY then
        runRangeSweep()
    else
        capture()
    end
end

function sysCall_nonSimulation()
end

function sysCall_cleanup()
    sim.addLog(sim.verbosity_scriptinfos, 'UR5 origin acquisition cleanup')
end
