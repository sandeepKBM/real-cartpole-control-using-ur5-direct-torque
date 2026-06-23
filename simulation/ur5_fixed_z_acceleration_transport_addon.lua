local COPPELIA_ROOT = os.getenv('COPPELIA_ROOT') or '/common/users/ss5772/real_Cartpole/third_party/coppelia_runtime/CoppeliaSim_Edu_V4_10_0_rev0_Ubuntu24_04'
local ROOT = os.getenv('REAL_CARTPOLE_ROOT') or '/common/users/ss5772/real_Cartpole'
local SCENE_PATH = COPPELIA_ROOT .. '/system/dfltscn.ttt'
local MODEL_PATH = COPPELIA_ROOT .. '/models/robots/non-mobile/UR5.ttm'
local OUTPUT_DIR = ROOT .. '/outputs/control_runs/coppelia_acceleration_transport_frames'
local STATE_DIR = ROOT .. '/outputs/control_runs/coppelia_acceleration_transport_state'
local VIDEO_PATH = ROOT .. '/demonstration_videos/ur5e_coppeliasim/coppeliasim_ur5_acceleration_transport.mp4'
local SUMMARY_PATH = STATE_DIR .. '/coppeliasim_ur5_acceleration_transport_summary.json'
local FRAME_PREFIX = 'frame'
local LOAD_MARKER = STATE_DIR .. '/ur5_acceleration_transport_addon_loaded.txt'
local START_MARKER = STATE_DIR .. '/ur5_acceleration_transport_addon_started.txt'
local SENSING_MARKER = STATE_DIR .. '/ur5_acceleration_transport_addon_sensing.txt'
local DONE_MARKER = STATE_DIR .. '/ur5_acceleration_transport_done.txt'

local FRAME_COUNT = tonumber(os.getenv('FRAME_COUNT') or '100')
local FPS = tonumber(os.getenv('FPS') or '25')
local DURATION_S = tonumber(os.getenv('MOVE_DURATION_S') or tostring(FRAME_COUNT / FPS))
local V_X_MAX = tonumber(os.getenv('V_X_MAX_MPS') or '0.35')
local A_X_MAX = tonumber(os.getenv('A_X_MAX_MPS2') or '1.2')
local TARGET_DX = tonumber(os.getenv('TARGET_DX_M') or '0.01')
local IK_WAYPOINTS = tonumber(os.getenv('IK_WAYPOINTS') or '72')
local MODEL_BASE_Z_OFFSET = tonumber(os.getenv('MODEL_BASE_Z_OFFSET_M') or '0.0')
local TASK_FRAME_MODE = string.lower(os.getenv('TASK_FRAME_MODE') or 'mujoco_attachment_dummy')
-- Keep the height-matched proxy offset that aligns the raw flange with the
-- MuJoCo fixed-Z seed, but rotate it into the MuJoCo attachment_site frame.
local TASK_FRAME_ATTACHMENT_OFFSET = {0.0, 0.0, -0.2}
local TASK_FRAME_ATTACHMENT_QUAT_WXYZ = {
    -1.0,
    1.0,
    0.0,
    0.0,
}
local TARGET_SITE_ROTATION_WORLD = {
    -1.0, 0.0, 0.0, 0.0,
    0.0, 0.0, -1.0, 0.0,
    0.0, -1.0, 0.0, 0.0,
}

-- Attachment-frame world rotation used for reporting guardrails.
-- Observed from the Coppelia pose matrices as the tool target rotated +90 deg
-- about local Z.
local TARGET_ATTACHMENT_ROTATION_WORLD = {
    0.0, 1.0, 0.0, 0.0,
    0.0, 0.0, -1.0, 0.0,
    -1.0, 0.0, 0.0, 0.0,
}

local IK_ROT_TOL = tonumber(os.getenv('IK_ROT_TOL_RAD') or tostring(math.pi / 2 + 0.05))

local JOINT_PATHS = {
    '/UR5/joint',
    '/UR5/link/joint',
    '/UR5/link/link/joint',
    '/UR5/link/link/link/joint',
    '/UR5/link/link/link/link/joint',
    '/UR5/link/link/link/link/link/joint',
}

local Q_START = {
    0.0,
    -0.1133064268431449,
    -0.664621645801302,
    4.921777393344012,
    -6.283185307179586,
    5.280928640069786,
}
local Q_START_SOURCE = 'mujoco_fixed_z_transport_start'

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

do
    local qOverride, parseErr = parseJointVectorEnv('Q_START_RAD', Q_START)
    Q_START = qOverride
    if parseErr == nil and os.getenv('Q_START_RAD') ~= nil and os.getenv('Q_START_RAD') ~= '' then
        Q_START_SOURCE = 'env_Q_START_RAD'
    elseif parseErr ~= nil then
        Q_START_SOURCE = 'mujoco_fixed_z_transport_start_env_parse_error_' .. parseErr
    end
end

local sim = require 'sim'
local joints = {}
local eeHandle = -1
local rawEeHandle = -1
local rawEeResolvedPath = ''
local taskFrameHandle = -1
local taskFrameResolvedPath = ''
local taskFrameDummyActive = false
local visionSensor = -1
local targetRot = nil
local captureStartMatrix = nil
local captureEndMatrix = nil
local path = {}
local pathLength = 0.0
local ikSuccess = true
local ikMaxPosErr = 0.0
local ikMaxRotErr = 0.0
local traces = {t={}, x={}, y={}, z={}, vx={}, speed={}, progress={}, ori={}, qdmax={}}

function sysCall_info()
    return {autoStart = true}
end

local function writeText(pathName, text)
    local f = io.open(pathName, 'w')
    if f then f:write(text or ''); f:close() end
end

local function normalizeQuatWxyz(quat)
    local n = math.sqrt(quat[1]^2 + quat[2]^2 + quat[3]^2 + quat[4]^2)
    if n < 1e-12 then
        return {1.0, 0.0, 0.0, 0.0}
    end
    return {quat[1]/n, quat[2]/n, quat[3]/n, quat[4]/n}
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

local function setQ(q)
    for i, h in ipairs(joints) do sim.setJointPosition(h, q[i]) end
end

local function getQ()
    local q = {}
    for i, h in ipairs(joints) do q[i] = sim.getJointPosition(h) end
    return q
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

local function solveIk(seed, pt, rt)
    local q = {}
    for i = 1, 6 do q[i] = seed[i] end
    local eps, lambda = 1e-4, 0.035
    local weights = {1.0, 12.0, 12.0, 2.5, 2.5, 2.5}
    for _ = 1, 100 do
        setQ(q)
        local e, p, r = poseError(pt, rt)
        local perr, rerr = math.sqrt(e[1]^2 + e[2]^2 + e[3]^2), rotAngle(r, rt)
        if perr < 7e-4 and rerr < IK_ROT_TOL then return q, true, perr, rerr end
        local j = {{},{},{},{},{},{}}
        for c = 1, 6 do
            local qp = {}
            for k = 1, 6 do qp[k] = q[k] end
            qp[c] = qp[c] + eps
            setQ(qp)
            local pp, rp = getPose()
            local re = rotError(r, rp)
            j[1][c], j[2][c], j[3][c] = (pp[1]-p[1])/eps, (pp[2]-p[2])/eps, (pp[3]-p[3])/eps
            j[4][c], j[5][c], j[6][c] = re[1]/eps, re[2]/eps, re[3]/eps
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
    end
    setQ(q)
    local e, _, r = poseError(pt, rt)
    local perr = math.sqrt(e[1]^2 + e[2]^2 + e[3]^2)
    return q, false, perr, rotAngle(r, rt)
end

local function resolveHandles()
    for i, p in ipairs(JOINT_PATHS) do joints[i] = sim.getObject(p) end
    for _, p in ipairs({'/UR5/UR5_connection', ':/UR5/UR5_connection', '/UR5_connection', ':/UR5_connection'}) do
        local ok, h = pcall(sim.getObject, p)
        if ok then
            rawEeHandle = h
            rawEeResolvedPath = p
            eeHandle = h
            taskFrameHandle = h
            taskFrameResolvedPath = p
            return
        end
    end
    rawEeHandle = joints[6]
    rawEeResolvedPath = JOINT_PATHS[#JOINT_PATHS]
    eeHandle = rawEeHandle
    taskFrameHandle = rawEeHandle
    taskFrameResolvedPath = rawEeResolvedPath
end

local function configureTaskFrame()
    if TASK_FRAME_MODE ~= 'mujoco_attachment_dummy' then
        taskFrameDummyActive = false
        return
    end
    if rawEeHandle < 0 then
        sim.addLog(sim.verbosity_warnings, 'UR5 fixed-Z acceleration: no raw EE handle available for attachment dummy')
        taskFrameDummyActive = false
        return
    end

    local dummy = sim.createDummy(0.025)
    if dummy < 0 then
        sim.addLog(sim.verbosity_warnings, 'UR5 fixed-Z acceleration: failed to create MuJoCo attachment-site dummy')
        taskFrameDummyActive = false
        return
    end

    local quat = normalizeQuatWxyz(TASK_FRAME_ATTACHMENT_QUAT_WXYZ)
    local pose = {
        TASK_FRAME_ATTACHMENT_OFFSET[1],
        TASK_FRAME_ATTACHMENT_OFFSET[2],
        TASK_FRAME_ATTACHMENT_OFFSET[3],
        quat[1],
        quat[2],
        quat[3],
        quat[4],
    }

    pcall(sim.setObjectAlias, dummy, 'real_cartpole_mujoco_attachment_site')
    pcall(sim.setObjectParent, dummy, rawEeHandle, false)

    local poseHandle = dummy
    if sim.handleflag_wxyzquat ~= nil then
        poseHandle = dummy + sim.handleflag_wxyzquat
    end
    local ok = pcall(function()
        sim.setObjectPose(poseHandle, pose, rawEeHandle)
    end)
    if not ok then
        sim.addLog(sim.verbosity_warnings, 'UR5 fixed-Z acceleration: failed to place MuJoCo attachment-site dummy; falling back to raw EE proxy')
        pcall(sim.removeObject, dummy)
        taskFrameDummyActive = false
        return
    end

    eeHandle = dummy
    taskFrameHandle = dummy
    taskFrameResolvedPath = rawEeResolvedPath .. '/real_cartpole_mujoco_attachment_site'
    taskFrameDummyActive = true
    sim.addLog(sim.verbosity_scriptinfos, 'UR5 fixed-Z acceleration: using MuJoCo attachment-site dummy task frame')
end

local function buildPath()
    setQ(Q_START)
    local p0, r0 = getPose()
    targetRot = TARGET_SITE_ROTATION_WORLD
    path, pathLength, ikSuccess = {}, 0.0, true
    local qseed = getQ()
    local prev = nil
    for i = 0, IK_WAYPOINTS-1 do
        local u = i / (IK_WAYPOINTS-1)
        local pt = {p0[1] + TARGET_DX*u, p0[2], p0[3]}
        local q, ok, pe, re = solveIk(qseed, pt, targetRot)
        ikSuccess = ikSuccess and ok
        ikMaxPosErr, ikMaxRotErr = math.max(ikMaxPosErr, pe), math.max(ikMaxRotErr, re)
        setQ(q)
        local p = getPose()
        if prev then pathLength = pathLength + dist(prev, p) end
        path[#path+1] = {q=q, p=p, u=u}
        qseed, prev = q, p
    end
end

local function qAt(u)
    if u <= 0 then return path[1].q end
    if u >= 1 then return path[#path].q end
    local f = u * (#path-1)
    local i = math.floor(f) + 1
    local a = f - math.floor(f)
    local q, q0, q1 = {}, path[i].q, path[i+1].q
    for j = 1, 6 do q[j] = q0[j] + a*(q1[j]-q0[j]) end
    return q
end

local function profile(t, duration, length)
    local vmax, amax = math.max(V_X_MAX, 1e-6), math.max(A_X_MAX, 1e-6)
    local tr = vmax / amax
    local dr = 0.5 * amax * tr * tr
    local vp, tf = vmax, 0.0
    if 2*dr >= length then tr = math.sqrt(length/amax); vp = amax*tr else tf = (length - 2*dr)/vmax end
    local natural = 2*tr + tf
    local scale = 1.0
    if natural > duration then scale = natural / duration end
    local a, v = amax*scale*scale, vp*scale
    tr, tf = tr/scale, tf/scale
    local d, vel = 0.0, 0.0
    if t < tr then d, vel = 0.5*a*t*t, a*t
    elseif t < tr+tf then d, vel = 0.5*a*tr*tr + v*(t-tr), v
    elseif t < 2*tr+tf then
        local td = t-tr-tf
        d, vel = 0.5*a*tr*tr + v*tf + v*td - 0.5*a*td*td, math.max(0, v-a*td)
    else d, vel = length, 0.0 end
    return math.min(d / math.max(length, 1e-9), 1.0), vel, v, a
end

local function createCamera()
    local sensor = sim.createVisionSensor(1|2|4|128, {640,360,0,0}, {0.02,7.0,math.rad(62),0.1,0,0,0.78,0.82,0.86,0,0})
    sim.setObjectAlias(sensor, 'FixedZAccelerationCamera')
    return sim.getObject('/FixedZAccelerationCamera')
end

local function cameraMatrix(i, n, target)
    local u = n > 1 and i/(n-1) or 0
    local yaw, radius = math.rad(-48 + 18*u), 2.05
    local cam = {target[1] + radius*math.cos(yaw), target[2] + radius*math.sin(yaw), target[3] + 0.42}
    local f = normalize({target[1]-cam[1], target[2]-cam[2], target[3]-cam[3]})
    local right = normalize(cross(f, {0,0,1}))
    local up = cross(right, f)
    return {right[1],up[1],f[1],cam[1], right[2],up[2],f[2],cam[2], right[3],up[3],f[3],cam[3]}
end

local function jarr(v)
    local out = {}
    for i, x in ipairs(v) do out[i] = string.format('%.9g', x) end
    return '[' .. table.concat(out, ',') .. ']'
end

local function j3(v)
    return string.format('[%.9g,%.9g,%.9g]', v[1], v[2], v[3])
end

local function writeSummary(peakPlannedV, plannedA)
    local first, last = {traces.x[1], traces.y[1], traces.z[1]}, {traces.x[#traces.x], traces.y[#traces.y], traces.z[#traces.z]}
    local xmin, xmax = traces.x[1], traces.x[1]
    local maxy, maxz, maxori, peakvx, peakspeed, peakqd = 0, 0, 0, 0, 0, 0
    for i = 1, #traces.x do
        xmin, xmax = math.min(xmin, traces.x[i]), math.max(xmax, traces.x[i])
        maxy = math.max(maxy, math.abs(traces.y[i] - first[2]))
        maxz = math.max(maxz, math.abs(traces.z[i] - first[3]))
        maxori = math.max(maxori, traces.ori[i])
        peakvx = math.max(peakvx, math.abs(traces.vx[i]))
        peakspeed = math.max(peakspeed, traces.speed[i])
        peakqd = math.max(peakqd, traces.qdmax[i])
    end
    local xNet = last[1] - first[1]
    local xErr = xNet - TARGET_DX
    local xTol = math.max(0.005, 0.10 * math.abs(TARGET_DX))
    local baseOnGround = math.abs(MODEL_BASE_Z_OFFSET) <= 1e-9
    local xOk = math.abs(xErr) <= xTol
    local yOk = maxy <= 0.005
    local zOk = maxz <= 0.001
    local oriOk = maxori <= math.rad(3.0)
    local success = ikSuccess and baseOnGround and xOk and yOk and zOk and oriOk
    local reasons = {}
    if not ikSuccess then reasons[#reasons+1] = '"ik_failed"' end
    if not baseOnGround then reasons[#reasons+1] = '"base_not_on_ground"' end
    if not xOk then reasons[#reasons+1] = '"x_tracking_error"' end
    if not yOk then reasons[#reasons+1] = '"y_drift_too_large"' end
    if not zOk then reasons[#reasons+1] = '"z_drift_too_large"' end
    if not oriOk then reasons[#reasons+1] = '"orientation_error_too_large"' end
    local lines = {
        '{',
        '  "controller_name": "coppeliasim_fixed_z_acceleration_x_transport_controller",',
        '  "controller_family": "mujoco_acceleration_x_transport_controller_port_position_servo_ik",',
        string.format('  "task_frame_mode": "%s",', TASK_FRAME_MODE),
        string.format('  "task_frame_dummy_active": %s,', tostring(taskFrameDummyActive)),
        string.format('  "task_frame_handle": %d,', taskFrameHandle),
        string.format('  "task_frame_resolved_path": "%s",', taskFrameResolvedPath),
        string.format('  "uses_position_servo_setpoints": true,'),
        string.format('  "uses_direct_torque_control": false,'),
        string.format('  "ik_success": %s,', tostring(ikSuccess)),
        string.format('  "ik_waypoints": %d,', IK_WAYPOINTS),
        string.format('  "ik_max_waypoint_position_error_m": %.9g,', ikMaxPosErr),
        string.format('  "ik_max_waypoint_orientation_error_rad": %.9g,', ikMaxRotErr),
        string.format('  "target_dx_m": %.9g,', TARGET_DX),
        string.format('  "duration_s": %.9g,', DURATION_S),
        string.format('  "fps": %d,', FPS),
        string.format('  "frames": %d,', FRAME_COUNT),
        string.format('  "a_x_max_m_s2": %.9g,', A_X_MAX),
        string.format('  "v_x_max_m_s": %.9g,', V_X_MAX),
        string.format('  "model_base_z_offset_m": %.9g,', MODEL_BASE_Z_OFFSET),
        string.format('  "planned_peak_speed_mps": %.9g,', peakPlannedV),
        string.format('  "planned_accel_mps2": %.9g,', plannedA),
        string.format('  "path_length_m": %.9g,', pathLength),
        string.format('  "avg_path_speed_mps": %.9g,', pathLength / math.max(DURATION_S, 1e-9)),
        string.format('  "peak_path_speed_mps": %.9g,', peakspeed),
        string.format('  "peak_abs_ee_vx_mps": %.9g,', peakvx),
        string.format('  "x_span_m": %.9g,', xmax-xmin),
        string.format('  "x_net_displacement_m": %.9g,', xNet),
        string.format('  "x_tracking_error_m": %.9g,', xErr),
        string.format('  "max_abs_y_drift_m": %.9g,', maxy),
        string.format('  "max_abs_z_drift_m": %.9g,', maxz),
        string.format('  "max_orientation_error_rad": %.9g,', maxori),
        string.format('  "max_orientation_error_deg": %.9g,', maxori*180/math.pi),
        string.format('  "peak_joint_speed_rad_s": %.9g,', peakqd),
        string.format('  "base_on_ground": %s,', tostring(baseOnGround)),
        string.format('  "x_tracking_ok": %s,', tostring(xOk)),
        string.format('  "single_axis_y_ok": %s,', tostring(yOk)),
        string.format('  "fixed_z_ok": %s,', tostring(zOk)),
        string.format('  "orientation_ok": %s,', tostring(oriOk)),
        string.format('  "success": %s,', tostring(success)),
        '  "failure_reasons": [' .. table.concat(reasons, ',') .. '],',
        '  "q_start_source": "' .. Q_START_SOURCE .. '",',
        '  "q_start": ' .. jarr(Q_START) .. ',',
        '  "ee_start_world_m": ' .. j3(first) .. ',',
        '  "ee_final_world_m": ' .. j3(last) .. ',',
        '  "ee_start_world_matrix": ' .. jarr(captureStartMatrix or {}) .. ',',
        '  "ee_final_world_matrix": ' .. jarr(captureEndMatrix or {}) .. ',',
        '  "time_s_trace": ' .. jarr(traces.t) .. ',',
        '  "ee_x_trace": ' .. jarr(traces.x) .. ',',
        '  "ee_y_trace": ' .. jarr(traces.y) .. ',',
        '  "ee_z_trace": ' .. jarr(traces.z) .. ',',
        '  "ee_vx_trace_mps": ' .. jarr(traces.vx) .. ',',
        '  "path_speed_trace_mps": ' .. jarr(traces.speed) .. ',',
        '  "orientation_error_rad_trace": ' .. jarr(traces.ori) .. ',',
        '  "video_path": "' .. VIDEO_PATH .. '"',
        '}',
        '',
    }
    writeText(SUMMARY_PATH, table.concat(lines, '\n'))
end

local function capture()
    buildPath()
    visionSensor = createCamera()
    writeText(SENSING_MARKER, 'capture\n')
    local startP, endP = path[1].p, path[#path].p
    local center = {0.5*(startP[1]+endP[1]), startP[2], startP[3] + 0.02}
    local prevP, prevQ, prevT = nil, nil, nil
    local peakV, accUsed = 0, 0
    for i = 0, FRAME_COUNT-1 do
        local t = DURATION_S * i / math.max(FRAME_COUNT-1, 1)
        local u, _, pv, aa = profile(t, DURATION_S, pathLength)
        peakV, accUsed = math.max(peakV, pv), math.max(accUsed, aa)
        local q = qAt(u)
        setQ(q)
        local p, r = getPose()
        local spd, vx, qdmax = 0, 0, 0
        if prevP then
            local dt = math.max(t-prevT, 1e-9)
            spd = dist(p, prevP) / dt
            vx = (p[1] - prevP[1]) / dt
            for j = 1, 6 do qdmax = math.max(qdmax, math.abs((q[j]-prevQ[j]) / dt)) end
        end
        traces.t[#traces.t+1], traces.x[#traces.x+1], traces.y[#traces.y+1], traces.z[#traces.z+1] = t, p[1], p[2], p[3]
        traces.vx[#traces.vx+1], traces.speed[#traces.speed+1], traces.progress[#traces.progress+1] = vx, spd, u
        traces.ori[#traces.ori+1], traces.qdmax[#traces.qdmax+1] = rotAngle(r, TARGET_ATTACHMENT_ROTATION_WORLD), qdmax
        if i == 0 then captureStartMatrix = r end
        if i == FRAME_COUNT - 1 then captureEndMatrix = r end
        sim.setObjectMatrix(visionSensor, cameraMatrix(i, FRAME_COUNT, center), sim.handle_world)
        sim.handleVisionSensor(visionSensor)
        local img, res = sim.getVisionSensorImg(visionSensor)
        img = sim.transformImage(img, res, 4)
        sim.saveImage(img, res, 0, string.format('%s/%s_%08d.png', OUTPUT_DIR, FRAME_PREFIX, i), -1)
        prevP, prevQ, prevT = p, q, t
    end
    writeSummary(peakV, accUsed)
    writeText(DONE_MARKER, 'done\n')
    sim.quitSimulator()
end

function sysCall_init()
    writeText(LOAD_MARKER, 'loaded\n')
    writeText(START_MARKER, 'init\n')
    sim.addLog(sim.verbosity_scriptinfos, 'UR5 fixed-Z acceleration-X transport add-on starting')
    sim.loadScene(SCENE_PATH)
    sim.setColorProperty(sim.handle_scene, 'ambientLight', {0.65, 0.65, 0.65}, {noError = true})
    local modelHandle = sim.loadModel(MODEL_PATH)
    if modelHandle and modelHandle >= 0 and MODEL_BASE_Z_OFFSET ~= 0.0 then
        sim.setObjectPosition(modelHandle, sim.handle_world, {0.0, 0.0, MODEL_BASE_Z_OFFSET})
    end
    resolveHandles()
    configureTaskFrame()
    setQ(Q_START)
    capture()
end

function sysCall_nonSimulation()
end

function sysCall_cleanup()
    sim.addLog(sim.verbosity_scriptinfos, 'UR5 fixed-Z acceleration-X transport cleanup')
end
