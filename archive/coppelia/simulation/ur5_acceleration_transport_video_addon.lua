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
local V_X_MAX = tonumber(os.getenv('V_X_MAX_MPS') or '0.55')
local A_X_MAX = tonumber(os.getenv('A_X_MAX_MPS2') or '2.0')

local JOINT_PATHS = {
    '/UR5/joint',
    '/UR5/link/joint',
    '/UR5/link/link/joint',
    '/UR5/link/link/link/joint',
    '/UR5/link/link/link/link/joint',
    '/UR5/link/link/link/link/link/joint',
}

-- Coppelia world-X transport endpoints. The official Coppelia UR5 model's
-- world axes differ from the MuJoCo UR5e setup: the MuJoCo fixed-Z endpoints
-- sweep mostly Coppelia Y/Z. This pair uses shoulder-pan transport so the
-- visible end effector actually crosses world X quickly.
local Q_START = {
    -1.10,
    -1.15,
    1.55,
    -1.80,
    -1.45,
    0.35,
}

local Q_STOP = {
    1.10,
    -1.15,
    1.55,
    -1.80,
    -1.45,
    0.35,
}

local sim = require 'sim'

local joints = {}
local eeHandle = -1
local visionSensor = -1
local captureStarted = false
local captureCount = 0

local timeTrace = {}
local xTrace = {}
local yTrace = {}
local zTrace = {}
local vxTrace = {}
local speedTrace = {}
local progressTrace = {}
local qTrace = {}
local jointSpeedTrace = {}

function sysCall_info()
    return {autoStart = true}
end

local function writeText(path, text)
    local f = io.open(path, 'w')
    if not f then
        return
    end
    f:write(text or '')
    f:close()
end

local function normalize(v)
    local norm = math.sqrt(v[1] * v[1] + v[2] * v[2] + v[3] * v[3])
    if norm < 1e-9 then
        return {0.0, 0.0, 0.0}
    end
    return {v[1] / norm, v[2] / norm, v[3] / norm}
end

local function cross(a, b)
    return {
        a[2] * b[3] - a[3] * b[2],
        a[3] * b[1] - a[1] * b[3],
        a[1] * b[2] - a[2] * b[1],
    }
end

local function createCamera()
    local sensor = sim.createVisionSensor(
        1 | 2 | 4 | 128,
        {640, 360, 0, 0},
        {0.02, 7.0, math.rad(62.0), 0.1, 0.0, 0.0, 0.78, 0.82, 0.86, 0.0, 0.0}
    )
    sim.setObjectAlias(sensor, 'AccelerationTransportCamera')
    return sim.getObject('/AccelerationTransportCamera')
end

local function cameraMatrix(stepIdx, totalSteps, target)
    local progress = 0.0
    if totalSteps > 1 then
        progress = stepIdx / (totalSteps - 1)
    end
    local yaw = math.rad(-58.0 + 32.0 * progress)
    local radius = 2.35
    local camPos = {
        target[1] + radius * math.cos(yaw),
        target[2] + radius * math.sin(yaw),
        target[3] + 0.54,
    }
    local forward = normalize({target[1] - camPos[1], target[2] - camPos[2], target[3] - camPos[3]})
    local worldUp = {0.0, 0.0, 1.0}
    local right = normalize(cross(forward, worldUp))
    local up = cross(right, forward)
    return {
        right[1], up[1], forward[1], camPos[1],
        right[2], up[2], forward[2], camPos[2],
        right[3], up[3], forward[3], camPos[3],
    }
end

local function resolveHandles()
    joints = {}
    for i, path in ipairs(JOINT_PATHS) do
        joints[i] = sim.getObject(path)
    end
    local candidates = {
        '/UR5/UR5_connection',
        ':/UR5/UR5_connection',
        '/UR5_connection',
        ':/UR5_connection',
    }
    for _, path in ipairs(candidates) do
        local ok, handle = pcall(sim.getObject, path)
        if ok then
            eeHandle = handle
            return
        end
    end
    eeHandle = joints[#joints]
end

local function setJoints(q)
    for i, handle in ipairs(joints) do
        sim.setJointPosition(handle, q[i])
    end
end

local function interpQ(s)
    local q = {}
    for i = 1, #Q_START do
        q[i] = Q_START[i] + s * (Q_STOP[i] - Q_START[i])
    end
    return q
end

local function eePos()
    local pose = sim.getObjectPose(eeHandle, sim.handle_world)
    return {pose[1], pose[2], pose[3]}
end

local function dist(a, b)
    local dx = a[1] - b[1]
    local dy = a[2] - b[2]
    local dz = a[3] - b[3]
    return math.sqrt(dx * dx + dy * dy + dz * dz)
end

local function jointDistance(q0, q1)
    local total = 0.0
    for i = 1, #q0 do
        local d = q1[i] - q0[i]
        total = total + d * d
    end
    return math.sqrt(total)
end

local function computePathSamples(n)
    local path = {}
    local cumulative = {0.0}
    local total = 0.0
    for i = 0, n - 1 do
        local s = i / (n - 1)
        local q = interpQ(s)
        setJoints(q)
        local p = eePos()
        path[i + 1] = {s = s, q = q, p = p}
        if i > 0 then
            total = total + dist(path[i].p, p)
            cumulative[i + 1] = total
        end
    end
    if total < 1e-9 then
        total = math.abs(path[#path].p[1] - path[1].p[1])
    end
    for i = 1, #path do
        path[i].u = cumulative[i] / total
    end
    return path, total
end

local function progressForDistanceFraction(path, u)
    if u <= 0.0 then
        return 0.0
    end
    if u >= 1.0 then
        return 1.0
    end
    for i = 2, #path do
        if path[i].u >= u then
            local u0 = path[i - 1].u
            local u1 = path[i].u
            local alpha = 0.0
            if u1 > u0 + 1e-12 then
                alpha = (u - u0) / (u1 - u0)
            end
            return path[i - 1].s + alpha * (path[i].s - path[i - 1].s)
        end
    end
    return 1.0
end

local function trapezoidU(t, duration, pathLength)
    local vMax = math.max(V_X_MAX, 1e-6)
    local aMax = math.max(A_X_MAX, 1e-6)
    local tRamp = vMax / aMax
    local dRamp = 0.5 * aMax * tRamp * tRamp
    local vPeak = vMax
    local tFlat = 0.0
    if 2.0 * dRamp >= pathLength then
        tRamp = math.sqrt(pathLength / aMax)
        vPeak = aMax * tRamp
    else
        tFlat = (pathLength - 2.0 * dRamp) / vMax
    end
    local naturalDuration = 2.0 * tRamp + tFlat
    local scale = 1.0
    if naturalDuration > duration then
        scale = naturalDuration / duration
    end
    local a = aMax * scale * scale
    local v = vPeak * scale
    local tr = tRamp / scale
    local tf = tFlat / scale
    local d = 0.0
    local vel = 0.0
    local acc = 0.0
    if t <= 0.0 then
        d = 0.0
        vel = 0.0
        acc = a
    elseif t < tr then
        d = 0.5 * a * t * t
        vel = a * t
        acc = a
    elseif t < tr + tf then
        d = 0.5 * a * tr * tr + v * (t - tr)
        vel = v
        acc = 0.0
    elseif t < 2.0 * tr + tf then
        local td = t - tr - tf
        d = 0.5 * a * tr * tr + v * tf + v * td - 0.5 * a * td * td
        vel = math.max(0.0, v - a * td)
        acc = -a
    else
        d = pathLength
        vel = 0.0
        acc = 0.0
    end
    if d > pathLength then d = pathLength end
    return d / pathLength, vel, acc, v, a
end

local function jsonArray(values)
    local out = {}
    for i, v in ipairs(values) do
        out[i] = string.format('%.9g', v)
    end
    return '[' .. table.concat(out, ',') .. ']'
end

local function jsonArray3(values)
    return string.format('[%.9g,%.9g,%.9g]', values[1], values[2], values[3])
end

local function writeSummary(pathLength, plannedPeakV, plannedAccel)
    local first = {xTrace[1], yTrace[1], zTrace[1]}
    local last = {xTrace[#xTrace], yTrace[#yTrace], zTrace[#zTrace]}
    local xMin = xTrace[1]
    local xMax = xTrace[1]
    local y0 = yTrace[1]
    local z0 = zTrace[1]
    local maxYDrift = 0.0
    local maxZDrift = 0.0
    local peakSpeed = 0.0
    local peakVx = 0.0
    local peakJointSpeed = 0.0
    local peakJointSpeedPerJoint = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0}
    for i = 1, #xTrace do
        if xTrace[i] < xMin then xMin = xTrace[i] end
        if xTrace[i] > xMax then xMax = xTrace[i] end
        local yd = math.abs(yTrace[i] - y0)
        local zd = math.abs(zTrace[i] - z0)
        if yd > maxYDrift then maxYDrift = yd end
        if zd > maxZDrift then maxZDrift = zd end
        if speedTrace[i] and speedTrace[i] > peakSpeed then peakSpeed = speedTrace[i] end
        if vxTrace[i] and math.abs(vxTrace[i]) > peakVx then peakVx = math.abs(vxTrace[i]) end
        if jointSpeedTrace[i] then
            for j = 1, 6 do
                local val = math.abs(jointSpeedTrace[i][j])
                if val > peakJointSpeedPerJoint[j] then peakJointSpeedPerJoint[j] = val end
                if val > peakJointSpeed then peakJointSpeed = val end
            end
        end
    end
    local avgSpeed = pathLength / math.max(DURATION_S, 1e-9)
    local lines = {
        '{',
        '  "controller_name": "coppeliasim_acceleration_transport_controller",',
        '  "controller_family": "acceleration_x_transport_controller_style",',
        string.format('  "duration_s": %.9g,', DURATION_S),
        string.format('  "fps": %d,', FPS),
        string.format('  "frames": %d,', captureCount),
        string.format('  "a_x_max_m_s2": %.9g,', A_X_MAX),
        string.format('  "v_x_max_m_s": %.9g,', V_X_MAX),
        string.format('  "planned_peak_speed_mps": %.9g,', plannedPeakV),
        string.format('  "planned_accel_mps2": %.9g,', plannedAccel),
        string.format('  "path_length_m": %.9g,', pathLength),
        string.format('  "avg_path_speed_mps": %.9g,', avgSpeed),
        string.format('  "peak_path_speed_mps": %.9g,', peakSpeed),
        string.format('  "peak_abs_ee_vx_mps": %.9g,', peakVx),
        string.format('  "x_span_m": %.9g,', xMax - xMin),
        string.format('  "x_net_displacement_m": %.9g,', last[1] - first[1]),
        string.format('  "max_abs_y_drift_m": %.9g,', maxYDrift),
        string.format('  "max_abs_z_drift_m": %.9g,', maxZDrift),
        string.format('  "peak_joint_speed_rad_s": %.9g,', peakJointSpeed),
        '  "peak_joint_speed_per_joint_rad_s": ' .. jsonArray(peakJointSpeedPerJoint) .. ',',
        '  "q_start": ' .. jsonArray(Q_START) .. ',',
        '  "q_stop": ' .. jsonArray(Q_STOP) .. ',',
        '  "ee_start_world_m": ' .. jsonArray3(first) .. ',',
        '  "ee_final_world_m": ' .. jsonArray3(last) .. ',',
        '  "time_s_trace": ' .. jsonArray(timeTrace) .. ',',
        '  "ee_x_trace": ' .. jsonArray(xTrace) .. ',',
        '  "ee_y_trace": ' .. jsonArray(yTrace) .. ',',
        '  "ee_z_trace": ' .. jsonArray(zTrace) .. ',',
        '  "ee_vx_trace_mps": ' .. jsonArray(vxTrace) .. ',',
        '  "path_speed_trace_mps": ' .. jsonArray(speedTrace) .. ',',
        '  "progress_trace": ' .. jsonArray(progressTrace) .. ',',
        '  "video_path": "' .. VIDEO_PATH .. '"',
        '}',
        '',
    }
    writeText(SUMMARY_PATH, table.concat(lines, '\n'))
end

local function captureFrames()
    if captureStarted then
        return
    end
    captureStarted = true
    if visionSensor < 0 then
        visionSensor = createCamera()
        sim.addLog(sim.verbosity_scriptinfos, 'Acceleration transport camera handle: ' .. tostring(visionSensor))
    end
    if visionSensor < 1 then
        sim.addLog(sim.verbosity_errors, 'Acceleration transport camera creation failed')
        writeText(DONE_MARKER, 'failed\n')
        sim.quitSimulator()
        return
    end

    local path, pathLength = computePathSamples(240)
    local center = {
        0.5 * (path[1].p[1] + path[#path].p[1]),
        0.5 * (path[1].p[2] + path[#path].p[2]),
        0.5 * (path[1].p[3] + path[#path].p[3]) + 0.05,
    }

    writeText(SENSING_MARKER, 'capture\n')
    local prevP = nil
    local prevQ = nil
    local prevTime = nil
    local plannedPeakV = 0.0
    local plannedAccel = 0.0
    for i = 0, FRAME_COUNT - 1 do
        local t = 0.0
        if FRAME_COUNT > 1 then
            t = DURATION_S * i / (FRAME_COUNT - 1)
        end
        local u, plannedV, plannedA, peakV, accelUsed = trapezoidU(t, DURATION_S, pathLength)
        if peakV > plannedPeakV then plannedPeakV = peakV end
        if accelUsed > plannedAccel then plannedAccel = accelUsed end
        local s = progressForDistanceFraction(path, u)
        local q = interpQ(s)
        setJoints(q)
        local p = eePos()

        local speed = 0.0
        local vx = 0.0
        local js = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0}
        if prevP ~= nil and prevTime ~= nil then
            local dt = math.max(t - prevTime, 1e-9)
            speed = dist(p, prevP) / dt
            vx = (p[1] - prevP[1]) / dt
            for j = 1, #q do
                js[j] = (q[j] - prevQ[j]) / dt
            end
        end

        table.insert(timeTrace, t)
        table.insert(xTrace, p[1])
        table.insert(yTrace, p[2])
        table.insert(zTrace, p[3])
        table.insert(vxTrace, vx)
        table.insert(speedTrace, speed)
        table.insert(progressTrace, s)
        table.insert(qTrace, q)
        table.insert(jointSpeedTrace, js)

        sim.setObjectMatrix(visionSensor, cameraMatrix(i, FRAME_COUNT, center), sim.handle_world)
        sim.handleVisionSensor(visionSensor)
        local img, res = sim.getVisionSensorImg(visionSensor)
        img = sim.transformImage(img, res, 4)
        local fileName = string.format('%s/%s_%08d.png', OUTPUT_DIR, FRAME_PREFIX, i)
        sim.saveImage(img, res, 0, fileName, -1)
        captureCount = i + 1

        prevP = p
        prevQ = q
        prevTime = t
    end

    writeSummary(pathLength, plannedPeakV, plannedAccel)
    writeText(DONE_MARKER, 'done\n')
    sim.quitSimulator()
end

function sysCall_init()
    writeText(LOAD_MARKER, 'loaded\n')
    sim.addLog(sim.verbosity_scriptinfos, 'UR5 acceleration transport video add-on starting')
    writeText(START_MARKER, 'init\n')
    sim.loadScene(SCENE_PATH)
    sim.setColorProperty(sim.handle_scene, 'ambientLight', {0.65, 0.65, 0.65}, {noError = true})
    sim.loadModel(MODEL_PATH)
    resolveHandles()
    setJoints(Q_START)
    visionSensor = -1
    captureStarted = false
    captureCount = 0
    sim.addLog(sim.verbosity_scriptinfos, 'Capturing acceleration transport frames to ' .. OUTPUT_DIR)
    captureFrames()
end

function sysCall_nonSimulation()
    captureFrames()
end

function sysCall_cleanup()
    sim.addLog(sim.verbosity_scriptinfos, string.format('Acceleration transport captured %d frames', captureCount))
end
