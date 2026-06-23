local COPPELIA_ROOT = os.getenv('COPPELIA_ROOT') or '/common/users/ss5772/real_Cartpole/third_party/coppelia_runtime/CoppeliaSim_Edu_V4_10_0_rev0_Ubuntu24_04'
local ROOT = os.getenv('REAL_CARTPOLE_ROOT') or '/common/users/ss5772/real_Cartpole'
local SCENE_PATH = COPPELIA_ROOT .. '/system/dfltscn.ttt'
local MODEL_PATH = COPPELIA_ROOT .. '/models/robots/non-mobile/UR5.ttm'
local OUTPUT_DIR = ROOT .. '/outputs/control_runs/coppelia_controller_video_frames'
local STATE_DIR = ROOT .. '/outputs/control_runs/coppelia_controller_video_state'
local FRAME_PREFIX = 'frame'
local LOAD_MARKER = STATE_DIR .. '/ur5_controller_video_addon_loaded.txt'
local START_MARKER = STATE_DIR .. '/ur5_controller_video_addon_started.txt'
local SENSING_MARKER = STATE_DIR .. '/ur5_controller_video_addon_sensing.txt'
local DONE_MARKER = STATE_DIR .. '/ur5_controller_video_done.txt'
local SUMMARY_PATH = STATE_DIR .. '/ur5_controller_video_summary.txt'
local FRAME_COUNT = tonumber(os.getenv('FRAME_COUNT') or '80')

local JOINT_PATHS = {
    '/UR5/joint',
    '/UR5/link/joint',
    '/UR5/link/link/joint',
    '/UR5/link/link/link/joint',
    '/UR5/link/link/link/link/joint',
    '/UR5/link/link/link/link/link/joint',
}

local BASE_Q = {0.2, -1.15, 1.55, -1.8, -1.45, 0.35}
local AMP_Q = {0.42, 0.24, -0.28, 0.32, 0.36, -0.22}
local PHASE_Q = {0.0, 0.8, 1.7, 2.4, 3.2, 4.0}

local captureCount = 0
local captureStarted = false
local visionSensor = -1
local joints = {}
local qMin = {}
local qMax = {}

local sim = require 'sim'

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

local function cameraMatrix(stepIdx, totalSteps)
    local progress = 0.0
    if totalSteps > 1 then
        progress = stepIdx / (totalSteps - 1)
    end
    local yaw = math.rad(-52.0 + 24.0 * progress)
    local radius = 1.95
    local target = {0.0, 0.0, 0.62}
    local camPos = {
        radius * math.cos(yaw),
        radius * math.sin(yaw),
        target[3] + 0.34,
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

local function createCamera()
    local sensor = sim.createVisionSensor(
        1 | 2 | 4 | 128,
        {640, 360, 0, 0},
        {0.02, 6.0, math.rad(60.0), 0.1, 0.0, 0.0, 0.78, 0.82, 0.86, 0.0, 0.0}
    )
    sim.setObjectAlias(sensor, 'ControllerVideoCamera')
    return sim.getObject('/ControllerVideoCamera')
end

local function resolveJoints()
    joints = {}
    for i, path in ipairs(JOINT_PATHS) do
        local handle = sim.getObject(path)
        joints[i] = handle
        qMin[i] = BASE_Q[i]
        qMax[i] = BASE_Q[i]
    end
end

local function commandedJointPose(stepIdx, totalSteps)
    local progress = 0.0
    if totalSteps > 1 then
        progress = stepIdx / (totalSteps - 1)
    end
    local envelope = math.sin(math.pi * progress)
    local q = {}
    for i = 1, #BASE_Q do
        local wave = math.sin((2.0 * math.pi * progress) + PHASE_Q[i])
        q[i] = BASE_Q[i] + AMP_Q[i] * envelope * wave
    end
    return q
end

local function applyJointSpaceController(q)
    for i, handle in ipairs(joints) do
        sim.setJointPosition(handle, q[i])
        if q[i] < qMin[i] then qMin[i] = q[i] end
        if q[i] > qMax[i] then qMax[i] = q[i] end
    end
end

local function captureFrames()
    if captureStarted then
        return
    end
    captureStarted = true
    if visionSensor < 0 then
        visionSensor = createCamera()
        sim.addLog(sim.verbosity_scriptinfos, 'Controller video sensor handle: ' .. tostring(visionSensor))
    end
    if visionSensor < 1 then
        sim.addLog(sim.verbosity_errors, 'Controller video sensor creation failed')
        writeText(DONE_MARKER, 'failed\n')
        sim.quitSimulator()
        return
    end

    writeText(SENSING_MARKER, 'capture\n')
    for i = 0, FRAME_COUNT - 1 do
        local q = commandedJointPose(i, FRAME_COUNT)
        applyJointSpaceController(q)
        sim.setObjectMatrix(visionSensor, cameraMatrix(i, FRAME_COUNT), sim.handle_world)
        sim.handleVisionSensor(visionSensor)
        local img, res = sim.getVisionSensorImg(visionSensor)
        img = sim.transformImage(img, res, 4)
        local fileName = string.format('%s/%s_%08d.png', OUTPUT_DIR, FRAME_PREFIX, i)
        sim.saveImage(img, res, 0, fileName, -1)
        captureCount = i + 1
    end

    local summary = string.format(
        'mode=lua_joint_space_controller\nframes=%d\nq0_min=%.6f\nq0_max=%.6f\nq1_min=%.6f\nq1_max=%.6f\nq2_min=%.6f\nq2_max=%.6f\n',
        captureCount,
        qMin[1], qMax[1],
        qMin[2], qMax[2],
        qMin[3], qMax[3]
    )
    writeText(SUMMARY_PATH, summary)
    writeText(DONE_MARKER, 'done\n')
    sim.quitSimulator()
end

function sysCall_init()
    writeText(LOAD_MARKER, 'loaded\n')
    sim.addLog(sim.verbosity_scriptinfos, 'UR5 controller video add-on starting')
    writeText(START_MARKER, 'init\n')
    sim.loadScene(SCENE_PATH)
    sim.addLog(sim.verbosity_scriptinfos, 'Loaded default scene: ' .. SCENE_PATH)
    sim.setColorProperty(sim.handle_scene, 'ambientLight', {0.65, 0.65, 0.65}, {noError = true})
    sim.loadModel(MODEL_PATH)
    resolveJoints()
    applyJointSpaceController(BASE_Q)
    visionSensor = -1
    captureCount = 0
    captureStarted = false
    sim.addLog(sim.verbosity_scriptinfos, 'Capturing controller video frames to ' .. OUTPUT_DIR)
    captureFrames()
end

function sysCall_nonSimulation()
    captureFrames()
end

function sysCall_cleanup()
    sim.addLog(sim.verbosity_scriptinfos, string.format('Controller video captured %d frames', captureCount))
end
