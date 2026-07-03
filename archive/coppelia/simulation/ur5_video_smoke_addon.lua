local COPPELIA_ROOT = os.getenv('COPPELIA_ROOT') or '/common/users/ss5772/real_Cartpole/third_party/coppelia_runtime/CoppeliaSim_Edu_V4_10_0_rev0_Ubuntu24_04'
local SCENE_PATH = COPPELIA_ROOT .. '/system/dfltscn.ttt'
local MODEL_PATH = COPPELIA_ROOT .. '/models/robots/non-mobile/UR5.ttm'
local OUTPUT_DIR = '/common/users/ss5772/real_Cartpole/outputs/control_runs/coppelia_video_smoke_frames'
local STATE_DIR = '/common/users/ss5772/real_Cartpole/outputs/control_runs/coppelia_video_smoke_state'
local FRAME_PREFIX = 'frame'
local LOAD_MARKER = STATE_DIR .. '/ur5_video_smoke_addon_loaded.txt'
local START_MARKER = STATE_DIR .. '/ur5_video_smoke_addon_started.txt'
local SENSING_MARKER = STATE_DIR .. '/ur5_video_smoke_addon_sensing.txt'
local DONE_MARKER = STATE_DIR .. '/ur5_video_smoke_startup_done.txt'
local RPC_READY_MARKER = STATE_DIR .. '/rpc_bootstrap_ready.txt'
local FRAME_COUNT = 40
local ENABLE_SMOKE = os.getenv('REAL_CARTPOLE_ENABLE_VIDEO_SMOKE') == '1'

local JOINT_PATHS = {
    '/UR5/joint',
    '/UR5/link/joint',
    '/UR5/link/link/joint',
    '/UR5/link/link/link/joint',
    '/UR5/link/link/link/link/joint',
    '/UR5/link/link/link/link/link/joint',
}

local JOINT_TARGETS = {0.2, -1.15, 1.55, -1.8, -1.45, 0.35}

local captureCount = 0
local captureStarted = false
local visionSensor = -1

local function writeMarker(path, text)
    local f = io.open(path, 'w')
    if not f then
        return
    end
    f:write(text or '')
    f:close()
end

local function fileExists(path)
    local f = io.open(path, 'r')
    if f then
        f:close()
        return true
    end
    return false
end

if ENABLE_SMOKE then
    writeMarker(LOAD_MARKER, 'loaded\n')
end

sim = require 'sim'

function sysCall_info()
    return {autoStart = true}
end

local function setJointPose()
    for i, path in ipairs(JOINT_PATHS) do
        local handle = sim.getObject(path)
        sim.setJointPosition(handle, JOINT_TARGETS[i])
    end
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
    local yaw = math.rad(-48.0 + 18.0 * progress)
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
    sim.setObjectAlias(sensor, 'SmokeVideoCamera')
    return sim.getObject('/SmokeVideoCamera')
end

local function captureFrames()
    if captureStarted then
        return
    end
    captureStarted = true
    if visionSensor < 0 then
        visionSensor = createCamera()
        sim.addLog(sim.verbosity_scriptinfos, 'Vision sensor handle: ' .. tostring(visionSensor))
    end
    if visionSensor < 1 then
        sim.addLog(sim.verbosity_errors, 'Vision sensor creation failed')
        writeMarker(DONE_MARKER, 'failed\n')
        sim.quitSimulator()
        return
    end
    writeMarker(SENSING_MARKER, 'capture\n')
    for i = 0, FRAME_COUNT - 1 do
        local matrix = cameraMatrix(i, FRAME_COUNT)
        sim.setObjectMatrix(visionSensor, matrix, sim.handle_world)
        sim.handleVisionSensor(visionSensor)
        local img, res = sim.getVisionSensorImg(visionSensor)
        img = sim.transformImage(img, res, 4)
        local fileName = string.format('%s/%s_%08d.png', OUTPUT_DIR, FRAME_PREFIX, i)
        sim.saveImage(img, res, 0, fileName, -1)
        captureCount = i + 1
    end
    writeMarker(DONE_MARKER, 'done\n')
    sim.quitSimulator()
end

function sysCall_init()
    if not ENABLE_SMOKE then
        sim.addLog(sim.verbosity_scriptinfos, 'UR5 video smoke add-on RPC bootstrap mode')
        sim.loadModel(MODEL_PATH)
        setJointPose()
        sim.addLog(sim.verbosity_scriptinfos, 'RPC bootstrap waiting for ' .. RPC_READY_MARKER)
        local deadline = sim.getSystemTime() + 120.0
        while not fileExists(RPC_READY_MARKER) and sim.getSystemTime() < deadline do
            sim.wait(0.1)
        end
        sim.addLog(sim.verbosity_scriptinfos, 'RPC bootstrap release observed; waiting grace period')
        for _ = 1, 50 do
            sim.wait(0.1)
        end
        sim.setStepping(true)
        sim.startSimulation()
        sim.addLog(sim.verbosity_scriptinfos, 'RPC bootstrap stepped simulation started')
        sim.addLog(sim.verbosity_scriptinfos, 'Loaded bootstrap UR5: ' .. MODEL_PATH)
        return
    end
    sim.addLog(sim.verbosity_scriptinfos, 'UR5 video smoke add-on starting')
    writeMarker(START_MARKER, 'init\n')
    sim.loadScene(SCENE_PATH)
    sim.addLog(sim.verbosity_scriptinfos, 'Loaded default scene: ' .. SCENE_PATH)
    sim.setColorProperty(sim.handle_scene, 'ambientLight', {0.65, 0.65, 0.65}, {noError = true})
    sim.loadModel(MODEL_PATH)
    setJointPose()
    visionSensor = -1
    captureCount = 0
    captureStarted = false
    sim.addLog(sim.verbosity_scriptinfos, 'Capturing frames to ' .. OUTPUT_DIR)
    captureFrames()
end

function sysCall_nonSimulation()
    if not ENABLE_SMOKE then
        return
    end
    captureFrames()
end

function sysCall_cleanup()
    if not ENABLE_SMOKE then
        return
    end
    sim.addLog(sim.verbosity_scriptinfos, string.format('Captured %d frames', captureCount))
end
