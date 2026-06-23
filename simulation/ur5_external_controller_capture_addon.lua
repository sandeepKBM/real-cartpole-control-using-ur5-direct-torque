local COPPELIA_ROOT = os.getenv('COPPELIA_ROOT') or '/common/users/ss5772/real_Cartpole/third_party/coppelia_runtime/CoppeliaSim_Edu_V4_10_0_rev0_Ubuntu24_04'
local ROOT = os.getenv('REAL_CARTPOLE_ROOT') or '/common/users/ss5772/real_Cartpole'
local OUTPUT_DIR = os.getenv('OUTPUT_DIR') or (ROOT .. '/outputs/control_runs/coppelia_external_controller_capture_frames')
local STATE_DIR = os.getenv('STATE_DIR') or (ROOT .. '/outputs/control_runs/coppelia_external_controller_capture_state')
local FRAME_PREFIX = os.getenv('FRAME_PREFIX') or 'frame'
local LOAD_MARKER = STATE_DIR .. '/ur5_external_controller_capture_loaded.txt'
local START_MARKER = STATE_DIR .. '/ur5_external_controller_capture_started.txt'
local SENSING_MARKER = STATE_DIR .. '/ur5_external_controller_capture_sensing.txt'
local DONE_MARKER = STATE_DIR .. '/ur5_external_controller_capture_done.txt'
local SUMMARY_PATH = os.getenv('SUMMARY_PATH') or (STATE_DIR .. '/coppelia_external_controller_capture_summary.txt')
local FRAME_COUNT = tonumber(os.getenv('FRAME_COUNT') or '80')
local CAPTURE_SKIP_FRAMES = tonumber(os.getenv('CAPTURE_SKIP_FRAMES') or '0')
local FPS = tonumber(os.getenv('FPS') or '20')

local captureCount = 0
local sensingCount = 0
local captureStarted = false
local captureFinished = false
local visionSensor = -1
local nextFrameTime = 0.0

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
    sim.setObjectAlias(sensor, 'ExternalControllerCaptureCamera')
    return sim.getObject('/ExternalControllerCaptureCamera')
end

local function tryResolveUr5()
    local ok, handle = pcall(sim.getObject, '/UR5')
    if ok then
        return handle
    end
    return -1
end

local function writeSummary(success, reason)
    writeText(
        SUMMARY_PATH,
        string.format(
            'success=%s\nreason=%s\nframes=%d\nframe_count_target=%d\ncapture_skip_frames=%d\nfps=%d\n',
            tostring(success),
            tostring(reason or ''),
            captureCount,
            FRAME_COUNT,
            CAPTURE_SKIP_FRAMES,
            FPS
        )
    )
end

local function captureFrame(simTime)
    local matrix = cameraMatrix(captureCount, FRAME_COUNT)
    sim.setObjectMatrix(visionSensor, matrix, sim.handle_world)
    sim.handleVisionSensor(visionSensor)
    local img, res = sim.getVisionSensorImg(visionSensor)
    img = sim.transformImage(img, res, 4)
    local fileName = string.format('%s/%s_%08d.png', OUTPUT_DIR, FRAME_PREFIX, captureCount)
    sim.saveImage(img, res, 0, fileName, -1)
    captureCount = captureCount + 1
    nextFrameTime = simTime + (1.0 / math.max(FPS, 1e-9))
end

local function ensureCaptureStarted(simTime)
    if captureStarted then
        return true
    end

    if tryResolveUr5() < 0 then
        return false
    end

    visionSensor = createCamera()
    sim.addLog(sim.verbosity_scriptinfos, 'External controller capture sensor handle: ' .. tostring(visionSensor))
    if visionSensor < 1 then
        sim.addLog(sim.verbosity_errors, 'External controller capture sensor creation failed')
        writeText(DONE_MARKER, 'failed\n')
        writeSummary(false, 'sensor_creation_failed')
        captureFinished = true
        return false
    end

    captureStarted = true
    nextFrameTime = simTime
    writeText(START_MARKER, 'init\n')
    writeText(SENSING_MARKER, 'capture\n')
    sim.addLog(sim.verbosity_scriptinfos, 'External controller capture started')
    return true
end

local function finalize(success, reason)
    if captureFinished then
        return
    end
    captureFinished = true
    writeSummary(success, reason)
    if captureCount > 0 then
        writeText(DONE_MARKER, 'done\n')
    end
end

function sysCall_init()
    writeText(LOAD_MARKER, 'loaded\n')
    sim.addLog(sim.verbosity_scriptinfos, 'UR5 external controller capture add-on starting')
end

function sysCall_sensing()
    sensingCount = sensingCount + 1
    if captureFinished then
        return
    end

    local simTime = sim.getSimulationTime()
    if sensingCount <= CAPTURE_SKIP_FRAMES then
        return
    end
    if not ensureCaptureStarted(simTime) then
        return
    end
    if simTime + 1e-9 < nextFrameTime then
        return
    end

    captureFrame(simTime)
    if captureCount >= FRAME_COUNT then
        finalize(true, 'completed')
    end
end

function sysCall_cleanup()
    if not captureFinished then
        local reason = 'interrupted'
        if captureCount == 0 then
            reason = 'no_frames'
        end
        finalize(captureCount > 0, reason)
    end
    sim.addLog(sim.verbosity_scriptinfos, string.format('External controller capture saved %d frames', captureCount))
end
