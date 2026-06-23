local REAL_CARTPOLE_ROOT = os.getenv('REAL_CARTPOLE_ROOT') or '/common/users/ss5772/real_Cartpole'
local COPPELIA_ROOT = os.getenv('COPPELIA_ROOT') or (REAL_CARTPOLE_ROOT .. '/third_party/coppelia_runtime/CoppeliaSim_Edu_V4_10_0_rev0_Ubuntu24_04')
local LOG_TAG = '[real_cartpole_controller_bootstrap]'
local SHUTDOWN_SIGNAL = 'real_cartpole_controller_shutdown'
local CONNECT_RELEASE_FILE = os.getenv('REAL_CARTPOLE_RPC_CONNECT_RELEASE_FILE') or ''
local CONNECT_READY_FILE = os.getenv('REAL_CARTPOLE_RPC_CONNECT_READY_FILE') or ''
local MODEL_PATH = COPPELIA_ROOT .. '/models/robots/non-mobile/UR5.ttm'
local LEGACY_MARKER_HANDOFF = (CONNECT_RELEASE_FILE ~= '') or (CONNECT_READY_FILE ~= '')
local CONNECT_RELEASE_SEEN = false
local CONNECT_READY_WRITTEN = false
local NONSIM_LOGGED = false

sim = require 'sim'

function sysCall_info()
    return {autoStart = true}
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

local function maybePublishReady()
    if CONNECT_RELEASE_SEEN then
        return
    end
    if CONNECT_RELEASE_FILE ~= '' and fileExists(CONNECT_RELEASE_FILE) then
        CONNECT_RELEASE_SEEN = true
        sim.addLog(sim.verbosity_scriptinfos,
            string.format('%s RPC release marker observed: %s', LOG_TAG, CONNECT_RELEASE_FILE))
        if CONNECT_READY_FILE ~= '' and not CONNECT_READY_WRITTEN then
            writeText(CONNECT_READY_FILE, 'ready\n')
            CONNECT_READY_WRITTEN = true
            sim.addLog(sim.verbosity_scriptinfos,
                string.format('%s RPC ready marker written: %s', LOG_TAG, CONNECT_READY_FILE))
        end
    end
end

function sysCall_init()
    sim.clearStringSignal(SHUTDOWN_SIGNAL)
    sim.addLog(sim.verbosity_scriptinfos, string.format(
        '%s controller launch bootstrap active',
        LOG_TAG
    ))
    if not LEGACY_MARKER_HANDOFF then
        sim.loadModel(MODEL_PATH)
        sim.addLog(sim.verbosity_scriptinfos,
            string.format('%s idle bootstrap active; Python owns stepping and simulation start', LOG_TAG))
        return
    end
    sim.addLog(sim.verbosity_scriptinfos, string.format(
        '%s loading UR5 model: %s',
        LOG_TAG,
        MODEL_PATH
    ))
    sim.loadModel(MODEL_PATH)
    sim.addLog(sim.verbosity_scriptinfos, string.format(
        '%s waiting for RPC release marker',
        LOG_TAG
    ))
    while not CONNECT_RELEASE_SEEN do
        maybePublishReady()
        sim.wait(0.1)
    end
    sim.addLog(sim.verbosity_scriptinfos, string.format(
        '%s RPC released; starting stepped simulation',
        LOG_TAG
    ))
    local grace_s = tonumber(os.getenv('REAL_CARTPOLE_RPC_CONNECT_GRACE_S') or '1.0') or 1.0
    if grace_s > 0.0 then
        sim.addLog(sim.verbosity_scriptinfos, string.format(
            '%s grace window %.2fs before starting simulation',
            LOG_TAG,
            grace_s
        ))
        local grace_deadline = sim.getSystemTime() + grace_s
        while sim.getSystemTime() < grace_deadline do
            maybePublishReady()
            sim.wait(0.1)
        end
    end
    sim.setStepping(true)
    if sim.getSimulationState() == sim.simulation_stopped then
        sim.startSimulation()
        sim.addLog(sim.verbosity_scriptinfos, string.format(
            '%s stepped simulation started',
            LOG_TAG
        ))
    else
        sim.addLog(sim.verbosity_scriptinfos, string.format(
            '%s simulation already running',
            LOG_TAG
        ))
    end
    maybePublishReady()
end

function sysCall_cleanup()
    sim.addLog(sim.verbosity_scriptinfos, string.format('%s cleaned up', LOG_TAG))
end

function sysCall_nonSimulation()
    if LEGACY_MARKER_HANDOFF then
        return
    end
    if not NONSIM_LOGGED then
        NONSIM_LOGGED = true
        sim.addLog(sim.verbosity_scriptinfos,
            string.format('%s idle non-simulation loop keeping CoppeliaSim resident', LOG_TAG))
    end
    while sim.getStringSignal(SHUTDOWN_SIGNAL) ~= '1' do
        sim.wait(0.1)
    end
end
