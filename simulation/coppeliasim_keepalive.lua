-- Minimal keepalive add-on: prevents CoppeliaSim from exiting in headless mode
-- while an external Python client connects and controls the simulator.
-- This add-on does NOT load scenes, spawn subprocesses, or manage signals.

sim = require 'sim'

local SHUTDOWN_SIGNAL = 'real_cartpole_shutdown'
local CONNECT_RELEASE_FILE = os.getenv('REAL_CARTPOLE_RPC_CONNECT_RELEASE_FILE') or ''
local CONNECT_READY_FILE = os.getenv('REAL_CARTPOLE_RPC_CONNECT_READY_FILE') or ''
local CONNECT_RELEASE_SEEN = false
local CONNECT_READY_WRITTEN = false

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
            '[keepalive] RPC release marker observed')
        if CONNECT_READY_FILE ~= '' and not CONNECT_READY_WRITTEN then
            writeText(CONNECT_READY_FILE, 'ready\n')
            CONNECT_READY_WRITTEN = true
            sim.addLog(sim.verbosity_scriptinfos,
                '[keepalive] RPC ready marker written')
        end
    end
end

function sysCall_info()
    return {autoStart = true}
end

function sysCall_init()
    sim.clearStringSignal(SHUTDOWN_SIGNAL)
    sim.addLog(sim.verbosity_scriptinfos,
        '[keepalive] active — waiting for external client on ZMQ')
    maybePublishReady()
end

function sysCall_thread()
    sim.addLog(sim.verbosity_scriptinfos,
        '[keepalive] keepalive thread active')
    while not CONNECT_RELEASE_SEEN do
        maybePublishReady()
        sim.wait(0.1, false)
    end
    while sim.getStringSignal(SHUTDOWN_SIGNAL) ~= '1' do
        sim.wait(0.1, false)
    end
    sim.addLog(sim.verbosity_scriptinfos, '[keepalive] shutdown signal received')
    sim.quitSimulator()
end

function sysCall_nonSimulation()
    maybePublishReady()
end

function sysCall_cleanup()
    sim.addLog(sim.verbosity_scriptinfos, '[keepalive] cleaned up')
end
