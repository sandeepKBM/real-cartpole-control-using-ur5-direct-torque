local REAL_CARTPOLE_ROOT = os.getenv('REAL_CARTPOLE_ROOT') or '/common/users/ss5772/real_Cartpole'
local COPPELIA_ROOT = os.getenv('COPPELIA_ROOT') or (REAL_CARTPOLE_ROOT .. '/third_party/coppelia_runtime/CoppeliaSim_Edu_V4_10_0_rev0_Ubuntu24_04')
local SCENE_PATH = COPPELIA_ROOT .. '/system/dfltscn.ttt'
local MODEL_PATH = COPPELIA_ROOT .. '/models/robots/non-mobile/UR5.ttm'
local OUTPUT_DIR = os.getenv('OUTPUT_DIR') or (REAL_CARTPOLE_ROOT .. '/outputs/control_runs/lua_direct_torque_probe')
local FRAME_DIR = os.getenv('FRAME_DIR') or (OUTPUT_DIR .. '/frames')
local SUMMARY_PATH = os.getenv('SUMMARY_PATH') or (OUTPUT_DIR .. '/lua_direct_torque_probe_summary.json')
local VIDEO_PATH = os.getenv('VIDEO_PATH') or (OUTPUT_DIR .. '/lua_direct_torque_probe.mp4')
local DONE_MARKER = os.getenv('DONE_MARKER') or (OUTPUT_DIR .. '/lua_direct_torque_probe_done.txt')
local LOAD_MARKER = os.getenv('LOAD_MARKER') or (OUTPUT_DIR .. '/lua_direct_torque_probe_loaded.txt')
local START_MARKER = os.getenv('START_MARKER') or (OUTPUT_DIR .. '/lua_direct_torque_probe_started.txt')
local SENSING_MARKER = os.getenv('SENSING_MARKER') or (OUTPUT_DIR .. '/lua_direct_torque_probe_sensing.txt')
local MODEL_LOADED_MARKER = os.getenv('MODEL_LOADED_MARKER') or (OUTPUT_DIR .. '/lua_direct_torque_probe_model_loaded.txt')

local FPS = tonumber(os.getenv('FPS') or '20')
local TOTAL_DURATION_S = tonumber(os.getenv('TOTAL_DURATION_S') or '4.0')
local ACTIVE_TORQUE_DURATION_S = tonumber(os.getenv('ACTIVE_TORQUE_DURATION_S') or '1.0')
local SETTLE_DURATION_S = tonumber(os.getenv('SETTLE_DURATION_S') or '0.5')
local MIN_ABS_DISPLACEMENT_RAD = tonumber(os.getenv('MIN_ABS_DISPLACEMENT_RAD') or '1e-5')
local TORQUE_NM = tonumber(os.getenv('TORQUE_NM') or '0.05')
local WARMUP_DURATION_S = tonumber(os.getenv('WARMUP_DURATION_S') or '0.5')
local CARTESIAN_FORCE_SCALE_N_PER_MPS2 = tonumber(os.getenv('CARTESIAN_FORCE_SCALE_N_PER_MPS2') or '4.0')
local MODEL_HEIGHT_SCALE = tonumber(os.getenv('MODEL_HEIGHT_SCALE') or '1.0')
local MODEL_BASE_Z_OFFSET_M = tonumber(os.getenv('MODEL_BASE_Z_OFFSET_M') or '0.0')
local ALL_JOINT_MICRO_TORQUE_VECTOR = {0.05, 0.03, -0.02, 0.0, 0.0, 0.0}

local JOINT_PATHS = {
    '/UR5/joint',
    '/UR5/link/joint',
    '/UR5/link/link/joint',
    '/UR5/link/link/link/joint',
    '/UR5/link/link/link/link/joint',
    '/UR5/link/link/link/link/link/joint',
}

local Y_REFERENCE_Q_START = {
    0.0,
    -0.1133064268431449,
    -0.664621645801302,
    4.921777393344012,
    -6.283185307179586,
    5.280928640069786,
}
local Y_REFERENCE_Q_END_POS = {
    2.35483229e-07,
    0.540370726,
    -1.0601276,
    4.62346643,
    -6.28336167,
    5.32106896,
}
local Y_TORQUE_KP = {5.0, 5.0, 5.5, 3.5, 3.0, 2.5}
local Y_TORQUE_KD = {1.0, 1.0, 1.1, 0.7, 0.6, 0.5}

local sim = require 'sim'

state = state or {}
state.jointHandles = state.jointHandles or {}
state.jointNames = state.jointNames or {}
state.candidatePathsTried = state.candidatePathsTried or {}
state.discoveredJointObjects = state.discoveredJointObjects or {}
state.selectedJointHandles = state.selectedJointHandles or {}
state.selectedJointNames = state.selectedJointNames or {}
state.jointResolutionAttempts = state.jointResolutionAttempts or 0
state.jointResolutionTimeout = tonumber(os.getenv('HANDLE_RESOLUTION_TIMEOUT_S') or '10.0')
state.torqueMode = string.lower(os.getenv('LUA_TORQUE_MODE') or 'single_joint_probe')
state.accelDirectionRaw = os.getenv('ACCEL_DIRECTION')
state.accelDirection = 1.0
state.accelDirectionSource = 'internal_default'
state.requiredUserInputs = state.requiredUserInputs or nil
state.internalDefaults = state.internalDefaults or nil
state.compatibilityFallbackInputs = state.compatibilityFallbackInputs or nil
state.directTorqueNote = state.directTorqueNote or nil
state.torqueModeConfigured = state.torqueModeConfigured or false
state.jointResolutionStartTime = state.jointResolutionStartTime or nil
state.jointResolutionWaitingLogged = state.jointResolutionWaitingLogged or false
state.endEffectorHandle = state.endEffectorHandle or nil
state.endEffectorResolvedPath = state.endEffectorResolvedPath or nil
state.modelReloadedInActuation = state.modelReloadedInActuation or false
state.actuationCount = state.actuationCount or 0
state.sensingCount = state.sensingCount or 0
state.firstActuationTime = state.firstActuationTime or nil
state.lastActuationTime = state.lastActuationTime or nil
state.simTimeTrace = state.simTimeTrace or {}
state.simTimeTraceLimit = state.simTimeTraceLimit or tonumber(os.getenv('SIM_TIME_TRACE_LIMIT') or '200')
state.phase = state.phase or 'init'
state.lastTorqueCmd = state.lastTorqueCmd or nil
state.lastTargetForceReadback = state.lastTargetForceReadback or nil
state.torqueApiSupported = state.torqueApiSupported or false
state.torqueApiReadbackSupported = state.torqueApiReadbackSupported or false
state.torqueApiUsed = state.torqueApiUsed or 'unknown'
state.signedTorqueApiProbeOk = state.signedTorqueApiProbeOk or false
state.signedTorqueApiProbeError = state.signedTorqueApiProbeError or nil
state.motionWindowReached = state.motionWindowReached or false
state.motionWindowCompleted = state.motionWindowCompleted or false
state.settleWindowReached = state.settleWindowReached or false
state.doneWindowReached = state.doneWindowReached or false
state.failureCategory = state.failureCategory or nil
state.failureStage = state.failureStage or nil
state.jointModeVerificationFailure = state.jointModeVerificationFailure or false
state.modelHeightScaleApplied = state.modelHeightScaleApplied or false
state.modelHeightScale = state.modelHeightScale or 1.0
state.modelBaseZOffsetM = state.modelBaseZOffsetM or 0.0
state.modelHeightScaleReferenceEeZ = state.modelHeightScaleReferenceEeZ or nil

local joints = state.jointHandles
local jointNames = state.jointNames

local control = {
    started = false,
    simulation_start_requested = false,
    finalized = false,
    manual_loop_running = false,
    error = nil,
    torque_api_available = false,
    torque_api_mode = 'setJointTargetForce_signed_plus_velocity_bias',
    joint_mode_summary = nil,
    frames = 0,
    next_frame_time = 0.0,
    q0 = nil,
    qf = nil,
    torque_start_time = nil,
    torque_end_time = nil,
    sim_time_start_s = nil,
    sim_time_end_s = nil,
    video_produced = false,
    motion_ok = false,
    direct_torque_supported = false,
    single_joint_active_logged = false,
    y_axis_active_logged = false,
}

local visionSensor = -1
local cameraPose = nil
local captureFrameIfNeeded

local function safeGetJointPosition(handle)
    if handle == nil or handle < 0 then
        return nil, 'invalid handle'
    end
    local ok, value = pcall(sim.getJointPosition, handle)
    if not ok then
        return nil, tostring(value)
    end
    return tonumber(value), nil
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

local function copyVec(v)
    local out = {}
    for i = 1, #v do
        out[i] = v[i]
    end
    return out
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
    local tv = type(v)
    if tv == 'number' then
        return jnum(v)
    end
    if tv == 'boolean' then
        return jbool(v)
    end
    if tv == 'string' then
        return jstr(v)
    end
    if tv ~= 'table' then
        return jstr(v)
    end
    local isArray = true
    local count = 0
    local maxIndex = 0
    for k, _ in pairs(v) do
        count = count + 1
        if type(k) ~= 'number' then
            isArray = false
        else
            if k > maxIndex then
                maxIndex = k
            end
        end
    end
    if isArray and maxIndex == count then
        local out = {}
        for i = 1, maxIndex do
            local item = v[i]
            local t = type(item)
            if t == 'number' then
                out[#out + 1] = jnum(item)
            elseif t == 'boolean' then
                out[#out + 1] = jbool(item)
            elseif t == 'table' then
                out[#out + 1] = jarr(item)
            else
                out[#out + 1] = jstr(item)
            end
        end
        return '[' .. table.concat(out, ',') .. ']'
    end
    local keys = {}
    for k, _ in pairs(v) do
        keys[#keys + 1] = k
    end
    table.sort(keys, function(a, b)
        return tostring(a) < tostring(b)
    end)
    local out = {}
    for _, k in ipairs(keys) do
        out[#out + 1] = jstr(tostring(k)) .. ':' .. jarr(v[k])
    end
    return '{' .. table.concat(out, ',') .. '}'
end

local function clamp(v, lo, hi)
    if v < lo then
        return lo
    end
    if v > hi then
        return hi
    end
    return v
end

local function sign(v)
    if v > 0 then
        return 1.0
    elseif v < 0 then
        return -1.0
    end
    return 0.0
end

local function parseAccelDirection(raw)
    if raw == nil then
        return 1.0, 'internal_default'
    end
    local s = tostring(raw):lower():gsub('%s+', '')
    if s == '1' or s == '+1' or s == 'y+' or s == '+y' or s == 'positive' or s == 'pos' then
        return 1.0, 'env_override'
    end
    if s == '-1' or s == 'y-' or s == '-y' or s == 'negative' or s == 'neg' then
        return -1.0, 'env_override'
    end
    local numeric = tonumber(s)
    if numeric ~= nil then
        if numeric >= 0 then
            return 1.0, 'env_override'
        end
        return -1.0, 'env_override'
    end
    return 1.0, 'internal_default'
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
    local yaw = math.rad(-50.0 + 20.0 * progress)
    local radius = 1.85
    local target = {0.0, 0.0, 0.58}
    local camPos = {
        radius * math.cos(yaw),
        radius * math.sin(yaw),
        target[3] + 0.28,
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
    sim.setObjectAlias(sensor, 'LuaDirectTorqueProbeCamera')
    return sim.getObject('/LuaDirectTorqueProbeCamera')
end

local function clearTable(t)
    for k in pairs(t) do
        t[k] = nil
    end
end

local function appendUnique(t, value)
    for i = 1, #t do
        if t[i] == value then
            return
        end
    end
    t[#t + 1] = value
end

local function safeObjectAlias(handle)
    if sim.getObjectAlias == nil then
        return nil
    end
    local ok, alias = pcall(sim.getObjectAlias, handle)
    if ok and alias ~= nil and alias ~= '' then
        return tostring(alias)
    end
    return nil
end

local function recordDiscoveredJoint(handle, sourcePath)
    local entry = {
        handle = handle,
        alias = safeObjectAlias(handle),
        source = sourcePath,
    }
    state.discoveredJointObjects[#state.discoveredJointObjects + 1] = entry
end

local function selectJointsFromHandles(handles, sourceLabel)
    if type(handles) ~= 'table' or #handles < #JOINT_PATHS then
        return false
    end
    clearTable(joints)
    clearTable(jointNames)
    clearTable(state.selectedJointHandles)
    clearTable(state.selectedJointNames)
    for i = 1, #JOINT_PATHS do
        local h = handles[i]
        if h == nil or h < 0 then
            return false
        end
        joints[i] = h
        jointNames[i] = sourceLabel and string.format('%s_%d', sourceLabel, i) or string.format('joint_%d', i)
        state.selectedJointHandles[i] = h
        state.selectedJointNames[i] = jointNames[i]
    end
    return true
end

local function collectSceneJointHandles()
    local scans = {}
    if robotModelHandle ~= nil and robotModelHandle >= 0 and sim.getObjectsInTree ~= nil and sim.object_joint_type ~= nil then
        local ok, handles = pcall(sim.getObjectsInTree, robotModelHandle, sim.object_joint_type, 0)
        if ok and type(handles) == 'table' then
            scans[#scans + 1] = {label = 'model_tree', handles = handles}
        end
    end
    if sim.getObjectsInTree ~= nil and sim.handle_scene ~= nil and sim.object_joint_type ~= nil then
        local ok, handles = pcall(sim.getObjectsInTree, sim.handle_scene, sim.object_joint_type, 0)
        if ok and type(handles) == 'table' then
            scans[#scans + 1] = {label = 'scene_tree', handles = handles}
        end
    end
    return scans
end

local function resolveJoints()
    state.jointResolutionAttempts = (state.jointResolutionAttempts or 0) + 1
    clearTable(state.candidatePathsTried)
    clearTable(state.discoveredJointObjects)

    local candidatePaths = {
        '/UR5/joint',
        '/UR5/link/joint',
        '/UR5/link/link/joint',
        '/UR5/link/link/link/joint',
        '/UR5/link/link/link/link/joint',
        '/UR5/link/link/link/link/link/joint',
    }
    local pathHandles = {}
    local pathOk = true
    for i, path in ipairs(candidatePaths) do
        appendUnique(state.candidatePathsTried, path)
        local ok, handle = pcall(sim.getObject, path)
        if ok and handle ~= nil and handle >= 0 then
            pathHandles[i] = handle
            recordDiscoveredJoint(handle, path)
        else
            pathOk = false
            break
        end
    end
    if pathOk and #pathHandles == #candidatePaths then
        if selectJointsFromHandles(pathHandles, 'path') then
            state.jointsResolved = true
            return true
        end
    end

    local scans = collectSceneJointHandles()
    for _, scan in ipairs(scans) do
        for _, handle in ipairs(scan.handles) do
            recordDiscoveredJoint(handle, scan.label)
        end
        if selectJointsFromHandles(scan.handles, scan.label) then
            state.jointsResolved = true
            return true
        end
    end

    if state.jointResolutionAttempts >= 1 and #state.selectedJointHandles == #JOINT_PATHS then
        state.jointsResolved = true
        return true
    end
    state.jointsResolved = false
    return false
end

local function allJointsResolved()
    if state.jointsResolved and #joints == #JOINT_PATHS then
        for i = 1, #JOINT_PATHS do
            if joints[i] == nil or joints[i] < 0 then
                return false
            end
        end
        return true
    end
    return false
end

local function resolveEndEffectorHandle()
    if state.endEffectorHandle ~= nil and state.endEffectorHandle >= 0 then
        return state.endEffectorHandle
    end
    local function createYAttachmentDummy(parentHandle, parentLabel)
        if state.torqueMode ~= 'y_axis_accel_direction' then
            return nil
        end
        if parentHandle == nil or parentHandle < 0 then
            return nil
        end
        if sim.createDummy == nil or sim.setObjectParent == nil then
            return nil
        end
        local okCreate, dummy = pcall(sim.createDummy, 0.025)
        if not okCreate or dummy == nil or dummy < 0 then
            return nil
        end
        pcall(sim.setObjectAlias, dummy, 'LuaDirectTorqueYAttachment')
        pcall(sim.setObjectParent, dummy, parentHandle, false)
        if sim.setObjectPosition ~= nil then
            pcall(sim.setObjectPosition, dummy, parentHandle, {0.0, 0.0, 0.08})
        end
        if sim.setObjectOrientation ~= nil then
            pcall(sim.setObjectOrientation, dummy, parentHandle, {0.0, 0.0, 0.0})
        end
        state.endEffectorHandle = dummy
        state.endEffectorResolvedPath = tostring(parentLabel or safeObjectAlias(parentHandle) or parentHandle) .. '/LuaDirectTorqueYAttachment'
        return dummy
    end
    local candidates = {
        '/UR5/UR5_connection',
        ':/UR5/UR5_connection',
        '/UR5_connection',
        ':/UR5_connection',
        '/UR5/tip',
        '/UR5/Tip',
        '/UR5/tool',
        '/UR5/Tool',
        '/UR5/end',
        '/UR5/End',
        '/UR5/ee',
        '/UR5/EE',
        '/UR5/endEffector',
        '/UR5/EndEffector',
        '/UR5/flange',
        '/UR5/Flange',
    }
    for _, path in ipairs(candidates) do
        local ok, handle = pcall(sim.getObject, path)
        if ok and handle ~= nil and handle >= 0 then
            state.endEffectorHandle = handle
            state.endEffectorResolvedPath = path
            return handle
        end
    end
    if robotModelHandle ~= nil and robotModelHandle >= 0 and sim.getObjectsInTree ~= nil then
        local ok, handles = pcall(sim.getObjectsInTree, robotModelHandle, -1, 0)
        if ok and type(handles) == 'table' and #handles > 0 then
            local function handleLooksLikeEndEffector(handle)
                if handle == nil or handle < 0 then
                    return false
                end
                for i = 1, #joints do
                    if joints[i] == handle then
                        return false
                    end
                end
                local alias = safeObjectAlias(handle)
                if alias == nil then
                    return false
                end
                local lowered = string.lower(alias)
                return lowered:find('tip', 1, true) ~= nil
                    or lowered:find('tool', 1, true) ~= nil
                    or lowered:find('tcp', 1, true) ~= nil
                    or lowered:find('ee', 1, true) ~= nil
                    or lowered:find('flange', 1, true) ~= nil
                    or lowered:find('wrist', 1, true) ~= nil
                    or lowered:find('end', 1, true) ~= nil
            end
            local fallback = nil
            for i = #handles, 1, -1 do
                local h = handles[i]
                local alias = safeObjectAlias(h)
                if alias ~= nil then
                    local lowered = string.lower(alias)
                    if handleLooksLikeEndEffector(h) then
                        state.endEffectorHandle = h
                        state.endEffectorResolvedPath = alias
                        return h
                    end
                    if fallback == nil and not lowered:find('joint', 1, true) then
                        fallback = h
                    end
                end
            end
            if fallback ~= nil then
                state.endEffectorHandle = fallback
                state.endEffectorResolvedPath = safeObjectAlias(fallback)
                local fallbackAlias = state.endEffectorResolvedPath or ''
                local lowered = string.lower(fallbackAlias)
                if state.torqueMode == 'y_axis_accel_direction'
                    and not (lowered:find('tip', 1, true) or lowered:find('tool', 1, true) or lowered:find('tcp', 1, true) or lowered:find('ee', 1, true) or lowered:find('flange', 1, true))
                then
                    local dummy = createYAttachmentDummy(fallback, fallbackAlias)
                    if dummy ~= nil then
                        return dummy
                    end
                end
            else
                state.endEffectorHandle = handles[#handles]
                state.endEffectorResolvedPath = safeObjectAlias(handles[#handles])
                if state.torqueMode == 'y_axis_accel_direction' then
                    local dummy = createYAttachmentDummy(handles[#handles], state.endEffectorResolvedPath)
                    if dummy ~= nil then
                        return dummy
                    end
                end
            end
            return state.endEffectorHandle
        end
    end
    if joints[6] ~= nil and joints[6] >= 0 then
        state.endEffectorHandle = joints[6]
        state.endEffectorResolvedPath = jointNames[6] or '/UR5/link/link/link/link/link/joint'
        if state.torqueMode == 'y_axis_accel_direction' then
            local dummy = createYAttachmentDummy(joints[6], state.endEffectorResolvedPath)
            if dummy ~= nil then
                return dummy
            end
        end
        return joints[6]
    end
    return nil
end

local function applyModelHeightScaleIfRequested()
    local scale = tonumber(MODEL_HEIGHT_SCALE or 1.0) or 1.0
    local baseOffset = tonumber(MODEL_BASE_Z_OFFSET_M or 0.0) or 0.0
    if state.modelHeightScaleApplied then
        return true
    end
    if math.abs(baseOffset) < 1e-9 then
        state.modelHeightScaleApplied = false
        state.modelHeightScale = 1.0
        state.modelBaseZOffsetM = 0.0
        return true
    end
    if robotModelHandle == nil or robotModelHandle < 0 then
        return false
    end
    if sim.setObjectPosition ~= nil then
        pcall(sim.setObjectPosition, robotModelHandle, sim.handle_world, {0.0, 0.0, baseOffset})
    end
    state.modelHeightScaleApplied = true
    state.modelHeightScale = scale
    state.modelBaseZOffsetM = baseOffset
    state.modelHeightScaleReferenceEeZ = nil
    state.eeInitialPosition = nil
    return true
end

local function getObjectPosition(handle)
    if handle == nil or handle < 0 then
        return nil
    end
    if sim.getObjectPosition == nil then
        return nil
    end
    local ok, value = pcall(sim.getObjectPosition, handle, sim.handle_world)
    if not ok or type(value) ~= 'table' or #value < 3 then
        return nil
    end
    return {tonumber(value[1]) or 0.0, tonumber(value[2]) or 0.0, tonumber(value[3]) or 0.0}
end

local function getObjectMatrix(handle)
    if handle == nil or handle < 0 or sim.getObjectMatrix == nil then
        return nil
    end
    local ok, value = pcall(sim.getObjectMatrix, handle, sim.handle_world)
    if not ok or type(value) ~= 'table' or #value < 12 then
        return nil
    end
    return value
end

local function jointWorldAxisFromMatrix(m)
    if m == nil then
        return nil
    end
    return {
        tonumber(m[3]) or 0.0,
        tonumber(m[7]) or 0.0,
        tonumber(m[11]) or 0.0,
    }
end

local function getJointVelocity(handle)
    if handle == nil or handle < 0 then
        return nil
    end
    if sim.getJointVelocity ~= nil then
        local ok, value = pcall(sim.getJointVelocity, handle)
        if ok and type(value) == 'number' then
            return value
        end
    end
    if sim.getObjectFloatParam ~= nil and sim.jointfloatparam_velocity ~= nil then
        local ok, value = pcall(sim.getObjectFloatParam, handle, sim.jointfloatparam_velocity)
        if ok and type(value) == 'number' then
            return value
        end
    end
    return nil
end

local function geometricTranslationalJacobian(eeHandle)
    local eePos = getObjectPosition(eeHandle)
    if eePos == nil then
        return nil
    end
    local jv = {}
    for i, h in ipairs(joints) do
        local jointPos = getObjectPosition(h)
        local jointMatrix = getObjectMatrix(h)
        local axis = jointWorldAxisFromMatrix(jointMatrix)
        if jointPos == nil or axis == nil then
            return nil
        end
        local delta = {eePos[1] - jointPos[1], eePos[2] - jointPos[2], eePos[3] - jointPos[3]}
        local col = cross(axis, delta)
        jv[i] = col
    end
    return jv, eePos
end

local function jacobianTransposeMultiplyColumns(jvCols, forceVec)
    local tau = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0}
    for i = 1, 6 do
        local col = jvCols and jvCols[i] or nil
        local cx = 0.0
        local cy = 0.0
        local cz = 0.0
        if type(col) == 'table' then
            cx = tonumber(col[1]) or 0.0
            cy = tonumber(col[2]) or 0.0
            cz = tonumber(col[3]) or 0.0
        end
        tau[i] = cx * (forceVec[1] or 0.0) + cy * (forceVec[2] or 0.0) + cz * (forceVec[3] or 0.0)
    end
    return tau
end

local function resolveAccDirection()
    state.accelDirection, state.accelDirectionSource = parseAccelDirection(state.accelDirectionRaw)
end

local function refreshLiveJoints(simTime)
    if allJointsResolved() then
        return true
    end
    if state.jointResolutionStartTime == nil then
        state.jointResolutionStartTime = simTime or sim.getSimulationTime()
    end
    if resolveJoints() and allJointsResolved() then
        if not state.modelHeightScaleApplied then
            applyModelHeightScaleIfRequested()
        end
        if not state.torqueModeConfigured then
            configureTorqueMode()
            state.torqueModeConfigured = true
        end
        return true
    end
    if not state.jointResolutionWaitingLogged then
        sim.addLog(sim.verbosity_scriptinfos, 'waiting for joint handles in actuation')
        state.jointResolutionWaitingLogged = true
    end
    local now = simTime or sim.getSimulationTime()
    local elapsed = now - (state.jointResolutionStartTime or now)
    if elapsed >= (state.jointResolutionTimeout or 10.0) then
        control.error = 'joint handles not resolved before timeout'
        return false
    end
    return false
end

local function readJointConfiguration()
    local q = {}
    for i, h in ipairs(joints) do
        local value, err = safeGetJointPosition(h)
        if value == nil then
            return nil, err
        end
        q[i] = value
    end
    return q, nil
end

local function readJointState()
    local q = {}
    local qd = {}
    for i, h in ipairs(joints) do
        local value, err = safeGetJointPosition(h)
        if value == nil then
            return nil, nil, err
        end
        q[i] = value
        qd[i] = getJointVelocity(h) or 0.0
    end
    return q, qd, nil
end

local function readJointConfigurationSummary()
    local jointsInfo = {}
    local motorVals, ctrlVals, dynVals, modeVals = {}, {}, {}, {}
    for i, h in ipairs(joints) do
        local entry = {handle = h, index = i, name = jointNames[i]}
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

local function configureTorqueMode()
    state.jointDynamicModeInfo = {}
    state.motorEnableInfo = {}
    state.controlModeInfo = {}
    for i, h in ipairs(joints) do
        local entry = {index = i, handle = h, name = jointNames[i]}
        if sim.getJointMode ~= nil then
            local ok, value = pcall(sim.getJointMode, h)
            if ok then
                entry.joint_mode = tonumber(value) or 0
                if sim.jointmode_dynamic ~= nil then
                    entry.is_dynamic = entry.joint_mode == sim.jointmode_dynamic
                end
            end
        end
        if sim.getObjectInt32Param ~= nil and sim.jointintparam_motor_enabled ~= nil then
            local ok, value = pcall(sim.getObjectInt32Param, h, sim.jointintparam_motor_enabled)
            if ok then
                entry.motor_enabled = tonumber(value) or 0
            end
        end
        if sim.getObjectInt32Param ~= nil and sim.jointintparam_ctrl_enabled ~= nil then
            local ok, value = pcall(sim.getObjectInt32Param, h, sim.jointintparam_ctrl_enabled)
            if ok then
                entry.ctrl_enabled = tonumber(value) or 0
            end
        end
        if sim.setJointMode ~= nil and sim.jointmode_dynamic ~= nil then
            pcall(sim.setJointMode, h, sim.jointmode_dynamic, 0)
        end
        if sim.setObjectInt32Param ~= nil and sim.jointintparam_motor_enabled ~= nil then
            pcall(sim.setObjectInt32Param, h, sim.jointintparam_motor_enabled, 1)
        end
        if sim.setObjectInt32Param ~= nil and sim.jointintparam_ctrl_enabled ~= nil then
            pcall(sim.setObjectInt32Param, h, sim.jointintparam_ctrl_enabled, 0)
        end
        if sim.setJointTargetVelocity ~= nil then
            pcall(sim.setJointTargetVelocity, h, 0.0)
        end
        if sim.setJointTargetForce ~= nil then
            pcall(sim.setJointTargetForce, h, 0.0, true)
        end
        state.jointDynamicModeInfo[#state.jointDynamicModeInfo + 1] = entry
        state.motorEnableInfo[#state.motorEnableInfo + 1] = {
            index = i,
            handle = h,
            motor_enabled = entry.motor_enabled,
            ctrl_enabled = entry.ctrl_enabled,
        }
        state.controlModeInfo[#state.controlModeInfo + 1] = {
            index = i,
            handle = h,
            joint_mode = entry.joint_mode,
            is_dynamic = entry.is_dynamic,
        }
    end
    state.torqueModeConfigurationStatus = 'best_effort'
    state.torqueModeConfigurationWarning = nil
    state.torqueModeConfigured = true
end

local function finiteNumber(x)
    return type(x) == 'number' and x == x and x ~= math.huge and x ~= -math.huge
end

local function zeroTorques()
    return {0.0, 0.0, 0.0, 0.0, 0.0, 0.0}
end

local function updatePhaseSchedule()
    local settleDuration = math.max(tonumber(os.getenv('SETTLE_DURATION_S') or tostring(SETTLE_DURATION_S)) or SETTLE_DURATION_S, 0.0)
    local warmupEnd = WARMUP_DURATION_S
    local motionEnd = warmupEnd + ACTIVE_TORQUE_DURATION_S
    local settleEnd = motionEnd + settleDuration
    local doneEnd = math.max(TOTAL_DURATION_S, settleEnd)
    state.phaseSchedule = {
        warmup_end = warmupEnd,
        motion_end = motionEnd,
        settle_end = settleEnd,
        done_end = doneEnd,
    }
    return state.phaseSchedule
end

local function getPhaseSchedule()
    if state.phaseSchedule == nil then
        return updatePhaseSchedule()
    end
    return state.phaseSchedule
end

local function phaseForTime(simTime)
    local schedule = getPhaseSchedule()
    if simTime == nil then
        return 'init'
    end
    if simTime < schedule.warmup_end then
        return 'warmup'
    end
    if simTime < schedule.motion_end then
        return 'motion'
    end
    if simTime < schedule.settle_end then
        return 'settle'
    end
    return 'done'
end

local function recordLifecycleSample(kind, simTime)
    if simTime == nil then
        return
    end
    local phase = phaseForTime(simTime)
    state.phase = phase
    if phase == 'motion' then
        state.motionWindowReached = true
    elseif phase == 'settle' then
        state.settleWindowReached = true
    elseif phase == 'done' then
        state.doneWindowReached = true
        state.motionWindowCompleted = true
    end
    local trace = state.simTimeTrace or {}
    if #trace < (state.simTimeTraceLimit or 200) then
        trace[#trace + 1] = {kind = kind, time = simTime, phase = phase}
        state.simTimeTrace = trace
    end
    if kind == 'actuation' then
        state.actuationCount = (state.actuationCount or 0) + 1
        if state.firstActuationTime == nil then
            state.firstActuationTime = simTime
        end
        state.lastActuationTime = simTime
    elseif kind == 'sensing' then
        state.sensingCount = (state.sensingCount or 0) + 1
    end
end

local function readTargetForceReadback()
    local readback = {}
    local supported = false
    if sim.getJointTargetForce ~= nil then
        supported = true
        for i, h in ipairs(joints) do
            local ok, value = pcall(sim.getJointTargetForce, h)
            if ok then
                readback[i] = tonumber(value) or 0.0
            else
                readback[i] = nil
            end
        end
    end
    state.torqueApiReadbackSupported = supported
    state.lastTargetForceReadback = readback
    return supported, readback
end

local function resampleTorqueProfile(requestedTorques, limit, maxBackoffIterations)
    local candidate = copyVec(requestedTorques)
    local attempts = 0
    local maxAbsRaw = 0.0
    for i = 1, #candidate do
        maxAbsRaw = math.max(maxAbsRaw, math.abs(tonumber(candidate[i]) or 0.0))
    end
    local maxAbsGuarded = maxAbsRaw
    local clippingApplied = false
    local scalingApplied = false
    -- Keep backing off the candidate torque profile until every element is
    -- inside the requested limit. This is deliberately resampling, not final
    -- clipping.
    while maxAbsGuarded > limit and attempts < (maxBackoffIterations or 8) do
        local scale = 0.9 * (limit / math.max(maxAbsGuarded, 1e-9))
        for i = 1, #candidate do
            candidate[i] = (tonumber(candidate[i]) or 0.0) * scale
        end
        maxAbsGuarded = 0.0
        for i = 1, #candidate do
            maxAbsGuarded = math.max(maxAbsGuarded, math.abs(tonumber(candidate[i]) or 0.0))
        end
        attempts = attempts + 1
        scalingApplied = true
    end
    return candidate, attempts, maxAbsRaw, maxAbsGuarded, scalingApplied, clippingApplied
end

local function qDelta(qf, q0)
    local out = {}
    if type(qf) ~= 'table' or type(q0) ~= 'table' then
        return out
    end
    for i = 1, math.max(#qf, #q0) do
        out[i] = (tonumber(qf[i]) or 0.0) - (tonumber(q0[i]) or 0.0)
    end
    return out
end

local function classifyFailure(torqueMode, errorText, simEnd, jointModeSummary)
    local text = tostring(errorText or '')
    if text:find('simulation_lifecycle_failed_before_motion_window', 1, true) then
        return 'lifecycle_failure'
    end
    if text:find('joint handles not resolved', 1, true)
        or text:find('end effector handle not resolved', 1, true)
        or text:find('joint position read failed', 1, true)
    then
        return 'joint_handle_failure'
    end
    if text:find('direct torque API unavailable', 1, true)
        or text:find('signed setJointTargetForce probe failed', 1, true)
        or text:find('torque API unavailable', 1, true)
    then
        return 'torque_api_failure'
    end
    if type(jointModeSummary) == 'table' then
        if jointModeSummary.motor_enabled_verified == false
            or jointModeSummary.ctrl_disabled_verified == false
            or jointModeSummary.dynamic_mode_verified == false
        then
            return 'joint_mode_verification_failure'
        end
    end
    if tonumber(simEnd or 0.0) < 1.0 then
        return 'lifecycle_failure'
    end
    if torqueMode == 'single_joint_probe' then
        return 'joint_motion_failure'
    end
    if torqueMode == 'all_joint_micro_torque_probe'
        or torqueMode == 'y_axis_constant_wrench_probe'
        or torqueMode == 'y_axis_accel_direction'
    then
        return 'controller_tracking_failure'
    end
    return nil
end

local function applyJointTorques(requestedTorques)
    local limit = math.abs(tonumber(os.getenv('LUA_DIRECT_TORQUE_MAX_NM') or '0.05'))
    state.luaDirectTorqueMaxNm = limit
    local sanitized = {}
    local nanInfTriggered = false
    for i = 1, 6 do
        local raw = tonumber(requestedTorques[i] or 0.0) or 0.0
        if not finiteNumber(raw) then
            nanInfTriggered = true
            raw = 0.0
        end
        sanitized[i] = raw
    end
    local clipped, backoffIterations, maxAbsRaw, maxAbsApplied, scalingApplied, clippingApplied = resampleTorqueProfile(sanitized, limit, 8)
    state.nanInfGuardTriggered = state.nanInfGuardTriggered or nanInfTriggered
    state.torqueClippingApplied = clippingApplied and true or false
    state.torqueScalingApplied = state.torqueScalingApplied or scalingApplied or backoffIterations > 0
    state.torqueBackoffIterations = math.max(tonumber(state.torqueBackoffIterations) or 0, backoffIterations)
    state.requestedTorqueNm = math.max(tonumber(state.requestedTorqueNm) or 0.0, maxAbsRaw)
    state.appliedTorqueNm = math.max(tonumber(state.appliedTorqueNm) or 0.0, maxAbsApplied)
    state.lastRequestedTorques = copyVec(sanitized)
    state.lastAppliedTorques = copyVec(clipped)
    state.lastTorqueCmd = copyVec(clipped)
    state.torqueGuardrailsEnabled = true

    local allSignedOk = true
    for i, h in ipairs(joints) do
        local torqueNm = clipped[i] or 0.0
        local ok = false
        if sim.setJointTargetForce ~= nil then
            local success, result = pcall(sim.setJointTargetForce, h, torqueNm, true)
            ok = success and (result == nil or result == true)
        end
        if not ok then
            allSignedOk = false
            break
        end
    end
    if allSignedOk then
        state.signedTorqueApiProbeOk = true
        state.torqueApiUsed = 'signed_setJointTargetForce'
        state.directTorquePurity = 'direct_signed_joint_force_command'
        control.torque_api_mode = 'signed_setJointTargetForce'
        state.signedTorqueApiProbeError = nil
        readTargetForceReadback()
        return clipped
    end

    state.signedTorqueApiProbeOk = false
    state.signedTorqueApiProbeError = 'signed setJointTargetForce probe failed'
    local largeVel = tonumber(os.getenv('LARGE_TORQUE_MODE_VELOCITY') or '10.0')
    for i, h in ipairs(joints) do
        local torqueNm = clipped[i] or 0.0
        if sim.setJointTargetVelocity ~= nil then
            pcall(sim.setJointTargetVelocity, h, (torqueNm >= 0.0 and 1.0 or -1.0) * largeVel)
        end
        if sim.setJointMaxForce ~= nil then
            pcall(sim.setJointMaxForce, h, math.abs(torqueNm))
        end
        if sim.setJointTargetForce ~= nil then
            pcall(sim.setJointTargetForce, h, math.abs(torqueNm), true)
        elseif sim.setJointForce ~= nil then
            pcall(sim.setJointForce, h, torqueNm)
        end
    end
    state.torqueApiUsed = 'target_velocity_plus_force_limit_fallback'
    state.directTorquePurity = 'approximate_motor_torque_mode'
    control.torque_api_mode = 'target_velocity_plus_force_limit_fallback'
    readTargetForceReadback()
    return clipped
end

local function applySafeTorques(active)
    local requested = zeroTorques()
    if active then
        requested[1] = TORQUE_NM
    end
    return applyJointTorques(requested)
end

local function jointVelocityDampingTorque(handle)
    local damping = tonumber(os.getenv('LUA_JOINT_DAMPING') or '0.01')
    local velocity = getJointVelocity(handle)
    if velocity == nil then
        return 0.0, false
    end
    return -damping * velocity, true
end

local function buildAllJointMicroTorqueProbeTorques(simTime)
    local schedule = getPhaseSchedule()
    state.torqueReferenceSource = 'fixed_joint_micro_torque_vector'
    state.directTorqueNote = 'This Lua lane uses simulator-side direct joint torque control via a fixed all-joint micro torque probe.'
    state.jacobianSource = 'unavailable'
    state.jacobianGuardrailsUsed = false
    state.jacobianRecomputedEachStep = false
    state.requiredUserInputs = {}
    state.internalDefaults = {
        TORQUE_VECTOR = ALL_JOINT_MICRO_TORQUE_VECTOR,
        ACTIVE_TORQUE_DURATION_S = ACTIVE_TORQUE_DURATION_S,
        TOTAL_DURATION_S = TOTAL_DURATION_S,
        SETTLE_DURATION_S = SETTLE_DURATION_S,
        LUA_DIRECT_TORQUE_MAX_NM = math.abs(tonumber(os.getenv('LUA_DIRECT_TORQUE_MAX_NM') or '0.05')),
    }
    state.compatibilityFallbackInputs = {}

    local phase = phaseForTime(simTime)
    state.phase = phase
    if phase == 'warmup' then
        return zeroTorques(), false, nil
    end
    if phase == 'motion' then
        state.motionWindowReached = true
        if state.eeInitialPosition == nil then
            local eeHandle = resolveEndEffectorHandle()
            state.eeInitialPosition = getObjectPosition(eeHandle)
        end
        return copyVec(ALL_JOINT_MICRO_TORQUE_VECTOR), false, nil
    end
    if phase == 'settle' then
        return zeroTorques(), false, nil
    end
    state.motionWindowCompleted = true
    return zeroTorques(), true, nil
end

local function buildYAxisConstantWrenchTorques(simTime)
    local accelMagnitudeEnv = os.getenv('ACCEL_MAGNITUDE_MPS2')
    local accelMagnitudeFallback = os.getenv('A_AXIS_MAX_MPS2')
    local travelDistanceEnv = os.getenv('TRAVEL_DISTANCE_M')
    local travelDistanceFallback = os.getenv('TARGET_DX_M')
    local accelMagnitude = math.abs(tonumber(accelMagnitudeEnv or accelMagnitudeFallback or '0.25'))
    local travelDistance = math.abs(tonumber(travelDistanceEnv or travelDistanceFallback or '0.35'))
    local directionSign = state.accelDirection or 1.0
    local forceScale = math.abs(CARTESIAN_FORCE_SCALE_N_PER_MPS2)
    local taskForce = directionSign * accelMagnitude * forceScale
    local phase = phaseForTime(simTime)
    state.phase = phase
    state.accelMagnitude = accelMagnitude
    state.travelDistance = travelDistance
    state.accelMagnitudeSource = (accelMagnitudeEnv ~= nil and accelMagnitudeEnv ~= '') and 'env_override' or ((accelMagnitudeFallback ~= nil and accelMagnitudeFallback ~= '') and 'compatibility_fallback_input' or 'internal_default')
    state.travelDistanceSource = (travelDistanceEnv ~= nil and travelDistanceEnv ~= '') and 'env_override' or ((travelDistanceFallback ~= nil and travelDistanceFallback ~= '') and 'compatibility_fallback_input' or 'internal_default')
    state.accelAxis = 'Y'
    state.targetAxis = 'Y'
    state.requiredUserInputs = {'ACCEL_DIRECTION'}
    state.internalDefaults = {
        ACCEL_MAGNITUDE_MPS2 = accelMagnitude,
        TRAVEL_DISTANCE_M = travelDistance,
        CARTESIAN_FORCE_SCALE_N_PER_MPS2 = forceScale,
        ACCEL_AXIS = 'Y',
        TARGET_AXIS = 'Y',
    }
    state.compatibilityFallbackInputs = {'TARGET_DX_M', 'A_AXIS_MAX_MPS2', 'TRANSPORT_AXIS'}
    state.directTorqueNote = 'This Lua Y-axis lane uses simulator-side direct joint torque control via a task-space wrench mapped with a Jacobian transpose.'
    state.torqueReferenceSource = 'jacobian_transpose_task_space_wrench'
    state.jacobianGuardrailsUsed = true
    state.jacobianRecomputedEachStep = true
    state.torqueTaskScale = taskForce

    local eeHandle = resolveEndEffectorHandle()
    if eeHandle == nil then
        return nil, nil, 'end effector handle not resolved'
    end
    local eePos = getObjectPosition(eeHandle)
    if eePos == nil then
        return nil, nil, 'end effector pose unavailable for y_axis_constant_wrench_probe torque mode'
    end
    if state.eeInitialPosition == nil then
        state.eeInitialPosition = copyVec(eePos)
    end
    local currentQ, currentQd = readJointState()
    if currentQ == nil or currentQd == nil then
        return nil, nil, 'joint state unavailable for y_axis_constant_wrench_probe torque mode'
    end

    local jvCols, jacobianEePos = geometricTranslationalJacobian(eeHandle)
    if jvCols == nil then
        state.jacobianSource = 'unavailable'
        return nil, nil, 'jacobian unavailable for y_axis_constant_wrench_probe torque mode'
    end
    state.jacobianSource = 'geometric_from_joint_transforms'
    local forceVec = {0.0, taskForce, 0.0}
    local tau = jacobianTransposeMultiplyColumns(jvCols, forceVec)
    local maxAbsRaw = 0.0
    for i = 1, 6 do
        local damp, haveDamp = jointVelocityDampingTorque(joints[i])
        if haveDamp then
            tau[i] = (tau[i] or 0.0) + damp
        end
        maxAbsRaw = math.max(maxAbsRaw, math.abs(tau[i] or 0.0))
    end
    state.maxAbsTauRawNm = maxAbsRaw
    state.jointDampingUsed = true
    state.targetAxisNetDisplacement = (jacobianEePos[2] or eePos[2] or 0.0) - (state.eeInitialPosition[2] or 0.0)
    control.target_axis_net_displacement_m = state.targetAxisNetDisplacement
    control.target_axis_start_m = state.eeInitialPosition[2]
    state.yTargetReached = math.abs(state.targetAxisNetDisplacement) >= travelDistance

    if phase == 'warmup' then
        return zeroTorques(), false, nil
    end
    if phase == 'motion' then
        return tau, false, nil
    end
    if phase == 'settle' then
        return zeroTorques(), false, nil
    end
    state.motionWindowCompleted = true
    return zeroTorques(), true, nil
end

local function buildYAxisAccelDirectionTorques(simTime)
    local accelMagnitudeEnv = os.getenv('ACCEL_MAGNITUDE_MPS2')
    local accelMagnitudeFallback = os.getenv('A_AXIS_MAX_MPS2')
    local travelDistanceEnv = os.getenv('TRAVEL_DISTANCE_M')
    local travelDistanceFallback = os.getenv('TARGET_DX_M')
    local accelMagnitude = math.abs(tonumber(accelMagnitudeEnv or accelMagnitudeFallback or '0.25'))
    local travelDistance = math.abs(tonumber(travelDistanceEnv or travelDistanceFallback or '0.35'))
    state.accelMagnitude = accelMagnitude
    state.travelDistance = travelDistance
    state.accelMagnitudeSource = (accelMagnitudeEnv ~= nil and accelMagnitudeEnv ~= '') and 'env_override' or ((accelMagnitudeFallback ~= nil and accelMagnitudeFallback ~= '') and 'compatibility_fallback_input' or 'internal_default')
    state.travelDistanceSource = (travelDistanceEnv ~= nil and travelDistanceEnv ~= '') and 'env_override' or ((travelDistanceFallback ~= nil and travelDistanceFallback ~= '') and 'compatibility_fallback_input' or 'internal_default')
    state.accelAxis = 'Y'
    state.targetAxis = 'Y'
    state.requiredUserInputs = {'ACCEL_DIRECTION'}
    state.internalDefaults = {
        ACCEL_MAGNITUDE_MPS2 = accelMagnitude,
        TRAVEL_DISTANCE_M = travelDistance,
        SETTLE_DURATION_S = SETTLE_DURATION_S,
        ACCEL_AXIS = 'Y',
        TARGET_AXIS = 'Y',
        Y_TORQUE_KP = Y_TORQUE_KP,
        Y_TORQUE_KD = Y_TORQUE_KD,
    }
    state.compatibilityFallbackInputs = {'TARGET_DX_M', 'A_AXIS_MAX_MPS2', 'TRANSPORT_AXIS'}
    state.directTorqueNote = 'This Lua Y-axis lane uses simulator-side direct joint torque control via a joint-space torque-tracked Y reference path.'
    state.torqueReferenceSource = 'joint_space_trajectory_template'
    state.jacobianSource = 'unavailable'
    state.jacobianGuardrailsUsed = false
    state.jacobianRecomputedEachStep = false

    local eeHandle = resolveEndEffectorHandle()
    if eeHandle == nil then
        return nil, nil, 'end effector handle not resolved'
    end
    local eePos = getObjectPosition(eeHandle)
    if eePos == nil then
        return nil, nil, 'end effector pose unavailable for y_axis_accel_direction torque mode'
    end
    if state.eeInitialPosition == nil then
        state.eeInitialPosition = copyVec(eePos)
    end

    local currentQ, currentQd = readJointState()
    if currentQ == nil or currentQd == nil then
        return nil, nil, 'joint state unavailable for y_axis_accel_direction torque mode'
    end

    local directionSign = state.accelDirection or 1.0
    local yStart = state.eeInitialPosition[2]
    local yDisp = eePos[2] - yStart

    state.targetAxisNetDisplacement = yDisp
    control.target_axis_net_displacement_m = yDisp
    control.target_axis_start_m = yStart
    state.torqueTaskScale = directionSign * accelMagnitude

    local modePhase = phaseForTime(simTime)
    state.phase = modePhase
    if modePhase == 'warmup' then
        return zeroTorques(), false, nil
    end
    if modePhase == 'settle' then
        return zeroTorques(), false, nil
    end
    if modePhase == 'done' then
        return zeroTorques(), true, nil
    end

    local motionDuration = math.max(ACTIVE_TORQUE_DURATION_S, 1e-6)
    local elapsed = math.max(0.0, simTime - WARMUP_DURATION_S)
    local phase = clamp(elapsed / motionDuration, 0.0, 1.0)
    local qDesired = {}
    local qdDesired = {}
    local baseDelta = {}
    for i = 1, 6 do
        local delta = (Y_REFERENCE_Q_END_POS[i] or 0.0) - (Y_REFERENCE_Q_START[i] or 0.0)
        if directionSign >= 0.0 then
            baseDelta[i] = delta
        else
            baseDelta[i] = -delta
        end
        qDesired[i] = (Y_REFERENCE_Q_START[i] or 0.0) + phase * baseDelta[i]
        qdDesired[i] = baseDelta[i] / motionDuration
    end

    local tau = {}
    local maxAbsRaw = 0.0
    for i = 1, 6 do
        local kp = tonumber(Y_TORQUE_KP[i] or 5.0)
        local kd = tonumber(Y_TORQUE_KD[i] or 1.0)
        local posErr = (qDesired[i] or 0.0) - (currentQ[i] or 0.0)
        local velErr = (qdDesired[i] or 0.0) - (currentQd[i] or 0.0)
        tau[i] = kp * posErr + kd * velErr
        maxAbsRaw = math.max(maxAbsRaw, math.abs(tau[i] or 0.0))
    end
    state.maxAbsTauRawNm = maxAbsRaw
    state.jointDampingUsed = true

    local limit = math.abs(tonumber(os.getenv('LUA_DIRECT_TORQUE_MAX_NM') or '0.50'))
    local guarded, backoffIterations, maxAbsRawCandidate, guardedMax, scalingApplied, clippingApplied = resampleTorqueProfile(tau, limit, 8)
    state.torqueBackoffIterations = backoffIterations
    state.torqueScalingApplied = scalingApplied
    state.maxAbsTauRawNm = math.max(state.maxAbsTauRawNm or 0.0, maxAbsRawCandidate)
    state.maxAbsTauGuardedNm = guardedMax
    state.torqueGuardrailsEnabled = true
    state.torqueClippingApplied = clippingApplied
    state.nanInfGuardTriggered = false

    local finished = false
    if state.accelDirection ~= nil and state.targetAxisNetDisplacement ~= nil then
        local reachedTarget = false
        if state.accelDirection > 0 then
            reachedTarget = state.targetAxisNetDisplacement >= travelDistance
        else
            reachedTarget = state.targetAxisNetDisplacement <= -travelDistance
        end
        state.yTargetReached = reachedTarget
    end
    if phase == 'done' then
        finished = true
    end

    return guarded, finished, nil
end

local function runSingleJointProbeLoop()
    control.manual_loop_running = true
    sim.addLog(sim.verbosity_scriptinfos, 'Lua direct torque probe starting stepped single-joint loop')
    if sim.setStepping ~= nil then
        pcall(sim.setStepping, true)
    end
    control.simulation_start_requested = true
    pcall(sim.startSimulation)

    local maxSteps = math.max(1, math.floor(TOTAL_DURATION_S * FPS * 4.0) + 10)
    local steps = 0
    while not control.finalized and steps < maxSteps do
        local simTime = sim.getSimulationTime()
        if control.sim_time_start_s == nil then
            control.sim_time_start_s = simTime
        end
        control.sim_time_end_s = simTime

        if not refreshLiveJoints(simTime) then
            if control.error ~= nil then
                break
            end
        else
            if control.q0 == nil then
                local q, err = readJointConfiguration()
                if q == nil then
                    control.error = 'joint position read failed: ' .. tostring(err)
                    break
                end
                control.q0 = copyVec(q)
            end

            if not state.torqueModeConfigured then
                control.joint_mode_summary = readJointConfigurationSummary()
                configureTorqueMode()
            end

            local withinActive = (simTime >= WARMUP_DURATION_S) and (simTime < (WARMUP_DURATION_S + ACTIVE_TORQUE_DURATION_S))
            if withinActive and not control.single_joint_active_logged then
                sim.addLog(sim.verbosity_scriptinfos, string.format(
                    'single-joint direct torque active: joint=1 torque=%.6f t=%.6f',
                    TORQUE_NM, simTime
                ))
                control.single_joint_active_logged = true
            end

            applySafeTorques(withinActive)
            captureFrameIfNeeded(simTime)

            if simTime >= TOTAL_DURATION_S then
                local qf, err = readJointConfiguration()
                if qf == nil then
                    control.error = 'joint position read failed: ' .. tostring(err)
                    break
                end
                control.qf = copyVec(qf)
                break
            end
        end

        if sim.step ~= nil then
            pcall(sim.step)
        else
            break
        end
        steps = steps + 1
    end

    local qf = control.qf
    if qf == nil then
        qf = readJointConfiguration()
    end
    if qf ~= nil then
        control.qf = copyVec(qf)
    end
    control.manual_loop_running = false
    finishAndQuit(control.error)
end

local function runYAxisAccelDirectionLoop()
    control.manual_loop_running = true
    sim.addLog(sim.verbosity_scriptinfos, 'Lua direct torque probe starting stepped Y-axis accel-direction loop')
    if sim.setStepping ~= nil then
        pcall(sim.setStepping, true)
    end
    control.simulation_start_requested = true
    pcall(sim.startSimulation)

    local maxSteps = math.max(1, math.floor(TOTAL_DURATION_S * FPS * 4.0) + 10)
    local steps = 0
    while not control.finalized and steps < maxSteps do
        local simTime = sim.getSimulationTime()
        if control.sim_time_start_s == nil then
            control.sim_time_start_s = simTime
        end
        control.sim_time_end_s = simTime

        if not refreshLiveJoints(simTime) then
            if control.error ~= nil then
                break
            end
        else
            if control.q0 == nil then
                local q, err = readJointConfiguration()
                if q == nil then
                    control.error = 'joint position read failed: ' .. tostring(err)
                    break
                end
                control.q0 = copyVec(q)
            end

            if not state.torqueModeConfigured then
                control.joint_mode_summary = readJointConfigurationSummary()
                configureTorqueMode()
            end

            local torques, finished, err = buildYAxisAccelDirectionTorques(simTime)
            if err ~= nil then
                control.error = err
                break
            end
            if torques == nil then
                -- fall through to advancing the simulation step
            else
                if not control.y_axis_active_logged then
                    sim.addLog(sim.verbosity_scriptinfos, string.format(
                        'Y-axis accel-direction direct torque active: direction=%.1f magnitude=%.6f travel=%.6f t=%.6f',
                        state.accelDirection or 1.0,
                        state.accelMagnitude or 0.0,
                        state.travelDistance or 0.0,
                        simTime
                    ))
                    control.y_axis_active_logged = true
                end

                applyJointTorques(torques)
                captureFrameIfNeeded(simTime)

                if finished then
                    local qf, readErr = readJointConfiguration()
                    if qf == nil then
                        control.error = 'joint position read failed: ' .. tostring(readErr)
                        break
                    end
                    control.qf = copyVec(qf)
                    break
                end
            end
        end

        if sim.step ~= nil then
            pcall(sim.step)
        else
            break
        end
        steps = steps + 1
    end

    local qf = control.qf
    if qf == nil then
        qf = readJointConfiguration()
    end
    if qf ~= nil then
        control.qf = copyVec(qf)
    end
    control.manual_loop_running = false
    finishAndQuit(control.error)
end

local function ensureCamera()
    if visionSensor >= 0 then
        return true
    end
    visionSensor = createCamera()
    if visionSensor == nil or visionSensor < 1 then
        return false
    end
    return true
end

captureFrameIfNeeded = function(simTime)
    if visionSensor < 1 or cameraPose == nil then
        return
    end
    if simTime + 1e-9 < control.next_frame_time then
        return
    end
    local function captureOnce()
        sim.setObjectMatrix(visionSensor, cameraPose, sim.handle_world)
        sim.handleVisionSensor(visionSensor)
        local img, res = sim.getVisionSensorImg(visionSensor)
        img = sim.transformImage(img, res, 4)
        local fileName = string.format('%s/frame_%08d.png', FRAME_DIR, control.frames)
        sim.saveImage(img, res, 0, fileName, -1)
        control.frames = control.frames + 1
        control.next_frame_time = control.next_frame_time + 1.0 / math.max(FPS, 1e-9)
    end
    local ok = pcall(captureOnce)
    if not ok then
        sim.addLog(sim.verbosity_scriptinfos, 'vision sensor handle stale after simulation start; recreating camera')
        visionSensor = -1
        if not ensureCamera() then
            return
        end
        ok = pcall(captureOnce)
        if not ok then
            return
        end
    end
end

local function writeSummary(forceError)
    local q0 = control.q0 or {0.0, 0.0, 0.0, 0.0, 0.0, 0.0}
    local qf = control.qf or q0
    local disp = (qf[1] or 0.0) - (q0[1] or 0.0)
    local absDisp = math.abs(disp)
    local torqueMode = state.torqueMode or 'single_joint_probe'
    local controllerFamily = 'lua_internal_direct_joint_torque_probe'
    local motionOk = absDisp >= MIN_ABS_DISPLACEMENT_RAD
    local yDisp = control.target_axis_net_displacement_m
    local yMotionOk = nil
    local requiredInputs = nil
    local accelMagnitudeSource = state.accelMagnitudeSource
    local travelDistanceSource = state.travelDistanceSource
    local requestedTorqueSummary = state.requestedTorqueNm
    local appliedTorqueSummary = state.appliedTorqueNm
    local directTorqueNote = state.directTorqueNote
    local simStart = control.sim_time_start_s or 0.0
    local simEnd = control.sim_time_end_s or sim.getSimulationTime()
    local jointModeSummary = control.joint_mode_summary or {joints = {}}
    local jointConfigurationStart = control.q0 or {0.0, 0.0, 0.0, 0.0, 0.0, 0.0}
    local jointConfigurationEnd = control.qf or jointConfigurationStart
    local jointConfigurationDelta = qDelta(jointConfigurationEnd, jointConfigurationStart)
    local eeStartPosition = state.eeInitialPosition
    local eeEndPosition = control.eeFinalPosition or getObjectPosition(resolveEndEffectorHandle())
    local eeDeltaPosition = qDelta(eeEndPosition or {}, eeStartPosition or {})
    local torqueApiSupported = control.torque_api_available == true
    local signedTorqueProbeOk = state.signedTorqueApiProbeOk == true
    local signedTorqueProbeError = state.signedTorqueApiProbeError
    local failureCategory = state.failureCategory
    local failureStage = state.failureStage

    if torqueMode == 'single_joint_probe' then
        requestedTorqueSummary = math.abs(TORQUE_NM)
        appliedTorqueSummary = math.min(math.abs(TORQUE_NM), math.abs(tonumber(os.getenv('LUA_DIRECT_TORQUE_MAX_NM') or '0.05')))
        requiredInputs = {'TORQUE_NM'}
        state.internalDefaults = {
            TORQUE_NM = math.abs(TORQUE_NM),
            ACTIVE_TORQUE_DURATION_S = ACTIVE_TORQUE_DURATION_S,
            TOTAL_DURATION_S = TOTAL_DURATION_S,
            SETTLE_DURATION_S = SETTLE_DURATION_S,
            LUA_DIRECT_TORQUE_MAX_NM = math.abs(tonumber(os.getenv('LUA_DIRECT_TORQUE_MAX_NM') or '0.05')),
        }
        state.compatibilityFallbackInputs = {}
        state.directTorqueNote = 'This Lua lane uses direct signed joint force on one UR5 joint.'
        directTorqueNote = state.directTorqueNote
    end

    if torqueMode == 'y_axis_accel_direction' then
        controllerFamily = 'lua_internal_y_axis_accel_direction_direct_torque'
        requiredInputs = state.requiredUserInputs or {'ACCEL_DIRECTION'}
        if yDisp ~= nil then
            local directionSign = state.accelDirection or 1.0
            local movedEnough = math.abs(yDisp) >= MIN_ABS_DISPLACEMENT_RAD
            local signOk = (yDisp * directionSign) > 0.0
            motionOk = movedEnough and signOk
        else
            motionOk = false
        end
        requestedTorqueSummary = state.maxAbsTauRawNm
        appliedTorqueSummary = state.maxAbsTauGuardedNm
        local travelDistance = state.travelDistance or tonumber(os.getenv('TRAVEL_DISTANCE_M') or '0.35')
        yMotionOk = motionOk
        state.travelTargetReached = state.yTargetReached == true
    elseif torqueMode == 'y_axis_constant_wrench_probe' then
        controllerFamily = 'lua_internal_y_axis_constant_wrench_direct_torque'
        requiredInputs = state.requiredUserInputs or {'ACCEL_DIRECTION'}
        if yDisp ~= nil then
            local directionSign = state.accelDirection or 1.0
            local movedEnough = math.abs(yDisp) >= MIN_ABS_DISPLACEMENT_RAD
            local signOk = (yDisp * directionSign) > 0.0
            motionOk = movedEnough and signOk
        else
            motionOk = false
        end
        requestedTorqueSummary = state.maxAbsTauRawNm
        appliedTorqueSummary = state.maxAbsTauGuardedNm
        yMotionOk = motionOk
        state.travelTargetReached = state.yTargetReached == true
    elseif torqueMode == 'all_joint_micro_torque_probe' then
        controllerFamily = 'lua_internal_all_joint_micro_torque_probe'
        requiredInputs = state.requiredUserInputs or {'TORQUE_NM'}
        local actuationCount = state.actuationCount or 0
        local anyJointMoved = false
        if control.q0 ~= nil and control.qf ~= nil then
            for i = 1, math.max(#control.q0, #control.qf) do
                if math.abs((control.qf[i] or 0.0) - (control.q0[i] or 0.0)) > 0.0 then
                    anyJointMoved = true
                    break
                end
            end
        end
        local eeMoved = false
        if eeStartPosition ~= nil and eeEndPosition ~= nil then
            for i = 1, 3 do
                if math.abs((eeEndPosition[i] or 0.0) - (eeStartPosition[i] or 0.0)) > 0.0 then
                    eeMoved = true
                    break
                end
            end
        end
        motionOk = actuationCount > 40 and (anyJointMoved or eeMoved)
        state.allJointMicroMotionOk = motionOk
        requestedTorqueSummary = state.maxAbsTauRawNm
        appliedTorqueSummary = state.maxAbsTauGuardedNm
    end

    local errorText = forceError
    local videoFound = fileExists(VIDEO_PATH)
    local videoNote = nil
    if motionOk and not videoFound then
        videoNote = 'Lua direct torque motion succeeded, but no video artifact was produced.'
    end
    if not motionOk and errorText == nil then
        local ranMotionWindow = state.motionWindowReached == true or (simEnd or 0.0) >= getPhaseSchedule().motion_end
        if (simEnd or 0.0) < 1.0 or not ranMotionWindow then
            errorText = 'simulation_lifecycle_failed_before_motion_window'
        elseif torqueMode == 'y_axis_accel_direction' then
            errorText = 'y-axis acceleration-direction torque motion did not reach the requested travel distance'
        elseif torqueMode == 'y_axis_constant_wrench_probe' then
            errorText = 'y-axis constant wrench motion did not move in the requested direction'
        elseif torqueMode == 'all_joint_micro_torque_probe' then
            errorText = 'all_joint_micro_torque_probe did not produce measurable motion'
        else
            errorText = 'joint did not move'
        end
    end
    local success = motionOk and errorText == nil
    if failureCategory == nil then
        failureCategory = classifyFailure(torqueMode, errorText, simEnd, jointModeSummary)
    end
    if failureStage == nil then
        if failureCategory == 'lifecycle_failure' then
            failureStage = 'cleanup'
        elseif failureCategory == 'joint_handle_failure' then
            failureStage = 'resolution'
        elseif failureCategory == 'torque_api_failure' then
            failureStage = 'torque_command'
        elseif failureCategory == 'joint_mode_verification_failure' then
            failureStage = 'mode_configuration'
        elseif failureCategory == 'controller_tracking_failure' or failureCategory == 'joint_motion_failure' then
            failureStage = 'motion'
        end
    end
    local lines = {}
    local function add(line)
        lines[#lines + 1] = line
    end

    add('{')
    add('  "success": ' .. jbool(success) .. ',')
    add('  "controller_family": ' .. jstr(controllerFamily) .. ',')
    add('  "lua_torque_mode": ' .. jstr(torqueMode) .. ',')
    add('  "uses_direct_torque_control": true,')
    add('  "external_python_zmq_validated": false,')
    add('  "stepping_owner": "coppeliasim_lua_or_internal",')
    add('  "simulation_started_by": "coppeliasim_or_lua",')
    add('  "lua_motion_enabled": true,')
    add('  "required_user_inputs": ' .. jarr(requiredInputs) .. ',')
    add('  "internal_defaults": ' .. jarr(state.internalDefaults) .. ',')
    add('  "compatibility_fallback_inputs": ' .. jarr(state.compatibilityFallbackInputs) .. ',')
    add('  "direct_torque_note": ' .. (directTorqueNote and jstr(directTorqueNote) or 'null') .. ',')
    add('  "torque_reference_source": ' .. (state.torqueReferenceSource and jstr(state.torqueReferenceSource) or 'null') .. ',')
    add('  "model_height_scale": ' .. jnum(state.modelHeightScale) .. ',')
    add('  "model_height_scale_applied": ' .. jbool(state.modelHeightScaleApplied) .. ',')
    add('  "model_base_z_offset_m": ' .. jnum(state.modelBaseZOffsetM) .. ',')
    add('  "model_height_scale_reference_ee_z": ' .. jnum(state.modelHeightScaleReferenceEeZ) .. ',')
    add('  "joint_handles_resolved": ' .. jbool(allJointsResolved()) .. ',')
    add('  "joint_resolution_attempts": ' .. tostring(state.jointResolutionAttempts or 0) .. ',')
    add('  "candidate_paths_tried": ' .. jarr(state.candidatePathsTried) .. ',')
    add('  "discovered_joint_objects": ' .. jarr(state.discoveredJointObjects) .. ',')
    add('  "selected_joint_names": ' .. jarr(state.selectedJointNames) .. ',')
    add('  "selected_joint_handles": ' .. jarr(state.selectedJointHandles) .. ',')
    add('  "joint_names": ' .. jarr(jointNames) .. ',')
    add('  "joint_handles": ' .. jarr(joints) .. ',')
    add('  "end_effector_handle": ' .. jnum(state.endEffectorHandle) .. ',')
    add('  "end_effector_resolved_path": ' .. (state.endEffectorResolvedPath and jstr(state.endEffectorResolvedPath) or 'null') .. ',')
    add('  "joint_mode_summary": {')
    add('    "motor_enabled_verified": ' .. jbool(jointModeSummary.motor_enabled_verified == true) .. ',')
    add('    "ctrl_disabled_verified": ' .. jbool(jointModeSummary.ctrl_disabled_verified == true) .. ',')
    add('    "dynamic_mode_verified": ' .. jbool(jointModeSummary.dynamic_mode_verified == true) .. ',')
    add('    "joint_mode_readback_available": ' .. jbool(jointModeSummary.joint_mode_readback_available) .. ',')
    add('    "motor_readback_available": ' .. jbool(jointModeSummary.motor_readback_available) .. ',')
    add('    "ctrl_readback_available": ' .. jbool(jointModeSummary.ctrl_readback_available))
    add('  },')
    add('  "joint_dynamic_mode_info": ' .. jarr(state.jointDynamicModeInfo) .. ',')
    add('  "motor_enable_info": ' .. jarr(state.motorEnableInfo) .. ',')
    add('  "control_mode_info": ' .. jarr(state.controlModeInfo) .. ',')
    add('  "torque_mode_configuration_status": ' .. jstr(state.torqueModeConfigurationStatus or 'best_effort') .. ',')
    add('  "torque_mode_configuration_warning": ' .. (state.torqueModeConfigurationWarning and jstr(state.torqueModeConfigurationWarning) or 'null') .. ',')
    add('  "torque_api_available": ' .. jbool(control.torque_api_available) .. ',')
    add('  "torque_api_mode": ' .. jstr(control.torque_api_mode) .. ',')
    add('  "torque_guardrails_enabled": ' .. jbool(state.torqueGuardrailsEnabled) .. ',')
    add('  "requested_torque_nm": ' .. jnum(requestedTorqueSummary) .. ',')
    add('  "applied_torque_nm": ' .. jnum(appliedTorqueSummary) .. ',')
    add('  "lua_direct_torque_max_nm": ' .. jnum(state.luaDirectTorqueMaxNm) .. ',')
    add('  "torque_clipping_applied": ' .. jbool(state.torqueClippingApplied) .. ',')
    add('  "nan_inf_guard_triggered": ' .. jbool(state.nanInfGuardTriggered) .. ',')
    add('  "torque_api_used": ' .. jstr(state.torqueApiUsed or 'unknown') .. ',')
    add('  "direct_torque_purity": ' .. jstr(state.directTorquePurity or 'unknown') .. ',')
    add('  "torque_api_supported": ' .. jbool(torqueApiSupported) .. ',')
    add('  "torque_api_readback_supported": ' .. jbool(state.torqueApiReadbackSupported == true) .. ',')
    add('  "signed_torque_api_probe_ok": ' .. jbool(signedTorqueProbeOk) .. ',')
    add('  "signed_torque_api_probe_error": ' .. (signedTorqueProbeError and jstr(signedTorqueProbeError) or 'null') .. ',')
    add('  "failure_category": ' .. (failureCategory and jstr(failureCategory) or 'null') .. ',')
    add('  "failure_stage": ' .. (failureStage and jstr(failureStage) or 'null') .. ',')
    add('  "phase": ' .. jstr(state.phase or 'init') .. ',')
    add('  "actuation_count": ' .. tostring(state.actuationCount or 0) .. ',')
    add('  "sensing_count": ' .. tostring(state.sensingCount or 0) .. ',')
    add('  "first_actuation_time": ' .. jnum(state.firstActuationTime) .. ',')
    add('  "last_actuation_time": ' .. jnum(state.lastActuationTime) .. ',')
    add('  "sim_time_trace": ' .. jarr(state.simTimeTrace) .. ',')
    add('  "last_torque_cmd": ' .. jarr(state.lastTorqueCmd) .. ',')
    add('  "last_target_force_readback": ' .. jarr(state.lastTargetForceReadback) .. ',')
    add('  "joint_0_initial_position_rad": ' .. jnum(q0[1]) .. ',')
    add('  "joint_0_final_position_rad": ' .. jnum(qf[1]) .. ',')
    add('  "joint_0_displacement_rad": ' .. jnum(disp) .. ',')
    add('  "abs_joint_0_displacement_rad": ' .. jnum(absDisp) .. ',')
    add('  "joint_0_displacement_nonzero": ' .. jbool(absDisp >= MIN_ABS_DISPLACEMENT_RAD) .. ',')
    add('  "joint_configuration_start": ' .. jarr(jointConfigurationStart) .. ',')
    add('  "joint_configuration_end": ' .. jarr(jointConfigurationEnd) .. ',')
    add('  "joint_configuration_delta": ' .. jarr(jointConfigurationDelta) .. ',')
    add('  "ee_start_position": ' .. jarr(eeStartPosition) .. ',')
    add('  "ee_end_position": ' .. jarr(eeEndPosition) .. ',')
    add('  "ee_delta_position": ' .. jarr(eeDeltaPosition) .. ',')
    add('  "torque_command_nm": ' .. jnum(TORQUE_NM) .. ',')
    add('  "active_torque_duration_s": ' .. jnum(ACTIVE_TORQUE_DURATION_S) .. ',')
    add('  "warmup_duration_s": ' .. jnum(WARMUP_DURATION_S) .. ',')
    add('  "total_duration_s": ' .. jnum(TOTAL_DURATION_S) .. ',')
    add('  "sim_time_start_s": ' .. jnum(simStart) .. ',')
    add('  "sim_time_end_s": ' .. jnum(simEnd) .. ',')
    add('  "frames": ' .. tostring(control.frames or 0) .. ',')
    add('  "video_produced": ' .. jbool(videoFound) .. ',')
    add('  "video_path": ' .. (videoFound and jstr(VIDEO_PATH) or 'null') .. ',')
    add('  "video_note": ' .. (videoNote and jstr(videoNote) or 'null') .. ',')
    add('  "external_python_zmq_validated": false,')

    if torqueMode == 'y_axis_accel_direction' then
        add('  "accel_axis": "Y",')
        add('  "accel_direction": ' .. jnum(state.accelDirection) .. ',')
        add('  "accel_direction_source": ' .. jstr(state.accelDirectionSource or 'internal_default') .. ',')
        add('  "accel_magnitude_mps2": ' .. jnum(state.accelMagnitude) .. ',')
        add('  "accel_magnitude_source": ' .. jstr(accelMagnitudeSource or 'internal_default') .. ',')
        add('  "travel_distance_m": ' .. jnum(state.travelDistance) .. ',')
        add('  "travel_distance_source": ' .. jstr(travelDistanceSource or 'internal_default') .. ',')
        add('  "target_axis": "Y",')
        add('  "target_axis_net_displacement_m": ' .. jnum(yDisp) .. ',')
        add('  "target_axis_sign_ok": ' .. jbool(yDisp ~= nil and ((yDisp * (state.accelDirection or 1.0)) > 0.0)) .. ',')
        add('  "travel_target_reached": ' .. jbool(state.travelTargetReached == true) .. ',')
        add('  "jacobian_source": ' .. jstr(state.jacobianSource or 'unavailable') .. ',')
        add('  "jacobian_guardrails_used": ' .. jbool(state.jacobianGuardrailsUsed) .. ',')
        add('  "jacobian_recomputed_each_step": ' .. jbool(state.jacobianRecomputedEachStep) .. ',')
        add('  "torque_task_scale": ' .. jnum(state.torqueTaskScale) .. ',')
        add('  "torque_scaling_applied": ' .. jbool(state.torqueScalingApplied) .. ',')
        add('  "torque_backoff_iterations": ' .. tostring(state.torqueBackoffIterations or 0) .. ',')
        add('  "max_abs_tau_raw_nm": ' .. jnum(state.maxAbsTauRawNm) .. ',')
        add('  "max_abs_tau_guarded_nm": ' .. jnum(state.maxAbsTauGuardedNm) .. ',')
    elseif torqueMode == 'y_axis_constant_wrench_probe' then
        add('  "accel_axis": "Y",')
        add('  "accel_direction": ' .. jnum(state.accelDirection) .. ',')
        add('  "accel_direction_source": ' .. jstr(state.accelDirectionSource or 'internal_default') .. ',')
        add('  "accel_magnitude_mps2": ' .. jnum(state.accelMagnitude) .. ',')
        add('  "accel_magnitude_source": ' .. jstr(accelMagnitudeSource or 'internal_default') .. ',')
        add('  "travel_distance_m": ' .. jnum(state.travelDistance) .. ',')
        add('  "travel_distance_source": ' .. jstr(travelDistanceSource or 'internal_default') .. ',')
        add('  "target_axis": "Y",')
        add('  "target_axis_net_displacement_m": ' .. jnum(yDisp) .. ',')
        add('  "target_axis_sign_ok": ' .. jbool(yDisp ~= nil and ((yDisp * (state.accelDirection or 1.0)) > 0.0)) .. ',')
        add('  "travel_target_reached": ' .. jbool(state.travelTargetReached == true) .. ',')
        add('  "jacobian_source": ' .. jstr(state.jacobianSource or 'unavailable') .. ',')
        add('  "jacobian_guardrails_used": ' .. jbool(state.jacobianGuardrailsUsed) .. ',')
        add('  "jacobian_recomputed_each_step": ' .. jbool(state.jacobianRecomputedEachStep) .. ',')
        add('  "torque_task_scale": ' .. jnum(state.torqueTaskScale) .. ',')
        add('  "torque_scaling_applied": ' .. jbool(state.torqueScalingApplied) .. ',')
        add('  "torque_backoff_iterations": ' .. tostring(state.torqueBackoffIterations or 0) .. ',')
        add('  "max_abs_tau_raw_nm": ' .. jnum(state.maxAbsTauRawNm) .. ',')
        add('  "max_abs_tau_guarded_nm": ' .. jnum(state.maxAbsTauGuardedNm) .. ',')
    elseif torqueMode == 'all_joint_micro_torque_probe' then
        add('  "micro_torque_vector": ' .. jarr(ALL_JOINT_MICRO_TORQUE_VECTOR) .. ',')
        add('  "micro_torque_duration_s": ' .. jnum(ACTIVE_TORQUE_DURATION_S) .. ',')
        add('  "actuation_count_ok": ' .. jbool((state.actuationCount or 0) > 40) .. ',')
        add('  "any_joint_moved": ' .. jbool(state.allJointMicroMotionOk == true) .. ',')
        add('  "jacobian_source": "unavailable",')
        add('  "jacobian_guardrails_used": false,')
        add('  "jacobian_recomputed_each_step": false,')
        add('  "torque_task_scale": null,')
        add('  "torque_scaling_applied": ' .. jbool(state.torqueScalingApplied) .. ',')
        add('  "torque_backoff_iterations": ' .. tostring(state.torqueBackoffIterations or 0) .. ',')
        add('  "max_abs_tau_raw_nm": ' .. jnum(state.maxAbsTauRawNm) .. ',')
        add('  "max_abs_tau_guarded_nm": ' .. jnum(state.maxAbsTauGuardedNm) .. ',')
    else
        add('  "accel_axis": null,')
        add('  "accel_direction": null,')
        add('  "accel_direction_source": null,')
        add('  "accel_magnitude_mps2": null,')
        add('  "accel_magnitude_source": null,')
        add('  "travel_distance_m": null,')
        add('  "travel_distance_source": null,')
        add('  "target_axis": null,')
        add('  "target_axis_net_displacement_m": null,')
        add('  "target_axis_sign_ok": null,')
        add('  "travel_target_reached": null,')
        add('  "jacobian_source": null,')
        add('  "jacobian_guardrails_used": false,')
        add('  "jacobian_recomputed_each_step": false,')
        add('  "torque_task_scale": null,')
        add('  "torque_scaling_applied": ' .. jbool(state.torqueScalingApplied) .. ',')
        add('  "torque_backoff_iterations": ' .. tostring(state.torqueBackoffIterations or 0) .. ',')
        add('  "max_abs_tau_raw_nm": null,')
        add('  "max_abs_tau_guarded_nm": null,')
    end

    add('  "motion_ok": ' .. jbool(motionOk) .. ',')
    add('  "error": ' .. (errorText and jstr(errorText) or 'null'))
    add('}')
    add('')
    writeText(SUMMARY_PATH, table.concat(lines, '\n'))
end

local function finishAndQuit(forceError)
    if control.finalized then
        return
    end
    control.finalized = true
    local savedTorqueSummary = {
        torque_api_used = state.torqueApiUsed,
        direct_torque_purity = state.directTorquePurity,
        requested_torque_nm = state.requestedTorqueNm,
        applied_torque_nm = state.appliedTorqueNm,
        lua_direct_torque_max_nm = state.luaDirectTorqueMaxNm,
        torque_clipping_applied = state.torqueClippingApplied,
        nan_inf_guard_triggered = state.nanInfGuardTriggered,
        torque_guardrails_enabled = state.torqueGuardrailsEnabled,
        last_requested_torques = state.lastRequestedTorques and copyVec(state.lastRequestedTorques) or nil,
        last_applied_torques = state.lastAppliedTorques and copyVec(state.lastAppliedTorques) or nil,
        last_torque_cmd = state.lastTorqueCmd and copyVec(state.lastTorqueCmd) or nil,
        last_target_force_readback = state.lastTargetForceReadback and copyVec(state.lastTargetForceReadback) or nil,
        signed_torque_api_probe_ok = state.signedTorqueApiProbeOk,
        signed_torque_api_probe_error = state.signedTorqueApiProbeError,
        torque_api_readback_supported = state.torqueApiReadbackSupported,
        torque_api_mode = control.torque_api_mode,
    }
    applySafeTorques(false)
    state.torqueApiUsed = savedTorqueSummary.torque_api_used
    state.directTorquePurity = savedTorqueSummary.direct_torque_purity
    state.requestedTorqueNm = savedTorqueSummary.requested_torque_nm
    state.appliedTorqueNm = savedTorqueSummary.applied_torque_nm
    state.luaDirectTorqueMaxNm = savedTorqueSummary.lua_direct_torque_max_nm
    state.torqueClippingApplied = savedTorqueSummary.torque_clipping_applied
    state.nanInfGuardTriggered = savedTorqueSummary.nan_inf_guard_triggered
    state.torqueGuardrailsEnabled = savedTorqueSummary.torque_guardrails_enabled
    state.lastRequestedTorques = savedTorqueSummary.last_requested_torques
    state.lastAppliedTorques = savedTorqueSummary.last_applied_torques
    state.lastTorqueCmd = savedTorqueSummary.last_torque_cmd
    state.lastTargetForceReadback = savedTorqueSummary.last_target_force_readback
    state.signedTorqueApiProbeOk = savedTorqueSummary.signed_torque_api_probe_ok
    state.signedTorqueApiProbeError = savedTorqueSummary.signed_torque_api_probe_error
    state.torqueApiReadbackSupported = savedTorqueSummary.torque_api_readback_supported
    control.torque_api_mode = savedTorqueSummary.torque_api_mode or control.torque_api_mode
    control.qf = control.qf or control.q0 or {0.0, 0.0, 0.0, 0.0, 0.0, 0.0}
    writeSummary(forceError or control.error)
    writeText(DONE_MARKER, 'done\n')
    if sim.getSimulationState() ~= nil and sim.getSimulationState() ~= sim.simulation_stopped then
        pcall(sim.stopSimulation)
    end
    sim.quitSimulator()
end

function sysCall_info()
    return {autoStart = true}
end

function sysCall_init()
    writeText(LOAD_MARKER, 'loaded\n')
    writeText(START_MARKER, 'init\n')
    sim.addLog(sim.verbosity_scriptinfos, 'Lua direct torque probe add-on starting')
    resolveAccDirection()
    updatePhaseSchedule()
    sim.setColorProperty(sim.handle_scene, 'ambientLight', {0.65, 0.65, 0.65}, {noError = true})
    sim.addLog(sim.verbosity_scriptinfos, 'Loading default scene for direct torque probe: ' .. SCENE_PATH)
    sim.loadScene(SCENE_PATH)
    sim.addLog(sim.verbosity_scriptinfos, 'Using default scene loaded in sysCall_init')
    local ok, modelHandle = pcall(sim.loadModel, MODEL_PATH)
    if not ok then
        control.error = 'UR5 model could not be loaded'
        sim.addLog(sim.verbosity_errors, 'Lua direct torque probe model load failed: ' .. tostring(modelHandle))
        writeSummary(control.error)
        writeText(DONE_MARKER, 'failed\n')
        writeText(MODEL_LOADED_MARKER, 'failed\n')
        sim.quitSimulator()
        return
    end
    robotModelHandle = tonumber(modelHandle) or -1
    state.endEffectorHandle = nil
    writeText(MODEL_LOADED_MARKER, string.format('loaded handle=%s\n', tostring(modelHandle)))
    sim.addLog(sim.verbosity_scriptinfos, 'UR5 model load returned handle: ' .. tostring(modelHandle))
    applyModelHeightScaleIfRequested()
    control.torque_api_available = (sim.setJointTargetForce ~= nil)
    state.torqueApiSupported = control.torque_api_available
    control.direct_torque_supported = control.torque_api_available
    if not control.torque_api_available then
        control.error = 'direct torque API unavailable'
        writeSummary(control.error)
        writeText(DONE_MARKER, 'failed\n')
        sim.quitSimulator()
        return
    end
    if not applyModelHeightScaleIfRequested() then
        sim.addLog(sim.verbosity_scriptinfos, string.format(
            'model height scale could not be applied immediately; requested scale=%.3f',
            tonumber(MODEL_HEIGHT_SCALE or 1.0) or 1.0
        ))
    end
    ensureCamera()
    if visionSensor < 1 then
        control.video_warning = 'video capture unavailable'
    end
    cameraPose = cameraMatrix(0, math.max(1, math.floor(TOTAL_DURATION_S * FPS)))
    control.next_frame_time = 0.0
    control.sim_time_start_s = sim.getSimulationTime()
    control.q0 = nil
    state.jointResolutionStartTime = control.sim_time_start_s
    if resolveJoints() and allJointsResolved() then
        applyModelHeightScaleIfRequested()
        control.joint_mode_summary = readJointConfigurationSummary()
        configureTorqueMode()
    else
        sim.addLog(sim.verbosity_scriptinfos, 'waiting for joint handles in actuation')
    end
    writeText(SENSING_MARKER, 'armed\n')
    sim.addLog(sim.verbosity_scriptinfos, 'Direct torque simulation start will be requested from non-simulation callback')
end

function sysCall_nonSimulation()
    if control.finalized or control.manual_loop_running then
        return
    end
    if control.simulation_start_requested then
        return
    end
    control.simulation_start_requested = true
    sim.addLog(sim.verbosity_scriptinfos, 'Direct torque simulation start requested from non-simulation callback')
    pcall(sim.startSimulation)
end

function sysCall_actuation()
    if control.finalized or control.manual_loop_running then
        return
    end
    local simTime = sim.getSimulationTime()
    recordLifecycleSample('actuation', simTime)
    if control.sim_time_start_s == nil then
        control.sim_time_start_s = simTime
    end
    control.sim_time_end_s = simTime
    if not refreshLiveJoints(simTime) then
        if control.error ~= nil then
            finishAndQuit(control.error)
        end
        return
    end
    if control.q0 == nil then
        local probeValue, probeErr = safeGetJointPosition(joints[1])
        if probeValue == nil then
            if not state.modelReloadedInActuation then
                sim.addLog(sim.verbosity_scriptinfos, 'joint handles stale after simulation start; reloading UR5 model once')
                local ok, modelHandle = pcall(sim.loadModel, MODEL_PATH)
                if not ok then
                    control.error = 'UR5 model could not be reloaded: ' .. tostring(modelHandle)
                    finishAndQuit(control.error)
                    return
                end
                robotModelHandle = tonumber(modelHandle) or robotModelHandle
                state.endEffectorHandle = nil
                state.jointsResolved = false
                state.modelReloadedInActuation = true
                if not resolveJoints() then
                    return
                end
                if not state.modelHeightScaleApplied then
                    applyModelHeightScaleIfRequested()
                end
                control.joint_mode_summary = readJointConfigurationSummary()
                configureTorqueMode()
                return
            end
            control.error = 'joint position read failed: ' .. tostring(probeErr)
            finishAndQuit(control.error)
            return
        end
        local q = {}
        for i, h in ipairs(joints) do
            local value, err = safeGetJointPosition(h)
            if value == nil then
                control.error = 'joint position read failed: ' .. tostring(err)
                finishAndQuit(control.error)
                return
            end
            q[i] = value
        end
        control.q0 = copyVec(q)
    end
    if state.eeInitialPosition == nil then
        local eeHandle = resolveEndEffectorHandle()
        local eePos = getObjectPosition(eeHandle)
        if eePos ~= nil then
            state.eeInitialPosition = copyVec(eePos)
        end
    end
    if not state.torqueModeConfigured then
        control.joint_mode_summary = readJointConfigurationSummary()
        configureTorqueMode()
    end
    if state.torqueMode == 'all_joint_micro_torque_probe' then
        local torques, finished, err = buildAllJointMicroTorqueProbeTorques(simTime)
        if err ~= nil then
            control.error = err
            finishAndQuit(control.error)
            return
        end
        if torques == nil then
            return
        end
        if state.phase == 'motion' and not control.micro_joint_active_logged then
            sim.addLog(sim.verbosity_scriptinfos, string.format(
                'all-joint micro torque active: vector=[%.4f, %.4f, %.4f, %.4f, %.4f, %.4f] t=%.6f',
                ALL_JOINT_MICRO_TORQUE_VECTOR[1], ALL_JOINT_MICRO_TORQUE_VECTOR[2], ALL_JOINT_MICRO_TORQUE_VECTOR[3],
                ALL_JOINT_MICRO_TORQUE_VECTOR[4], ALL_JOINT_MICRO_TORQUE_VECTOR[5], ALL_JOINT_MICRO_TORQUE_VECTOR[6],
                simTime
            ))
            control.micro_joint_active_logged = true
        end
        applyJointTorques(torques)
        captureFrameIfNeeded(simTime)
        if finished then
            local q = {}
            for i, h in ipairs(joints) do
                local value, readErr = safeGetJointPosition(h)
                if value == nil then
                    control.error = 'joint position read failed: ' .. tostring(readErr)
                    finishAndQuit(control.error)
                    return
                end
                q[i] = value
            end
            control.qf = copyVec(q)
            local eeHandle = resolveEndEffectorHandle()
            control.eeFinalPosition = getObjectPosition(eeHandle)
            finishAndQuit(control.error)
        end
        return
    end
    if state.torqueMode == 'y_axis_constant_wrench_probe' then
        local torques, finished, err = buildYAxisConstantWrenchTorques(simTime)
        if err ~= nil then
            control.error = err
            finishAndQuit(control.error)
            return
        end
        if torques == nil then
            return
        end
        if state.phase == 'motion' and not control.y_wrench_active_logged then
            sim.addLog(sim.verbosity_scriptinfos, string.format(
                'Y-axis constant wrench active: direction=%.1f forceScale=%.6f t=%.6f',
                state.accelDirection or 1.0,
                state.torqueTaskScale or 0.0,
                simTime
            ))
            control.y_wrench_active_logged = true
        end
        applyJointTorques(torques)
        captureFrameIfNeeded(simTime)
        if finished then
            local q = {}
            for i, h in ipairs(joints) do
                local value, readErr = safeGetJointPosition(h)
                if value == nil then
                    control.error = 'joint position read failed: ' .. tostring(readErr)
                    finishAndQuit(control.error)
                    return
                end
                q[i] = value
            end
            control.qf = copyVec(q)
            local eeHandle = resolveEndEffectorHandle()
            control.eeFinalPosition = getObjectPosition(eeHandle)
            finishAndQuit(control.error)
        end
        return
    end
    if state.torqueMode == 'y_axis_accel_direction' then
        local torques, finished, err = buildYAxisAccelDirectionTorques(simTime)
        if err ~= nil then
            control.error = err
            finishAndQuit(control.error)
            return
        end
        if torques == nil then
            return
        end
        applyJointTorques(torques)
        captureFrameIfNeeded(simTime)
        if finished then
            local q = {}
            for i, h in ipairs(joints) do
                local value, readErr = safeGetJointPosition(h)
                if value == nil then
                    control.error = 'joint position read failed: ' .. tostring(readErr)
                    finishAndQuit(control.error)
                    return
                end
                q[i] = value
            end
            control.qf = copyVec(q)
            local eeHandle = resolveEndEffectorHandle()
            control.eeFinalPosition = getObjectPosition(eeHandle)
            finishAndQuit(control.error)
        end
        return
    end

    local activeStart = WARMUP_DURATION_S
    local activeEnd = WARMUP_DURATION_S + ACTIVE_TORQUE_DURATION_S
    local withinActive = (state.phase == 'motion') or ((simTime >= activeStart) and (simTime < activeEnd))
    if withinActive then
        control.torque_start_time = control.torque_start_time or simTime
        control.torque_end_time = simTime
    elseif control.torque_start_time ~= nil and control.torque_end_time == nil then
        control.torque_end_time = simTime
    end
    applySafeTorques(withinActive)
    if simTime >= TOTAL_DURATION_S then
        local q = {}
        for i, h in ipairs(joints) do
            local value, err = safeGetJointPosition(h)
            if value == nil then
                control.error = 'joint position read failed: ' .. tostring(err)
                finishAndQuit(control.error)
                return
            end
            q[i] = value
        end
        control.qf = copyVec(q)
        finishAndQuit(control.error)
    end
end

function sysCall_sensing()
    if control.finalized or control.manual_loop_running then
        return
    end
    local simTime = sim.getSimulationTime()
    recordLifecycleSample('sensing', simTime)
    control.sim_time_end_s = simTime
    if not refreshLiveJoints(simTime) then
        return
    end
    writeText(SENSING_MARKER, string.format('time=%.6f frames=%d\n', simTime, control.frames or 0))
    captureFrameIfNeeded(simTime)
    local q = {}
    for i, h in ipairs(joints) do
        local value, err = safeGetJointPosition(h)
        if value == nil then
            control.error = 'joint position read failed: ' .. tostring(err)
            finishAndQuit(control.error)
            return
        end
        q[i] = value
    end
    control.qf = copyVec(q)
    if state.torqueMode == 'y_axis_accel_direction' then
        local eeHandle = resolveEndEffectorHandle()
        local eePos = getObjectPosition(eeHandle)
        if eePos ~= nil and state.eeInitialPosition ~= nil then
            control.target_axis_net_displacement_m = (eePos[2] or 0.0) - (state.eeInitialPosition[2] or 0.0)
        end
        if state.phase == 'done' or simTime >= getPhaseSchedule().done_end then
            finishAndQuit(control.error)
        end
    elseif state.torqueMode == 'y_axis_constant_wrench_probe' then
        local eeHandle = resolveEndEffectorHandle()
        local eePos = getObjectPosition(eeHandle)
        if eePos ~= nil and state.eeInitialPosition ~= nil then
            control.target_axis_net_displacement_m = (eePos[2] or 0.0) - (state.eeInitialPosition[2] or 0.0)
        end
        if state.phase == 'done' or simTime >= getPhaseSchedule().done_end then
            finishAndQuit(control.error)
        end
    elseif state.torqueMode == 'all_joint_micro_torque_probe' then
        if state.phase == 'done' or simTime >= getPhaseSchedule().done_end then
            finishAndQuit(control.error)
        end
    elseif simTime >= TOTAL_DURATION_S then
        local disp = (control.qf[1] or 0.0) - ((control.q0 and control.q0[1]) or 0.0)
        control.motion_ok = math.abs(disp) >= MIN_ABS_DISPLACEMENT_RAD
        finishAndQuit(control.error)
    end
end

function sysCall_cleanup()
    if not control.finalized then
        local cleanupSimTime = sim.getSimulationTime()
        recordLifecycleSample('cleanup', cleanupSimTime)
        refreshLiveJoints(cleanupSimTime)
        local q = {}
        for i, h in ipairs(joints) do
            local value, _ = safeGetJointPosition(h)
            q[i] = value or 0.0
        end
        control.qf = copyVec(q)
        local schedule = getPhaseSchedule()
        local lifecycleFailed = (control.sim_time_end_s or cleanupSimTime or 0.0) < 1.0
        if control.error == nil and lifecycleFailed then
            control.error = 'simulation_lifecycle_failed_before_motion_window'
        end
        if state.torqueMode == 'y_axis_accel_direction' then
            local eeHandle = resolveEndEffectorHandle()
            local eePos = getObjectPosition(eeHandle)
            control.eeFinalPosition = eePos and copyVec(eePos) or control.eeFinalPosition
            if eePos ~= nil and state.eeInitialPosition ~= nil then
                control.target_axis_net_displacement_m = (eePos[2] or 0.0) - (state.eeInitialPosition[2] or 0.0)
            end
            local ranMotionWindow = state.motionWindowReached == true or (control.sim_time_end_s or cleanupSimTime or 0.0) >= schedule.motion_end
            if control.error == nil and ranMotionWindow and (control.target_axis_net_displacement_m == nil or math.abs(control.target_axis_net_displacement_m) < MIN_ABS_DISPLACEMENT_RAD) then
                control.error = 'y-axis acceleration-direction torque motion did not reach the requested travel distance'
            end
        elseif state.torqueMode == 'y_axis_constant_wrench_probe' then
            local eeHandle = resolveEndEffectorHandle()
            local eePos = getObjectPosition(eeHandle)
            control.eeFinalPosition = eePos and copyVec(eePos) or control.eeFinalPosition
            if eePos ~= nil and state.eeInitialPosition ~= nil then
                control.target_axis_net_displacement_m = (eePos[2] or 0.0) - (state.eeInitialPosition[2] or 0.0)
            end
            local ranMotionWindow = state.motionWindowReached == true or (control.sim_time_end_s or cleanupSimTime or 0.0) >= schedule.motion_end
            if control.error == nil and ranMotionWindow then
                local signOk = control.target_axis_net_displacement_m ~= nil and ((control.target_axis_net_displacement_m * (state.accelDirection or 1.0)) > 0.0)
                if not signOk then
                    control.error = 'y-axis constant wrench motion did not move in the requested direction'
                end
            end
        elseif state.torqueMode == 'all_joint_micro_torque_probe' then
            local eeHandle = resolveEndEffectorHandle()
            local eePos = getObjectPosition(eeHandle)
            if eePos ~= nil then
                control.eeFinalPosition = copyVec(eePos)
            end
            local actuationCount = state.actuationCount or 0
            local anyJointMoved = false
            if control.q0 ~= nil and control.qf ~= nil then
                for i = 1, math.max(#control.q0, #control.qf) do
                    local delta = math.abs((control.qf[i] or 0.0) - (control.q0[i] or 0.0))
                    if delta > 0.0 then
                        anyJointMoved = true
                        break
                    end
                end
            end
            local eeMoved = false
            if state.eeInitialPosition ~= nil and control.eeFinalPosition ~= nil then
                for i = 1, 3 do
                    if math.abs((control.eeFinalPosition[i] or 0.0) - (state.eeInitialPosition[i] or 0.0)) > 0.0 then
                        eeMoved = true
                        break
                    end
                end
            end
            if control.error == nil and not (actuationCount > 40 and (anyJointMoved or eeMoved)) then
                control.error = 'all_joint_micro_torque_probe did not produce measurable motion'
            end
        else
            local disp = (control.qf[1] or 0.0) - ((control.q0 and control.q0[1]) or 0.0)
            control.motion_ok = math.abs(disp) >= MIN_ABS_DISPLACEMENT_RAD
            if control.error == nil and not control.motion_ok then
                control.error = 'joint did not move'
            end
        end
        writeSummary(control.error)
    end
    sim.addLog(sim.verbosity_scriptinfos, 'Lua direct torque probe cleanup')
end
