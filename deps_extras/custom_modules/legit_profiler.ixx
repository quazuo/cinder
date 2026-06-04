module;

#define GLM_ENABLE_CXX_20
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/glm.hpp>

#include <algorithm>
#include <sstream>
#include <chrono>

#include <ImGuiProfilerRenderer.h>
#include <ProfilerTask.h>

export module legit_profiler;

export {
    using ImGuiUtils::ProfilerGraph;
    using ImGuiUtils::ProfilersWindow;
    using legit::ProfilerTask;
}