module;

#include "deps/LegitProfiler/ImGuiProfilerRenderer.h"
#include "deps/LegitProfiler/ProfilerTask.h"

export module legit_profiler;

export {
    using ImGuiUtils::ProfilerGraph;
    using ImGuiUtils::ProfilersWindow;
    using legit::ProfilerTask;
}