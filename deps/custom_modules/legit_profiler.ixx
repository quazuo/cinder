module;

#include "deps/LegitProfiler/ImGuiProfilerRenderer.h"
#include "deps/LegitProfiler/ProfilerTask.h"

export module legit_profiler;

export {
    using ImGuiUtils::ProfilerGraph;
    using ImGuiUtils::ProfilersWindow;
    using legit::ProfilerTask;

    namespace legit::colors {
        using legit::Colors::turqoise;
        using legit::Colors::greenSea;
        using legit::Colors::emerald;
        using legit::Colors::nephritis;
        using legit::Colors::peterRiver;
        using legit::Colors::belizeHole;
        using legit::Colors::amethyst;
        using legit::Colors::wisteria;
        using legit::Colors::sunFlower;
        using legit::Colors::orange;
        using legit::Colors::carrot;
        using legit::Colors::pumpkin;
        using legit::Colors::alizarin;
        using legit::Colors::pomegranate;
        using legit::Colors::clouds;
        using legit::Colors::silver;
        using legit::Colors::imguiText;
    }
}