module;

#define IMGUI_DEFINE_MATH_OPERATORS
#define NOMINMAX 1
#include <../imgui/imgui.h>
#include <../imgui/backends/imgui_impl_glfw.h>
#include <../imgui/backends/imgui_impl_vulkan.h>
#include "../imGuIZMO.quat/imGuIZMO.quat/imGuIZMOquat.h"

export module imguizmo_quat;

export {
    using ::imguiGizmo;
}

export namespace ImGui {
    using ImGui::gizmo3D;
}
