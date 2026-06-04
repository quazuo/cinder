module;

#define GLM_ENABLE_CXX_20
#define GLM_ENABLE_EXPERIMENTAL
#define GLM_FORCE_DEPTH_ZERO_TO_ONE

#define IMGUIZMO_USES_GLM
#define VGIZMO_USES_GLM

#define IMGUIZMO_IMGUI_FOLDER

#define IMGUI_IMPL_VULKAN_HAS_DYNAMIC_RENDERING
#define IMGUI_DEFINE_MATH_OPERATORS

#define NOMINMAX 1

#include <imgui.h>
#include <backends/imgui_impl_glfw.h>
#include <backends/imgui_impl_vulkan.h>
#include "imguizmo_quat.h"

export module imguizmo_quat;

export {
    using ::imguiGizmo;
}

export namespace ImGui {
    using ImGui::gizmo3D;
}
