module;

#define IMGUI_DEFINE_MATH_OPERATORS
#define NOMINMAX 1
#include <../imgui/imgui.h>
#include <../imgui/backends/imgui_impl_glfw.h>
#include <../imgui/backends/imgui_impl_vulkan.h>
#include "../imgui-filebrowser/imfilebrowser.h"

export module imfilebrowser;

export namespace ImGui {
    using ImGui::FileBrowser;
}
