#pragma once

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include "src/render/libs.hpp"

#define IMGUI_DEFINE_MATH_OPERATORS
#define NOMINMAX 1
#include <imgui/imgui.h>
#include <imgui/backends/imgui_impl_glfw.h>
#include <imgui/backends/imgui_impl_vulkan.h>
#include <imgui-filebrowser/imfilebrowser.h>
#include <imGuIZMO.quat/imGuIZMOquat.h>

namespace zrx {
class GuiRenderer {
    GLFWwindow *window;

public:
    explicit GuiRenderer(GLFWwindow *w, ImGui_ImplVulkan_InitInfo &imgui_init_info);

    ~GuiRenderer();

    GuiRenderer(const GuiRenderer& other) = delete;

    GuiRenderer& operator=(const GuiRenderer& other) = delete;

    void begin_rendering();

    void end_rendering(const vk::raii::CommandBuffer& command_buffer);
};
} // zrx
