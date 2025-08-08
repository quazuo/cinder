module;

export module Cinder.Render.Gui;

import imgui;
import vulkan_hpp;
import glfw;

export namespace zrx {
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
