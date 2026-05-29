module;

export module Cinder.Render.Gui;

import std;
import imgui;
import vulkan;
import glfw;
import legit_profiler;

import Cinder.Globals;

export namespace zrx {
class GuiRenderer {
public:
    struct ProfilerNodeInfo {
        double start_time;
        double end_time;
        std::string name;
        ImColor color = ImColor(255, 255, 255, 255);
    };

private:
    static constexpr int PROFILER_FRAME_COUNT = 200;
    static constexpr int PROFILER_GRAPH_WIDTH = 700;
    static constexpr int PROFILER_LEGEND_WIDTH = 300;
    static constexpr int PROFILER_HEIGHT = 400;
    static constexpr int PROFILER_FRAME_WIDTH = 5;
    static constexpr float PROFILER_MAX_FRAME_TIME = 16.6f;
    static constexpr float PROFILER_HEIGHT_MULT = 1.3f;

    GLFWwindow *window;
    ProfilerGraph profiler_graph { PROFILER_FRAME_COUNT };

public:
    explicit GuiRenderer(GLFWwindow *w, ImGui_ImplVulkan_InitInfo &imgui_init_info);

    ~GuiRenderer();

    GuiRenderer(const GuiRenderer& other) = delete;
    GuiRenderer& operator=(const GuiRenderer& other) = delete;

    void begin_rendering();

    void end_rendering(const vk::raii::CommandBuffer& command_buffer);

    void render_profiler(const vector<ProfilerNodeInfo>& node_infos);
};
} // zrx
