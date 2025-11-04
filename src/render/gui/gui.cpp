module;

module Cinder.Render.Gui;

import imguizmo_quat;
import glm;
import glfw;

namespace zrx {
GuiRenderer::GuiRenderer(GLFWwindow *w, ImGui_ImplVulkan_InitInfo &imgui_init_info) : window(w) {
    // IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO &io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;

    ImGui::StyleColorsDark();

    ImGui_ImplGlfw_InitForOpenGL(window, true);

    ImGui_ImplVulkan_Init(&imgui_init_info);

    // imguiGizmo::setGizmoFeelingRot(0.3);
}

GuiRenderer::~GuiRenderer() {
    ImGui_ImplVulkan_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
}

void GuiRenderer::begin_rendering() {
    ImGui_ImplVulkan_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    constexpr ImGuiWindowFlags flags = ImGuiWindowFlags_NoTitleBar
                                       | ImGuiWindowFlags_NoCollapse
                                       | ImGuiWindowFlags_NoSavedSettings
                                       | ImGuiWindowFlags_NoResize
                                       | ImGuiWindowFlags_NoMove;

    ImGui::SetNextWindowPos(ImVec2(0, 0));

    glm::ivec2 window_size;
    glfwGetWindowSize(window, &window_size.x, &window_size.y);
    ImGui::SetNextWindowSize(ImVec2(0, window_size.y));

    ImGui::Begin("main window", nullptr, flags);
}

void GuiRenderer::end_rendering(const vk::raii::CommandBuffer &command_buffer) {
    ImGui::End();
    ImGui::Render();
    ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), *command_buffer);
}

void GuiRenderer::render_profiler(const vector<ProfilerNodeInfo> &node_infos) {
    static int graph_width              = PROFILER_GRAPH_WIDTH;
    static int legend_width             = PROFILER_LEGEND_WIDTH;
    static int height                   = PROFILER_HEIGHT;
    static float rolling_frame_time_avg = PROFILER_MAX_FRAME_TIME;

    ImGui::SliderInt("graph width",     &graph_width,   0, PROFILER_GRAPH_WIDTH * 2);
    ImGui::SliderInt("legend width",    &legend_width,  0, PROFILER_LEGEND_WIDTH * 2);
    ImGui::SliderInt("height",          &height,        0, PROFILER_HEIGHT * 2);

    if (node_infos.empty()) {
        profiler_graph.LoadFrameData(nullptr, 0);
        profiler_graph.RenderTimings(graph_width, legend_width, height, 0, rolling_frame_time_avg * PROFILER_HEIGHT_MULT);
        return;
    }

    vector<ProfilerTask> tasks;

    for (const auto& node_info: node_infos) {
        tasks.emplace_back(ProfilerTask {
            .startTime = node_info.start_time,
            .endTime = node_info.end_time,
            .name = node_info.name,
            .color = node_info.color,
        });
    }

    const auto curr_frame_time = static_cast<float>(tasks.back().endTime - tasks[0].startTime);
    constexpr float alpha = 0.01f;
    rolling_frame_time_avg = (alpha * curr_frame_time) + (1.0f - alpha) * rolling_frame_time_avg;

    profiler_graph.LoadFrameData(tasks.data(), tasks.size());
    profiler_graph.RenderTimings(graph_width, legend_width, height, 0, rolling_frame_time_avg * PROFILER_HEIGHT_MULT);
}
} // zrx
