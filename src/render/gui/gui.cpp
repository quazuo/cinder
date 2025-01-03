#include "gui.hpp"

namespace zrx {
GuiRenderer::GuiRenderer(GLFWwindow *w, ImGui_ImplVulkan_InitInfo &imgui_init_info) : window(w) {
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO &io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;

    ImGui::StyleColorsDark();

    ImGui_ImplGlfw_InitForOpenGL(window, true);

    ImGui_ImplVulkan_Init(&imgui_init_info, nullptr);

    imguiGizmo::setGizmoFeelingRot(0.3);
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
}
