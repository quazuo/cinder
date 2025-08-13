module;

#define IMGUI_IMPL_VULKAN_HAS_DYNAMIC_RENDERING
#define IMGUI_DEFINE_MATH_OPERATORS
#define NOMINMAX 1
#include <../imgui/imgui.h>
#include <../imgui/backends/imgui_impl_glfw.h>
#include <../imgui/backends/imgui_impl_vulkan.h>

export module imgui;

export {
    using ::ImGui_ImplVulkan_InitInfo;
    using ::ImGuiIO;
    using ::ImGui_ImplGlfw_InitForOpenGL;
    using ::ImGui_ImplVulkan_Init;
    using ::ImGui_ImplVulkan_Shutdown;
    using ::ImGui_ImplGlfw_Shutdown;
    using ::ImGui_ImplVulkan_NewFrame;
    using ::ImGui_ImplGlfw_NewFrame;
    using ::ImGuiWindowFlags;
    using ::ImVec2;
    using ::ImVec4;
    using ::ImGui_ImplVulkan_RenderDrawData;
    using ::ImDrawList;
    using ::ImColor;
    using ::ImU32;

    using ::ImGuiWindowFlags_NoTitleBar;
    using ::ImGuiWindowFlags_NoCollapse;
    using ::ImGuiWindowFlags_NoSavedSettings;
    using ::ImGuiWindowFlags_NoResize;
    using ::ImGuiWindowFlags_NoMove;
    using ::ImGuiWindowFlags_AlwaysAutoResize;

    using ::ImGuiConfigFlags_NavEnableKeyboard;

    using ::ImGuiTreeNodeFlags_DefaultOpen;

    using ::ImGuiComboFlags_WidthFitPreview;

    using ::ImGuiHoveredFlags_AnyWindow;
}

export namespace ImGui {
    using ImGui::CreateContext;
    using ImGui::GetIO;
    using ImGui::StyleColorsDark;
    using ImGui::DestroyContext;
    using ImGui::NewFrame;
    using ImGui::SetNextWindowPos;
    using ImGui::SetNextWindowSize;
    using ImGui::Begin;
    using ImGui::End;
    using ImGui::Render;
    using ImGui::GetDrawData;
    using ImGui::CollapsingHeader;
    using ImGui::Text;
    using ImGui::Checkbox;
    using ImGui::Separator;
    using ImGui::Button;
    using ImGui::OpenPopup;
    using ImGui::DragFloat;
    using ImGui::SameLine;
    using ImGui::SliderFloat;
    using ImGui::ColorEdit3;
    using ImGui::BeginPopupModal;
    using ImGui::BeginCombo;
    using ImGui::Selectable;
    using ImGui::SetItemDefaultFocus;
    using ImGui::EndCombo;
    using ImGui::BeginDisabled;
    using ImGui::CloseCurrentPopup;
    using ImGui::EndDisabled;
    using ImGui::EndPopup;
    using ImGui::IsWindowHovered;
    using ImGui::IsAnyItemActive;
    using ImGui::IsAnyItemFocused;
    using ImGui::GetWindowDrawList;
    using ImGui::BeginChild;
    using ImGui::GetWindowPos;
    using ImGui::EndChild;
    using ImGui::RadioButton;
    using ImGui::SliderInt;
    using ImGui::ColorConvertHSVtoRGB;
}
