module;

export module Cinder.Engine;

import imgui;
import imguizmo_quat;
import imfilebrowser;
import std;
import glfw;
import glm;
import vulkan;

#include <vulkan/vulkan_hpp_macros.hpp>

#if VULKAN_HPP_DISPATCH_LOADER_DYNAMIC == 1
VULKAN_HPP_DEFAULT_DISPATCH_LOADER_DYNAMIC_STORAGE
#endif

import Cinder.Utils;
import Cinder.Render;
import Cinder.Render.Graph;
import Cinder.Render.Vulkan;
import Cinder.Render.Gui;
import Cinder.Render.Mesh;
import Cinder.Globals;

export namespace zrx {
class Engine {
    GLFWwindow *window = nullptr;
    VulkanRenderer renderer;
    unique_ptr<InputManager> input_manager;

    unique_ptr<Camera> camera;

    glm::ivec2 window_size;

    float curr_delta_time = 0.0f;
    float last_time = 0.0f;

    // misc state variables

    float model_scale = 1.0f;
    glm::vec3 model_translate{};
    glm::quat model_rotation{1, 0, 0, 0};

    glm::quat light_direction = glm::normalize(glm::vec3(0, 1, 0));
    glm::vec3 light_color     = glm::normalize(glm::vec3(23.47, 21.31, 20.79));
    float light_intensity     = 20.0f;

    struct ShadowMapConfig {
        float left   = -10.0f;
        float right  = 10.0f;
        float bottom = -10.0f;
        float top    = 10.0f;
        float z_near = 0.01f;
        float z_far  = 100.0f;
        float bias_weight_1 = 0.0f;
        float bias_weight_2 = 0.0008f;
    } shadow_map_config;

    float debug_number = 0;

    struct RenderFrameSettings {
        bool is_gui_enabled        = true;
        bool show_debug_quad       = false;
        bool use_ssao              = false;
        bool should_capture_skybox = true;
        bool do_blur               = false;

        bool operator==(const RenderFrameSettings &rhs) const = default;
    } render_frame_settings;

    enum RenderingResource {
        Model_Scene,
        VB_Skybox,
        VB_ScreenSpaceQuad,
        UBO_General,
        UBO_Materials,
        Tex_Envmap,
        Tex_Skybox,
        Tex_GNormal,
        Tex_GPos,
        Tex_GDepth,
        Tex_SSAO,
        Tex_Shadowmap,
        Tex_BasePass,
        Tex_PostBlurX,
        Tex_PostBlurY,
        Tex_PostGui,
        Pipe_SsQuad,
        Pipe_CubeCapture,
        Pipe_Shadowmap,
        Pipe_Prepass,
        Pipe_SSAO,
        Pipe_Skybox,
        Pipe_Main,
        Pipe_BlurX,
        Pipe_BlurY,
        Pipe_Final,
        Count
    };

    enum_map<RenderingResource, ResourceHandle> render_resources;

public:
    Engine();

    void run();

private:
    void tick();

    void register_render_graph_resources();

    void build_render_graph();

    void update_graphics_uniform_buffer(const Buffer &buffer) const;

    void update_materials_uniform_buffer(const Buffer &buffer, ResourceHandle model_handle, const ResourceManager& resource_manager) const;

    void bind_key_actions();

    void bind_mouse_drag_actions();

    // ========================== gui ==========================

    void render_gui_section(float delta_time);
};
}
