import imgui;
import imguizmo_quat;
import imfilebrowser;
import std;
import glfw;
import glm;
import vulkan_hpp;

import Cinder.Utils;
import Cinder.Render;
import Cinder.Render.Graph;
import Cinder.Render.Vulkan;
import Cinder.Render.Gui;
import Cinder.Render.Mesh;
import Cinder.Globals;

// #define GLFW_INCLUDE_VULKAN
// #include <GLFW/glfw3.h>
//
// #define GLFW_EXPOSE_NATIVE_WIN32
// #define NOMINMAX 1
// #include <GLFW/glfw3native.h>

// #define VULKAN_HPP_ENABLE_STD_MODULE
// #define VULKAN_HPP_STD_MODULE
// #include <vulkan/vulkan_hpp_macros.hpp>

#if VULKAN_HPP_DISPATCH_LOADER_DYNAMIC == 1
VULKAN_HPP_DEFAULT_DISPATCH_LOADER_DYNAMIC_STORAGE
#endif

struct GraphicsUBO {
    struct WindowRes {
        uint32_t window_width;
        uint32_t window_height;
    };

    struct Matrices {
        glm::mat4 model;
        glm::mat4 view;
        glm::mat4 proj;
        glm::mat4 view_inverse;
        glm::mat4 proj_inverse;
        glm::mat4 vp_inverse;
        glm::mat4 static_view;
        glm::mat4 cubemap_capture_views[6];
        glm::mat4 cubemap_capture_proj;
    };

    struct MiscData {
        float debug_number;
        float z_near;
        float z_far;
        uint32_t use_ssao;
        float light_intensity;
        glm::vec3 light_dir;
        glm::vec3 light_color;
        glm::vec3 camera_pos;
    };

    alignas(16) WindowRes window{};
    alignas(16) Matrices matrices{};
    alignas(16) MiscData misc{};
};

namespace zrx {
class Engine {
    GLFWwindow *window = nullptr;
    VulkanRenderer renderer;
    unique_ptr<InputManager> input_manager;

    unique_ptr<Camera> camera;

    glm::ivec2 window_size;

    float curr_delta_time = 0.0f;
    float last_time = 0.0f;

    ImGui::FileBrowser file_browser;
    optional<FileType> current_type_being_chosen;
    std::unordered_map<FileType, std::filesystem::path> chosen_paths{};
    uint32_t load_scheme_idx = 0;

    string curr_error_message;

    // misc state variables

    float model_scale = 1.0f;
    glm::vec3 model_translate{};
    glm::quat model_rotation{1, 0, 0, 0};

    glm::quat light_direction = glm::normalize(glm::vec3(1, 1.5, -2));
    glm::vec3 light_color     = glm::normalize(glm::vec3(23.47, 21.31, 20.79));
    float light_intensity     = 20.0f;

    float debug_number = 0;

    bool is_gui_enabled        = true;
    bool show_debug_quad       = false;
    bool use_ssao              = false;
    bool should_capture_skybox = true;
    bool do_blur               = false;

public:
    Engine() {
        window = renderer.get_window();
        camera = make_unique<Camera>(window);

        input_manager = std::make_unique<InputManager>(window);
        bind_key_actions();
        bind_mouse_drag_actions();

        build_render_graph();
    }

    void run() {
        while (!glfwWindowShouldClose(window)) {
            tick();
        }

        renderer.wait_idle();
    }

private:
    void tick() {
        glfwPollEvents();

        const auto current_time = static_cast<float>(glfwGetTime());
        const float delta_time  = current_time - last_time;
        last_time               = current_time;
        curr_delta_time         = delta_time;

        input_manager->tick(delta_time);
        renderer.tick(delta_time);
        camera->tick(delta_time);

        glfwGetWindowSize(window, &window_size.x, &window_size.y);

        renderer.run_render_graph();
        should_capture_skybox = false;

        if (file_browser.HasSelected()) {
            const std::filesystem::path path = file_browser.GetSelected().string();

            if (*current_type_being_chosen == FileType::ENVMAP_HDR) {
                // renderer.load_environment_map(path);
            } else {
                chosen_paths[*current_type_being_chosen] = path;
            }

            file_browser.ClearSelected();
            current_type_being_chosen = {};
        }
    }

    void build_render_graph() {
        RenderGraph render_graph;

        // ================== models and vertex buffers ==================

        const auto scene_model = render_graph.add_resource(ModelResourceDesc{
            .name = "scene-model",
            .path = "../assets/example models/kettle/kettle.obj"
        });

        const auto skybox_vert_buf = render_graph.add_resource(VertexBufferResourceDesc{
            .name = "skybox-vb",
            .size = skybox_vertices.size() * sizeof(SkyboxVertex),
            .data = skybox_vertices.data()
        });

        const auto ss_quad_vert_buf = render_graph.add_resource(VertexBufferResourceDesc{
            .name = "ss-quad-vb",
            .size = screen_space_quad_vertices.size() * sizeof(ScreenSpaceQuadVertex),
            .data = screen_space_quad_vertices.data()
        });

        // ================== uniform buffers ==================

        const auto uniform_buffer = render_graph.add_resource(UniformBufferResourceDesc{
            .name = "general-ubo",
            .size = sizeof(GraphicsUBO)
        });

        render_graph.add_frame_begin_action([this, uniform_buffer](const FrameBeginActionContext &fba_ctx) {
            update_graphics_uniform_buffer(fba_ctx.resource_manager.get().get_buffer(uniform_buffer));
        });

        // ================== external textures ==================

        const auto base_color_texture = render_graph.add_resource(ExternalTextureResourceDesc{
            .name = "base-color-texture",
            .paths = {"../assets/example models/kettle/kettle-albedo.png"},
            .format = vk::Format::eR8G8B8A8Srgb
        });

        const auto normal_texture = render_graph.add_resource(ExternalTextureResourceDesc{
            .name = "normal-texture",
            .paths = {"../assets/example models/kettle/kettle-normal.png"},
            .format = vk::Format::eR8G8B8A8Unorm,
        });

        const auto orm_texture = render_graph.add_resource(ExternalTextureResourceDesc{
            .name = "orm-texture",
            .paths = {"../assets/example models/kettle/kettle-orm.png"},
           .format = vk::Format::eR8G8B8A8Unorm,
        });

        const auto envmap_texture = render_graph.add_resource(ExternalTextureResourceDesc{
            .name = "envmap-texture",
            .paths = {"../assets/envmaps/vienna.hdr"},
            .format = vk::Format::eR32G32B32A32Sfloat,
            .flags = vk::TextureFlagBitsZRX::HDR | vk::TextureFlagBitsZRX::MIPMAPS
        });

        // ================== render target textures ==================

        constexpr auto skybox_tex_format = vk::Format::eR8G8B8A8Srgb;
        const auto skybox_texture = render_graph.add_resource(TargetTextureResourceDesc{
            .name = "skybox-texture",
            .format = skybox_tex_format,
            .extent = {2048, 2048},
            .flags = vk::TextureFlagBitsZRX::CUBEMAP // | vk::TextureFlagBitsZRX::MIPMAPS
        });

        constexpr auto g_buffer_color_format = vk::Format::eR16G16B16A16Sfloat;
        const auto g_buffer_normal = render_graph.add_resource(TargetTextureResourceDesc{
            .name = "g-buffer-normal",
            .format = g_buffer_color_format,
        });

        const auto g_buffer_pos = render_graph.add_resource(TargetTextureResourceDesc{
            .name = "g-buffer-pos",
            .format = g_buffer_color_format,
        });

        constexpr auto g_buffer_depth_format = vk::Format::eD32Sfloat;
        const auto g_buffer_depth = render_graph.add_resource(TargetTextureResourceDesc{
            .name = "g-buffer-depth",
            .format = g_buffer_depth_format,
            .flags = {},
        });

        constexpr auto ssao_tex_format = vk::Format::eR8G8B8A8Unorm;
        const auto ssao_texture = render_graph.add_resource(TargetTextureResourceDesc{
            .name = "ssao-texture",
            .format = ssao_tex_format,
        });

        constexpr auto final_no_gamma_format = vk::Format::eR8G8B8A8Unorm;
        const auto base_pass_texture = render_graph.add_resource(TargetTextureResourceDesc{
            .name = "base-pass-texture",
            .format = final_no_gamma_format,
            .flags = {}
        });
        const auto post_blur_x_texture = render_graph.add_resource(TargetTextureResourceDesc{
            .name = "post-blur-x-texture",
            .format = final_no_gamma_format,
            .flags = {}
        });
        const auto post_blur_y_texture = render_graph.add_resource(TargetTextureResourceDesc{
            .name = "post-blur-y-texture",
            .format = final_no_gamma_format,
            .flags = {}
        });
        const auto post_gui_texture = render_graph.add_resource(TargetTextureResourceDesc{
            .name = "post-blur-y-texture",
            .format = final_no_gamma_format,
            .flags = {}
        });

        // ================== shaders ==================

        const auto cubecap_pipeline = render_graph.add_pipeline(GraphicsPipelineDesc{
            .vertex_path = "../shaders/obj/sphere-cube-vert.spv",
            .fragment_path = "../shaders/obj/sphere-cube-frag.spv",
            .binding_descriptions = SkyboxVertex::get_binding_descriptions(),
            .attr_descriptions = SkyboxVertex::get_attribute_descriptions(),
            .color_formats = {skybox_tex_format},
            .depth_format = {},
            .custom_properties = {
                .multiview_count = 6
            }
        });

        const auto prepass_pipeline = render_graph.add_pipeline(GraphicsPipelineDesc{
            .vertex_path = "../shaders/obj/prepass-vert.spv",
            .fragment_path = "../shaders/obj/prepass-frag.spv",
            .binding_descriptions = ModelVertex::get_binding_descriptions(),
            .attr_descriptions = ModelVertex::get_attribute_descriptions(),
            .color_formats = {g_buffer_color_format, g_buffer_color_format},
            .depth_format = g_buffer_depth_format
        });

        const auto ssao_pipeline = render_graph.add_pipeline(GraphicsPipelineDesc{
            .vertex_path = "../shaders/obj/ssao-vert.spv",
            .fragment_path = "../shaders/obj/ssao-frag.spv",
            .binding_descriptions = ScreenSpaceQuadVertex::get_binding_descriptions(),
            .attr_descriptions = ScreenSpaceQuadVertex::get_attribute_descriptions(),
            .color_formats = {ssao_tex_format}
        });

        const auto skybox_pipeline = render_graph.add_pipeline(GraphicsPipelineDesc{
            .vertex_path = "../shaders/obj/skybox-vert.spv",
            .fragment_path = "../shaders/obj/skybox-frag.spv",
            .binding_descriptions = SkyboxVertex::get_binding_descriptions(),
            .attr_descriptions = SkyboxVertex::get_attribute_descriptions(),
            .color_formats = {final_no_gamma_format},
            .depth_format = FINAL_FORMAT,
            .custom_properties = {
                .depth_compare_op = vk::CompareOp::eLessOrEqual,
            }
        });

        const auto main_pipeline = render_graph.add_pipeline(GraphicsPipelineDesc{
            .vertex_path = "../shaders/obj/main-vert.spv",
            .fragment_path = "../shaders/obj/main-frag.spv",
            .binding_descriptions = ModelVertex::get_binding_descriptions(),
            .attr_descriptions = ModelVertex::get_attribute_descriptions(),
            .color_formats = {final_no_gamma_format},
            .depth_format = FINAL_FORMAT
        });

        const auto blur_x_pipeline = render_graph.add_pipeline(ComputePipelineDesc{
            .path = "../shaders/obj/blur-x-comp.spv",
        });

        const auto blur_y_pipeline = render_graph.add_pipeline(ComputePipelineDesc{
            .path = "../shaders/obj/blur-y-comp.spv",
        });

        const auto final_pipeline = render_graph.add_pipeline(GraphicsPipelineDesc{
            .vertex_path = "../shaders/obj/ss-quad-vert.spv",
            .fragment_path = "../shaders/obj/ss-quad-frag.spv",
            .binding_descriptions = ScreenSpaceQuadVertex::get_binding_descriptions(),
            .attr_descriptions = ScreenSpaceQuadVertex::get_attribute_descriptions(),
            .color_formats = {FINAL_FORMAT},
            .depth_format = FINAL_FORMAT
        });

        // ================== nodes ==================

        render_graph.add_node(RenderNodeGraphics {
            .name = "cubemap-capture",
            .bound_resources = {uniform_buffer, envmap_texture},
            .color_targets = {skybox_texture},
            .body = [=](RenderPassContext &ctx) {
                ctx.bind_pipeline(cubecap_pipeline);
                ctx.bind_resources({uniform_buffer, envmap_texture});
                ctx.draw(skybox_vert_buf, skybox_vertices.size(), 1, 0, 0);
            },
            .should_run_predicate = [&] { return should_capture_skybox; },
            .custom_properties = RenderNodeGraphics::CustomProperties {
                .multiview_count = 6
            }
        });

        render_graph.add_node(RenderNodeGraphics {
            .name = "prepass",
            .bound_resources = {uniform_buffer},
            .color_targets = {g_buffer_normal, g_buffer_pos},
            .depth_target = g_buffer_depth,
            .body = [=](RenderPassContext &ctx) {
                ctx.bind_pipeline(prepass_pipeline);
                ctx.bind_resources({uniform_buffer});
                ctx.draw_model(scene_model);
            },
            .should_run_predicate = [&] { return use_ssao; }
        });

        render_graph.add_node(RenderNodeGraphics {
            .name = "ssao",
            .bound_resources = {uniform_buffer, g_buffer_depth, g_buffer_normal, g_buffer_pos},
            .color_targets = {ssao_texture},
            .body = [=](RenderPassContext &ctx) {
                ctx.bind_pipeline(ssao_pipeline);
                ctx.bind_resources({uniform_buffer, g_buffer_depth, g_buffer_normal, g_buffer_pos});
                ctx.draw(ss_quad_vert_buf, screen_space_quad_vertices.size(), 1, 0, 0);
            },
            .should_run_predicate = [&] { return use_ssao; }
        });

        const auto main_node = render_graph.add_node(RenderNodeGraphics {
            .name = "main",
            .bound_resources = {uniform_buffer, ssao_texture, base_color_texture, normal_texture, orm_texture, skybox_texture},
            .color_targets = {base_pass_texture},
            .depth_target = FINAL_IMAGE_HANDLE,
            .body = [=](RenderPassContext &ctx) {
                ctx.bind_pipeline(main_pipeline);
                ctx.bind_resources({uniform_buffer, ssao_texture, base_color_texture, normal_texture, orm_texture});
                ctx.draw_model(scene_model);

                ctx.bind_pipeline(skybox_pipeline);
                ctx.bind_resources({uniform_buffer, skybox_texture});
                ctx.draw(skybox_vert_buf, skybox_vertices.size(), 1, 0, 0);
            },
        });

        const auto post_processing_nodes = render_graph.add_nodes_sequential({
            RenderNodeCompute {
                .name = "blur-x",
                .bound_read_resources = {base_pass_texture},
                .bound_write_resources = {post_blur_x_texture},
                .body = [=](ComputePassContext &ctx) {
                    ctx.bind_pipeline(blur_x_pipeline);
                    ctx.bind_resources({FINAL_IMAGE_HANDLE, post_blur_x_texture});
                    ctx.dispatch(window_size.x / 32, window_size.y / 32, 1);
                },
                .should_run_predicate = [&] { return do_blur; },
                .explicit_dependencies = { main_node },
            },
            RenderNodeCompute {
                .name = "blur-y",
                .bound_read_resources = {post_blur_x_texture},
                .bound_write_resources = {post_blur_y_texture},
                .body = [=](ComputePassContext &ctx) {
                    ctx.bind_pipeline(blur_y_pipeline);
                    ctx.bind_resources({post_blur_x_texture, FINAL_IMAGE_HANDLE});
                    ctx.dispatch(window_size.x / 32, window_size.y / 32, 1);
                },
                .should_run_predicate = [&] { return do_blur; }
            }
        });

        // todo - make it work!
        render_graph.resource_alias_hint(post_blur_y_texture, post_gui_texture);

        render_graph.add_node(RenderNodeGraphics {
            .name = "gui",
            .bound_resources = {},
            .color_targets = {post_gui_texture},
            .body = [=](const RenderPassContext &ctx) {
                renderer.get_gui_renderer().begin_rendering();
                render_gui_section(curr_delta_time);
                renderer.get_gui_renderer().end_rendering(ctx.get_raw_cmd_buffer());
            },
            .should_run_predicate = [&] { return is_gui_enabled; },
            .explicit_dependencies = { main_node, post_processing_nodes.back() },
        });

        render_graph.add_node(RenderNodeGraphics {
            .name = "final",
            .bound_resources = {post_gui_texture},
            .color_targets = {FINAL_IMAGE_HANDLE},
            .body = [=](RenderPassContext &ctx) {
                ctx.bind_pipeline(final_pipeline);
                ctx.bind_resources({post_gui_texture});
                ctx.draw(ss_quad_vert_buf, screen_space_quad_vertices.size(), 1, 0, 0);
            },
        });

        renderer.register_render_graph(render_graph);
    }

    void update_graphics_uniform_buffer(Buffer &buffer) const {
        const glm::mat4 model = glm::translate(glm::identity<glm::mat4>(), model_translate)
                                * glm::mat4_cast(model_rotation)
                                * glm::scale(glm::identity<glm::mat4>(), glm::vec3(model_scale));
        const glm::mat4 view = camera->get_view_matrix();
        const glm::mat4 proj = camera->get_projection_matrix();

        glm::ivec2 window_size{};
        glfwGetWindowSize(window, &window_size.x, &window_size.y);

        const auto [z_near, z_far] = camera->get_clipping_planes();

        static const glm::mat4 cubemap_face_projection = glm::perspective(glm::radians(90.0f), 1.0f, 0.1f, 10.0f);

        GraphicsUBO graphics_ubo{
            .window = {
                .window_width = static_cast<uint32_t>(window_size.x),
                .window_height = static_cast<uint32_t>(window_size.y),
            },
            .matrices = {
                .model = model,
                .view = view,
                .proj = proj,
                .view_inverse = glm::inverse(view),
                .proj_inverse = glm::inverse(proj),
                .vp_inverse = glm::inverse(proj * view),
                .static_view = camera->get_static_view_matrix(),
                .cubemap_capture_proj = cubemap_face_projection
            },
            .misc = {
                .debug_number = debug_number,
                .z_near = z_near,
                .z_far = z_far,
                .use_ssao = use_ssao ? 1u : 0,
                .light_intensity = light_intensity,
                .light_dir = glm::vec3(mat4_cast(light_direction) * glm::vec4(-1, 0, 0, 0)),
                .light_color = light_color,
                .camera_pos = camera->get_pos(),
            }
        };

        static const array cubemap_face_views{
            glm::lookAt(glm::vec3(0), glm::vec3(-1, 0, 0), glm::vec3(0, 1, 0)),
            glm::lookAt(glm::vec3(0), glm::vec3(1, 0, 0), glm::vec3(0, 1, 0)),
            glm::lookAt(glm::vec3(0), glm::vec3(0, 1, 0), glm::vec3(0, 0, -1)),
            glm::lookAt(glm::vec3(0), glm::vec3(0, -1, 0), glm::vec3(0, 0, 1)),
            glm::lookAt(glm::vec3(0), glm::vec3(0, 0, 1), glm::vec3(0, 1, 0)),
            glm::lookAt(glm::vec3(0), glm::vec3(0, 0, -1), glm::vec3(0, 1, 0))
        };

        for (size_t i = 0; i < 6; i++) {
            graphics_ubo.matrices.cubemap_capture_views[i] = cubemap_face_views[i];
        }

        std::memcpy(buffer.map(), &graphics_ubo, sizeof(graphics_ubo));
    }

    void bind_key_actions() {
        input_manager->bind_callback(ZRX_GLFW_KEY_GRAVE_ACCENT, EActivationType::PRESS_ONCE, [&](const float delta_time) {
            (void) delta_time;
            is_gui_enabled = !is_gui_enabled;
        });

        input_manager->bind_callback(ZRX_GLFW_KEY_F1, EActivationType::PRESS_ONCE, [&](const float delta_time) {
            (void) delta_time;
            do_blur = !do_blur;
        });
    }

    void bind_mouse_drag_actions() {
        input_manager->bind_mouse_drag_callback(ZRX_GLFW_MOUSE_BUTTON_RIGHT, [&](const double dx, const double dy) {
            static constexpr float speed = 0.002;
            const float camera_distance  = glm::length(camera->get_pos());

            const auto view_vectors = camera->get_view_vectors();

            model_translate += camera_distance * speed * view_vectors.right * static_cast<float>(dx);
            model_translate -= camera_distance * speed * view_vectors.up * static_cast<float>(dy);
        });
    }

    // ========================== gui ==========================

    void render_gui_section(const float delta_time) {
        static float fps = 1 / delta_time;

        constexpr float smoothing = 0.95f;
        fps                       = fps * smoothing + (1 / delta_time) * (1.0f - smoothing);

        constexpr auto section_flags = ImGuiTreeNodeFlags_DefaultOpen;

        if (ImGui::CollapsingHeader("Engine ", section_flags)) {
            ImGui::Text("FPS: %.2f", fps);

            ImGui::Checkbox("Debug quad", &show_debug_quad);
            ImGui::Separator();

            if (ImGui::Button("Reload shaders")) {
                // renderer.reload_shaders();
            }
            ImGui::Separator();

            render_load_model_popup();

            if (!curr_error_message.empty()) {
                ImGui::OpenPopup("Model load error");
            }

            render_model_load_error_popup();
        }

        if (ImGui::CollapsingHeader("Environment ", section_flags)) {
            render_tex_load_button("Choose environment map...", FileType::ENVMAP_HDR, {".hdr"});

            file_browser.Display();
        }

        if (ImGui::CollapsingHeader("Model ", section_flags)) {
            if (ImGui::Button("Load model...")) {
                ImGui::OpenPopup("Load model");
            }

            ImGui::Separator();

            ImGui::DragFloat("Model scale", &model_scale, 0.01, 0, numeric_limits<float>::max());

            ImGui::gizmo3D("Model rotation", model_rotation, 160);

            if (ImGui::Button("Reset scale")) { model_scale = 1; }
            ImGui::SameLine();
            if (ImGui::Button("Reset rotation")) { model_rotation = {1, 0, 0, 0}; }
            ImGui::SameLine();
            if (ImGui::Button("Reset position")) { model_translate = {0, 0, 0}; }
        }

        if (ImGui::CollapsingHeader("Advanced ", section_flags)) {
            ImGui::Checkbox("SSAO", &use_ssao);

#ifndef NDEBUG
            ImGui::Separator();
            ImGui::DragFloat("Debug number", &debug_number, 0.01, 0, numeric_limits<float>::max());
#endif
        }

        if (ImGui::CollapsingHeader("Lighting ", section_flags)) {
            ImGui::SliderFloat("Light intensity", &light_intensity, 0.0f, 100.0f, "%.2f");
            ImGui::ColorEdit3("Light color", &light_color.x);
            ImGui::gizmo3D("Light direction", light_direction, 160, imguiGizmo::modeDirection);
        }

        camera->render_gui_section();
        renderer.render_gui_section();
    }

    void render_tex_load_button(const string &label, const FileType file_type,
                                const vector<string> &type_filters) {
        if (ImGui::Button(label.c_str(), ImVec2(180, 0))) {
            current_type_being_chosen = file_type;
            file_browser.SetTypeFilters(type_filters);
            file_browser.Open();
        }

        if (chosen_paths.contains(file_type)) {
            ImGui::SameLine();
            ImGui::Text(chosen_paths.at(file_type).filename().string().c_str());
        }
    }

    void render_load_model_popup() {
        constexpr auto combo_flags = ImGuiComboFlags_WidthFitPreview;

        if (ImGui::BeginPopupModal("Load model", nullptr, ImGuiWindowFlags_AlwaysAutoResize)) {
            ImGui::Text("Load scheme:");

            if (ImGui::BeginCombo("##scheme", file_load_schemes[load_scheme_idx].name.c_str(),
                                  combo_flags)) {
                for (uint32_t i = 0; i < file_load_schemes.size(); i++) {
                    const bool is_selected = load_scheme_idx == i;

                    if (ImGui::Selectable(file_load_schemes[i].name.c_str(), is_selected)) {
                        load_scheme_idx = i;
                    }

                    if (is_selected) {
                        ImGui::SetItemDefaultFocus();
                    }
                }
                ImGui::EndCombo();
            }

            ImGui::Separator();

            for (const auto &type: file_load_schemes[load_scheme_idx].requirements) {
                render_tex_load_button(
                    get_file_type_load_label(type),
                    type,
                    get_file_type_extensions(type)
                );
            }

            ImGui::Separator();

            const bool can_submit = std::ranges::all_of(
                file_load_schemes[load_scheme_idx].requirements,
                [&](const auto &t) {
                    return is_file_type_optional(t) || chosen_paths.contains(t);
                }
            );

            if (!can_submit) {
                ImGui::BeginDisabled();
            }

            if (ImGui::Button("OK", ImVec2(120, 0))) {
                load_model();
                chosen_paths.clear();
                ImGui::CloseCurrentPopup();
            }

            if (!can_submit) {
                ImGui::EndDisabled();
            }

            ImGui::SameLine();

            if (ImGui::Button("Cancel", ImVec2(120, 0))) {
                chosen_paths.clear();
                ImGui::CloseCurrentPopup();
            }

            file_browser.Display();

            ImGui::EndPopup();
        }
    }

    void load_model() {
        const auto &reqs = file_load_schemes[load_scheme_idx].requirements;

        try {
            // if (reqs.contains(FileType::BASE_COLOR_PNG)) {
            //     renderer.load_model(chosen_paths.at(FileType::MODEL));
            //     renderer.load_base_color_texture(chosen_paths.at(FileType::BASE_COLOR_PNG));
            // } else {
            //     renderer.load_model_with_materials(chosen_paths.at(FileType::MODEL));
            // }
            //
            // if (reqs.contains(FileType::NORMAL_PNG)) {
            //     renderer.load_normal_map(chosen_paths.at(FileType::NORMAL_PNG));
            // }
            //
            // if (reqs.contains(FileType::ORM_PNG)) {
            //     renderer.load_orm_map(chosen_paths.at(FileType::ORM_PNG));
            // } else if (reqs.contains(FileType::RMA_PNG)) {
            //     renderer.load_rma_map(chosen_paths.at(FileType::RMA_PNG));
            // } else if (reqs.contains(FileType::ROUGHNESS_PNG)) {
            //     const auto roughness_path = chosen_paths.at(FileType::ROUGHNESS_PNG);
            //     const auto ao_path        = chosen_paths.contains(FileType::AO_PNG)
            //                                     ? chosen_paths.at(FileType::AO_PNG)
            //                                     : "";
            //     const auto metallic_path = chosen_paths.contains(FileType::METALLIC_PNG)
            //                                    ? chosen_paths.at(FileType::METALLIC_PNG)
            //                                    : "";
            //
            //     renderer.load_orm_map(ao_path, roughness_path, metallic_path);
            // }
        } catch (std::exception &e) {
            curr_error_message = e.what();
        }
    }

    void render_model_load_error_popup() {
        if (ImGui::BeginPopupModal("Model load error", nullptr, ImGuiWindowFlags_AlwaysAutoResize)) {
            ImGui::Text("An error occurred while loading the model:");
            ImGui::Text(curr_error_message.c_str());

            ImGui::Separator();

            if (ImGui::Button("OK", ImVec2(120, 0))) {
                ImGui::CloseCurrentPopup();
                curr_error_message = "";
            }

            ImGui::EndPopup();
        }
    }
};
}

static void show_error_box(const string &message) {
    // MessageBox(
    //     nullptr,
    //     static_cast<LPCSTR>(message.c_str()),
    //     static_cast<LPCSTR>("Error"),
    //     MB_OK
    // );
}

int main() {
    if (!glfwInit()) {
        show_error_box("Fatal error: GLFW initialization failed.");
        return 1; //EXIT_FAILURE;
    }

#ifdef NDEBUG
    try {
        zrx::Engine engine;
        engine.run();
    } catch (std::exception &e) {
        show_error_box(string("Fatal error: ") + e.what());
        glfw_terminate();
        return EXIT_FAILURE;
    }
#else
    zrx::Engine engine;
    engine.run();
#endif

    glfwTerminate();

    return 0; // EXIT_SUCCESS;
}
