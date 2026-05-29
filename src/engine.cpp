module;

module Cinder.Engine;

import imgui;
import imguizmo_quat;
import imfilebrowser;
import std;
import glfw;
import glm;
import vulkan;

import Cinder.Utils;
import Cinder.Render;
import Cinder.Render.Graph;
import Cinder.Render.Vulkan;
import Cinder.Render.Gui;
import Cinder.Render.Mesh;
import Cinder.Globals;

namespace zrx {
namespace glsl {
    #include "render/glsl_to_cpp.inl"
    #include "shaders/utils/ubo.glsl"
    #include "shaders/utils/material.glsl"
}

struct GraphicsUBO {
    GLSL_ALIGN16 glsl::WindowRes window{};
    GLSL_ALIGN16 glsl::Matrices matrices{};
    GLSL_ALIGN16 glsl::LightData light{};
    GLSL_ALIGN16 glsl::MiscData misc{};
};

struct MaterialsUBO {
    GLSL_ALIGN16 glsl::Material mats[MAX_MATERIAL_COUNT];
};

Engine::Engine() {
    window = renderer.get_window();
    camera = make_unique<Camera>(window);

    input_manager = std::make_unique<InputManager>(window);
    bind_key_actions();
    bind_mouse_drag_actions();

    register_render_graph_resources();
    build_render_graph();
}

void Engine::run() {
    while (!glfwWindowShouldClose(window)) {
        tick();
    }

    renderer.wait_idle();
}

void Engine::tick() {
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
}

void Engine::register_render_graph_resources() {
    auto& resource_manager = renderer.get_resource_manager();
    auto& rr = render_resources;

    // ================== models and vertex buffers ==================

    rr[Model_Scene] = resource_manager.add_from_desc(ModelResourceDesc{
        .name = "scene-model",
        .path = "../assets/example models/Sponza/Sponza.gltf",
        .has_materials = true,
    });

    rr[VB_Skybox] = resource_manager.add_from_desc(VertexBufferResourceDesc{
        .name = "skybox-vb",
        .size = skybox_vertices.size() * sizeof(SkyboxVertex),
        .data = skybox_vertices.data()
    });

    rr[VB_ScreenSpaceQuad] = resource_manager.add_from_desc(VertexBufferResourceDesc{
        .name = "ss-quad-vb",
        .size = screen_space_quad_vertices.size() * sizeof(ScreenSpaceQuadVertex),
        .data = screen_space_quad_vertices.data()
    });

    // ================== uniform buffers ==================

    rr[UBO_General] = resource_manager.add_from_desc(UniformBufferResourceDesc{
        .name = "general-ubo",
        .size = sizeof(GraphicsUBO)
    });

    renderer.add_frame_begin_action([&](const FrameBeginActionContext &fba_ctx) {
        Buffer& buffer = fba_ctx.resource_manager.get().get<Buffer>(rr[UBO_General]);
        update_graphics_uniform_buffer(buffer);
    });

    rr[UBO_Materials] = resource_manager.add_from_desc(UniformBufferResourceDesc{
        .name = "material-ubo",
        .size = sizeof(MaterialsUBO)
    });

    renderer.add_frame_begin_action([&](const FrameBeginActionContext &fba_ctx) {
        static bool has_been_done = false;
        if (!has_been_done) {
            auto& resource_manager = fba_ctx.resource_manager.get();
            Buffer& material_ubo_buffer = resource_manager.get<Buffer>(rr[UBO_Materials]);
            update_materials_uniform_buffer(material_ubo_buffer, rr[Model_Scene], resource_manager);
            has_been_done = true;
        }
    });

    // ================== external textures ==================

    rr[Tex_Envmap] = resource_manager.add_from_desc(ExternalTextureResourceDesc{
        .name = "envmap-texture",
        .paths = {"../assets/envmaps/vienna.hdr"},
        .format = vk::Format::eR32G32B32A32Sfloat,
        .flags = ImageFlags::HDR
    });

    // ================== render target textures ==================

    constexpr auto skybox_tex_format = vk::Format::eR8G8B8A8Srgb;
    rr[Tex_Skybox] = resource_manager.add_from_desc(TargetTextureResourceDesc{
        .name = "skybox-texture",
        .format = skybox_tex_format,
        .extent = {2048, 2048},
        .flags = ImageFlags::CUBEMAP | ImageFlags::NO_MIPMAPS
    });

    constexpr auto g_buffer_color_format = vk::Format::eR16G16B16A16Sfloat;
    rr[Tex_GNormal] = resource_manager.add_from_desc(TargetTextureResourceDesc{
        .name = "g-buffer-normal",
        .format = g_buffer_color_format,
    });

    rr[Tex_GPos] = resource_manager.add_from_desc(TargetTextureResourceDesc{
        .name = "g-buffer-pos",
        .format = g_buffer_color_format,
    });

    constexpr auto g_buffer_depth_format = vk::Format::eD32Sfloat;
    rr[Tex_GDepth] = resource_manager.add_from_desc(TargetTextureResourceDesc{
        .name = "g-buffer-depth",
        .format = g_buffer_depth_format,
        .flags = ImageFlags::NO_MIPMAPS,
    });

    constexpr auto ssao_tex_format = vk::Format::eR8G8B8A8Unorm;
    rr[Tex_SSAO] = resource_manager.add_from_desc(TargetTextureResourceDesc{
        .name = "ssao-texture",
        .format = ssao_tex_format,
    });

    constexpr auto shadowmap_tex_format = vk::Format::eD32Sfloat;
    rr[Tex_Shadowmap] = resource_manager.add_from_desc(TargetTextureResourceDesc{
        .name = "shadowmap-texture",
        .format = shadowmap_tex_format,
        .extent = {2048, 2048},
        .overrides = {
            .mag_filter = vk::Filter::eNearest,
            .min_filter = vk::Filter::eNearest,
        },
        .flags = ImageFlags::NO_MIPMAPS,
    });

    constexpr auto final_no_gamma_format = vk::Format::eR8G8B8A8Unorm;
    rr[Tex_BasePass] = resource_manager.add_from_desc(TargetTextureResourceDesc{
        .name = "base-pass-texture",
        .format = final_no_gamma_format,
        .flags = ImageFlags::NO_MIPMAPS
    });
    rr[Tex_PostBlurX] = resource_manager.add_from_desc(TargetTextureResourceDesc{
        .name = "post-blur-x-texture",
        .format = final_no_gamma_format,
        .flags = ImageFlags::NO_MIPMAPS
    });
    rr[Tex_PostBlurY] = resource_manager.add_from_desc(TargetTextureResourceDesc{
        .name = "post-blur-y-texture",
        .format = final_no_gamma_format,
        .flags = ImageFlags::NO_MIPMAPS
    });
    rr[Tex_PostGui] = resource_manager.add_from_desc(TargetTextureResourceDesc{
        .name = "post-gui-texture",
        .format = final_no_gamma_format,
        .flags = ImageFlags::NO_MIPMAPS
    });

    // ================== shaders ==================

    renderer.set_shader_base_path("../shaders/obj/");

    rr[Pipe_SsQuad] = resource_manager.add_from_desc(GraphicsPipelineResourceDesc{
        .vertex_path = "ss-quad-depth-vert.spv",
        .fragment_path = "ss-quad-depth-frag.spv",
        .vertex_bindings = ScreenSpaceQuadVertex::get_binding_descriptions(),
        .vertex_attributes = ScreenSpaceQuadVertex::get_attribute_descriptions(),
        .color_formats = { FINAL_FORMAT },
    });

    rr[Pipe_CubeCapture] = resource_manager.add_from_desc(GraphicsPipelineResourceDesc{
        .vertex_path = "sphere-cube-vert.spv",
        .fragment_path = "sphere-cube-frag.spv",
        .vertex_bindings = SkyboxVertex::get_binding_descriptions(),
        .vertex_attributes = SkyboxVertex::get_attribute_descriptions(),
        .color_formats = {skybox_tex_format},
        .custom_properties = {
            .multiview_count = 6
        }
    });

    rr[Pipe_Shadowmap] = resource_manager.add_from_desc(GraphicsPipelineResourceDesc{
        .vertex_path = "shadowmap-vert.spv",
        .fragment_path = "shadowmap-frag.spv",
        .vertex_bindings = ModelVertex::get_binding_descriptions(),
        .vertex_attributes = ModelVertex::get_attribute_descriptions(),
        .depth_format = shadowmap_tex_format
    });

    rr[Pipe_Prepass] = resource_manager.add_from_desc(GraphicsPipelineResourceDesc{
        .vertex_path = "prepass-vert.spv",
        .fragment_path = "prepass-frag.spv",
        .vertex_bindings = ModelVertex::get_binding_descriptions(),
        .vertex_attributes = ModelVertex::get_attribute_descriptions(),
        .color_formats = {g_buffer_color_format, g_buffer_color_format},
        .depth_format = g_buffer_depth_format
    });

    rr[Pipe_SSAO] = resource_manager.add_from_desc(GraphicsPipelineResourceDesc{
        .vertex_path = "ssao-vert.spv",
        .fragment_path = "ssao-frag.spv",
        .vertex_bindings = ScreenSpaceQuadVertex::get_binding_descriptions(),
        .vertex_attributes = ScreenSpaceQuadVertex::get_attribute_descriptions(),
        .color_formats = {ssao_tex_format}
    });

    rr[Pipe_Skybox] = resource_manager.add_from_desc(GraphicsPipelineResourceDesc{
        .vertex_path = "skybox-vert.spv",
        .fragment_path = "skybox-frag.spv",
        .vertex_bindings = SkyboxVertex::get_binding_descriptions(),
        .vertex_attributes = SkyboxVertex::get_attribute_descriptions(),
        .color_formats = {final_no_gamma_format},
        .depth_format = FINAL_FORMAT,
        .custom_properties = {
            .depth_compare_op = vk::CompareOp::eLessOrEqual,
        }
    });

    rr[Pipe_Main] = resource_manager.add_from_desc(GraphicsPipelineResourceDesc{
        .vertex_path = "main-vert.spv",
        .fragment_path = "main-frag.spv",
        .vertex_bindings = ModelVertex::get_binding_descriptions(),
        .vertex_attributes = ModelVertex::get_attribute_descriptions(),
        .color_formats = {final_no_gamma_format},
        .depth_format = FINAL_FORMAT
    });

    rr[Pipe_BlurX] = resource_manager.add_from_desc(ComputePipelineResourceDesc{
        .path = "blur-x-comp.spv",
    });

    rr[Pipe_BlurY] = resource_manager.add_from_desc(ComputePipelineResourceDesc{
        .path = "blur-y-comp.spv",
    });

    rr[Pipe_Final] = resource_manager.add_from_desc(GraphicsPipelineResourceDesc{
        .vertex_path = "ss-quad-vert.spv",
        .fragment_path = "ss-quad-frag.spv",
        .vertex_bindings = ScreenSpaceQuadVertex::get_binding_descriptions(),
        .vertex_attributes = ScreenSpaceQuadVertex::get_attribute_descriptions(),
        .color_formats = {FINAL_FORMAT},
    });
}

void Engine::build_render_graph() {
    RenderGraph render_graph;
    const auto& rr = render_resources;

    // ================== nodes ==================

    if (should_capture_skybox) {
        render_graph.add_node(RenderNodeGraphics {
            .name = "cubemap-capture",
            .bound_resources = {rr[UBO_General], rr[Tex_Envmap]},
            .color_targets = {rr[Tex_Skybox]},
            .body = [&](RenderPassContext &ctx) {
                ctx.bind_pipeline(rr[Pipe_CubeCapture]);
                ctx.bind_resources({rr[UBO_General], rr[Tex_Envmap]});
                ctx.draw(rr[VB_Skybox], skybox_vertices.size(), 1, 0, 0);
            },
            .custom_properties = RenderNodeGraphics::CustomProperties {
                .multiview_count = 6
            }
        });
    }

    render_graph.add_node(RenderNodeGraphics {
        .name = "shadowmap",
        .bound_resources = {rr[UBO_General]},
        .depth_target = rr[Tex_Shadowmap],
        .body = [&](RenderPassContext &ctx) {
            ctx.bind_pipeline(rr[Pipe_Shadowmap]);
            ctx.bind_resources({rr[UBO_General]});
            ctx.draw_model(rr[Model_Scene]);
        },
    });

    if (use_ssao) {
        render_graph.add_node(RenderNodeGraphics {
            .name = "prepass",
            .bound_resources = {rr[UBO_General]},
            .color_targets = {rr[Tex_GNormal], rr[Tex_GPos]},
            .depth_target = rr[Tex_GDepth],
            .body = [&](RenderPassContext &ctx) {
                ctx.bind_pipeline(rr[Pipe_Prepass]);
                ctx.bind_resources({rr[UBO_General]});
                ctx.draw_model(rr[Model_Scene]);
            },
        });

        render_graph.add_node(RenderNodeGraphics {
            .name = "ssao",
            .bound_resources = {rr[UBO_General], rr[Tex_GDepth], rr[Tex_GNormal], rr[Tex_GPos]},
            .color_targets = {rr[Tex_SSAO]},
            .body = [&](RenderPassContext &ctx) {
                ctx.bind_pipeline(rr[Pipe_SSAO]);
                ctx.bind_resources({rr[UBO_General], rr[Tex_GDepth], rr[Tex_GNormal], rr[Tex_GPos]});
                ctx.draw(rr[VB_ScreenSpaceQuad], screen_space_quad_vertices.size(), 1, 0, 0);
            },
        });
    }

    render_graph.add_node(RenderNodeGraphics {
        .name = "main",
        .bound_resources = {rr[UBO_General], rr[Tex_SSAO], rr[Tex_Skybox], rr[Tex_Shadowmap]},
        .color_targets = {rr[Tex_BasePass]},
        .depth_target = FINAL_IMAGE_HANDLE,
        .body = [&](RenderPassContext &ctx) {
            ctx.bind_pipeline(rr[Pipe_Main]);
            ctx.bind_resources({rr[UBO_General], rr[Tex_SSAO], rr[Tex_Shadowmap], rr[UBO_Materials], CURRENT_MATERIAL_HANDLE});
            ctx.draw_model(rr[Model_Scene]);

            ctx.bind_pipeline(rr[Pipe_Skybox]);
            ctx.bind_resources({rr[UBO_General], rr[Tex_Skybox]});
            ctx.draw(rr[VB_Skybox], skybox_vertices.size(), 1, 0, 0);
        },
    });

    optional<RenderNodeHandle> final_handle;

    if (do_blur) {
        const auto post_processing_nodes = render_graph.add_nodes_sequential({
            RenderNodeCompute {
                .name = "blur-x",
                .bound_read_resources = {rr[Tex_BasePass]},
                .bound_write_resources = {rr[Tex_PostBlurX]},
                .body = [&](ComputePassContext &ctx) {
                    ctx.bind_pipeline(rr[Pipe_BlurX]);
                    ctx.bind_resources({rr[Tex_BasePass], rr[Tex_PostBlurX]});
                    ctx.dispatch(std::ceil(window_size.x / 32.0f), std::ceil(window_size.y / 32.0f), 1);
                },
            },
            RenderNodeCompute {
                .name = "blur-y",
                .bound_read_resources = {rr[Tex_PostBlurX]},
                .bound_write_resources = {rr[Tex_PostBlurY]},
                .body = [&](ComputePassContext &ctx) {
                    ctx.bind_pipeline(rr[Pipe_BlurY]);
                    ctx.bind_resources({rr[Tex_PostBlurX], rr[Tex_PostBlurY]});
                    ctx.dispatch(std::ceil(window_size.x / 32.0f), std::ceil(window_size.y / 32.0f), 1);
                },
            }
        });

        final_handle = render_graph.add_node(RenderNodeGraphics {
            .name = "final-blurred",
            .bound_resources = {rr[Tex_PostBlurY]},
            .color_targets = {FINAL_IMAGE_HANDLE},
            .body = [&](RenderPassContext &ctx) {
                ctx.bind_pipeline(rr[Pipe_Final]);
                ctx.bind_resources({rr[UBO_General], rr[Tex_PostBlurY]});
                ctx.draw(rr[VB_ScreenSpaceQuad], screen_space_quad_vertices.size(), 1, 0, 0);
            },
        });

    } else {
        final_handle = render_graph.add_node(RenderNodeGraphics {
            .name = "final-unblurred",
            .bound_resources = {rr[Tex_BasePass]},
            .color_targets = {FINAL_IMAGE_HANDLE},
            .body = [&](RenderPassContext &ctx) {
                if (show_debug_quad) {
                    ctx.bind_pipeline(rr[Pipe_SsQuad]);
                    ctx.bind_resources({rr[UBO_General], rr[Tex_Shadowmap]});
                } else {
                    ctx.bind_pipeline(rr[Pipe_Final]);
                    ctx.bind_resources({rr[UBO_General], rr[Tex_BasePass]});
                }
                ctx.draw(rr[VB_ScreenSpaceQuad], screen_space_quad_vertices.size(), 1, 0, 0);
            },
        });
    }

    if (is_gui_enabled) {
        render_graph.add_node(RenderNodeGraphics {
            .name = "gui",
            .bound_resources = {},
            .color_targets = {FINAL_IMAGE_HANDLE},
            .body = [&](const RenderPassContext &ctx) {
                renderer.get_gui_renderer().begin_rendering();
                render_gui_section(curr_delta_time);
                renderer.get_gui_renderer().end_rendering(ctx.get_raw_cmd_buffer());
            },
            .explicit_dependencies = { *final_handle },
        });
    }

    renderer.register_render_graph(render_graph);
}

void Engine::update_graphics_uniform_buffer(const Buffer &buffer) const {
    const glm::mat4 model = glm::gtc::translate(glm::gtc::identity<glm::mat4>(), model_translate)
                            * glm::gtc::mat4_cast(model_rotation)
                            * glm::gtc::scale(glm::gtc::identity<glm::mat4>(), glm::vec3(model_scale));
    const glm::mat4 view = camera->get_view_matrix();
    const glm::mat4 proj = camera->get_projection_matrix();

    glm::ivec2 window_size{};
    glfwGetWindowSize(window, &window_size.x, &window_size.y);

    const auto [z_near, z_far] = camera->get_clipping_planes();

    static const glm::mat4 cubemap_face_projection = glm::gtc::perspective(glm::radians(90.0f), 1.0f, 0.1f, 10.0f);

    const auto light_direction_vec = glm::vec3(glm::gtc::mat4_cast(light_direction) * glm::vec4(1)) * 30.0f;
    const auto light_view = glm::gtc::lookAt(light_direction_vec, glm::vec3(0), glm::vec3(0, 1, 0));
    const auto light_proj = glm::gtc::ortho(
        shadow_map_config.left, shadow_map_config.right,
        shadow_map_config.bottom, shadow_map_config.top,
        shadow_map_config.z_near, shadow_map_config.z_far
    );

    GraphicsUBO graphics_ubo{
        .window = {
            .width = static_cast<uint32_t>(window_size.x),
            .height = static_cast<uint32_t>(window_size.y),
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
        .light = {
            .direction = light_direction_vec,
            .color = light_color,
            .intensity = light_intensity,
            .proj_x_view = light_proj * light_view,
        },
        .misc = {
            .debug_number = debug_number,
            .z_near = z_near,
            .z_far = z_far,
            .use_ssao = use_ssao ? 1u : 0,
            .camera_pos = camera->get_pos(),
            .bias_weight_1 = shadow_map_config.bias_weight_1,
            .bias_weight_2 = shadow_map_config.bias_weight_2,
        }
    };

    static const array cubemap_face_views{
        glm::gtc::lookAt(glm::vec3(0), glm::vec3(-1, 0, 0), glm::vec3(0, 1, 0)),
        glm::gtc::lookAt(glm::vec3(0), glm::vec3(1, 0, 0),  glm::vec3(0, 1, 0)),
        glm::gtc::lookAt(glm::vec3(0), glm::vec3(0, 1, 0),  glm::vec3(0, 0, -1)),
        glm::gtc::lookAt(glm::vec3(0), glm::vec3(0, -1, 0), glm::vec3(0, 0, 1)),
        glm::gtc::lookAt(glm::vec3(0), glm::vec3(0, 0, 1),  glm::vec3(0, 1, 0)),
        glm::gtc::lookAt(glm::vec3(0), glm::vec3(0, 0, -1), glm::vec3(0, 1, 0))
    };

    for (size_t i = 0; i < 6; i++) {
        graphics_ubo.matrices.cubemap_capture_views[i] = cubemap_face_views[i];
    }

    buffer.copy_from_ptr(&graphics_ubo, sizeof(graphics_ubo));
}

void Engine::update_materials_uniform_buffer(const Buffer &buffer, const ResourceHandle model_handle, const ResourceManager& resource_manager) const {
    MaterialsUBO materials_ubo {};
    std::memset(&materials_ubo, 0, sizeof(materials_ubo));

    const auto& material_handles = resource_manager.get_model_material_handles(model_handle);

    for (const auto material_handle : material_handles) {
        const auto& texture_handles = resource_manager.get_material_tex_handles(material_handle);

        materials_ubo.mats[material_handle] = glsl::Material {
            .base_color = static_cast<uint32_t>(texture_handles.base_color),
            .normal     = static_cast<uint32_t>(texture_handles.normal),
            .orm        = static_cast<uint32_t>(texture_handles.orm),
        };
    }

    buffer.copy_from_ptr(&materials_ubo, sizeof(materials_ubo));
}

void Engine::bind_key_actions() {
    input_manager->bind_callback(glfw::Key::KEY_GRAVE_ACCENT, EActivationType::PRESS_ONCE, [&](const float delta_time) {
        (void) delta_time;
        is_gui_enabled = !is_gui_enabled;
    });

    input_manager->bind_callback(glfw::Key::KEY_F1, EActivationType::PRESS_ONCE, [&](const float delta_time) {
        (void) delta_time;
        do_blur = !do_blur;
    });
}

void Engine::bind_mouse_drag_actions() {
    input_manager->bind_mouse_drag_callback(glfw::MouseButton::MOUSE_BUTTON_RIGHT, [&](const double dx, const double dy) {
        static constexpr float speed = 0.002;
        const float camera_distance  = glm::length(camera->get_pos());

        const auto view_vectors = camera->get_view_vectors();

        model_translate += camera_distance * speed * view_vectors.right * static_cast<float>(dx);
        model_translate -= camera_distance * speed * view_vectors.up * static_cast<float>(dy);
    });
}

// ========================== gui ==========================

void Engine::render_gui_section(const float delta_time) {
    static constexpr float smoothing = 0.95f;
    static float fps = 1 / delta_time;

    fps = fps * smoothing + (1 / delta_time) * (1.0f - smoothing);

    constexpr auto section_flags = ImGuiTreeNodeFlags_DefaultOpen;

    if (ImGui::CollapsingHeader("Engine ", section_flags)) {
        ImGui::Text("FPS: %.2f", fps);

        ImGui::Checkbox("Debug quad", &show_debug_quad);
        ImGui::Separator();

        if (ImGui::Button("Reload shaders")) {
            renderer.reload_all_pipelines();
        }
    }

    if (ImGui::CollapsingHeader("Model ", section_flags)) {
        if (ImGui::Button("Load model...")) {
            ImGui::OpenPopup("Load model");
        }

        ImGui::Separator();

        ImGui::DragFloat("Model scale", &model_scale, 0.01, 0, numeric_limits<float>::max());

        //ImGui::gizmo3D("Model rotation", model_rotation, 160);

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
        ImGui::gizmo3D("Light direction", light_direction, 160.0f, imguiGizmo::modeDirection);
    }

    if (ImGui::CollapsingHeader("Shadowmap ", section_flags)) {
        ImGui::PushItemWidth(300.0f);

        ImGui::SliderFloat("left",   &shadow_map_config.left,   -100.0f, 100.0f, "%.2f");
        ImGui::SameLine();
        ImGui::SliderFloat("right",  &shadow_map_config.right,  -100.0f, 100.0f, "%.2f");

        ImGui::SliderFloat("bottom", &shadow_map_config.bottom, -100.0f, 100.0f, "%.2f");
        ImGui::SameLine();
        ImGui::SliderFloat("top",    &shadow_map_config.top,    -100.0f, 100.0f, "%.2f");

        ImGui::SliderFloat("z_near", &shadow_map_config.z_near,    0.0f, 100.0f, "%.2f");
        ImGui::SameLine();
        ImGui::SliderFloat("z_far",  &shadow_map_config.z_far,     0.0f, 100.0f, "%.2f");

        ImGui::SliderFloat("bias_weight_1", &shadow_map_config.bias_weight_1, 0.0f, 0.01f, "%.5f");
        ImGui::SameLine();
        ImGui::SliderFloat("bias_weight_2", &shadow_map_config.bias_weight_2, 0.0f, 0.01f, "%.5f");

        ImGui::PopItemWidth();
    }

    camera->render_gui_section();
    renderer.render_gui_section();
    Logger::render_gui_section();
}
} // zrx
