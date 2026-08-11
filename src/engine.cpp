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
}

struct GraphicsUBO {
    GLSL_ALIGN16 glsl::WindowRes window{};
    GLSL_ALIGN16 glsl::Matrices matrices{};
    GLSL_ALIGN16 glsl::LightData light{};
    GLSL_ALIGN16 glsl::MiscData misc{};
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

    const auto old_render_frame_settings = render_frame_settings;

    const auto current_time = static_cast<float>(glfwGetTime());
    const float delta_time  = current_time - last_time;
    last_time               = current_time;
    curr_delta_time         = delta_time;

    input_manager->tick(delta_time);
    renderer.tick(delta_time);
    camera->tick(delta_time);

    static bool is_first_frame = true;
    if (!is_first_frame) render_frame_settings.should_capture_skybox = false;

    glfwGetWindowSize(window, &window_size.x, &window_size.y);

    if (old_render_frame_settings != render_frame_settings) {
        build_render_graph();
    }

    renderer.run_render_graph();

    is_first_frame = false;
}

void Engine::register_render_graph_resources() {
    auto& rr = render_resources;

    // ================== models and vertex buffers ==================

    scene_model = Model { "sponza", "../assets/example models/Sponza/Sponza.gltf", true };
    scene_model->register_render_graph_resources(renderer);

    renderer.add_frame_begin_action([&](const FrameBeginActionContext &fba_ctx) {
        Buffer& buffer = fba_ctx.resource_manager.get().get<Buffer>(scene_model->get_mesh_descriptions_buffer());
        const auto& mds = scene_model->get_mesh_descriptions();
        buffer.copy_from_ptr(mds.data(), mds.size() * sizeof(decltype(mds[0])));
    });

    rr[VB_Skybox] = renderer.register_resource(VertexBufferResourceDesc{
        .name = "skybox-vb",
        .size = skybox_vertices.size() * sizeof(SkyboxVertex),
        .data = skybox_vertices.data()
    });

    rr[VB_ScreenSpaceQuad] = renderer.register_resource(VertexBufferResourceDesc{
        .name = "ss-quad-vb",
        .size = screen_space_quad_vertices.size() * sizeof(ScreenSpaceQuadVertex),
        .data = screen_space_quad_vertices.data()
    });

    // ================== uniform buffers ==================

    rr[UBO_General] = renderer.register_resource(UniformBufferResourceDesc{
        .name = "general-ubo",
        .size = sizeof(GraphicsUBO)
    });

    renderer.add_frame_begin_action([&](const FrameBeginActionContext &fba_ctx) {
        Buffer& buffer = fba_ctx.resource_manager.get().get<Buffer>(rr[UBO_General]);
        update_graphics_uniform_buffer(buffer);
    });

    // ================== external textures ==================

    rr[Tex_Envmap] = renderer.register_resource(ExternalTextureResourceDesc{
        .name = "envmap-texture",
        .paths = {"../assets/envmaps/vienna.hdr"},
        .format = vk::Format::eR32G32B32A32Sfloat,
        .flags = ImageFlags::HDR
    });

    // ================== render target textures ==================

    constexpr auto skybox_tex_format = vk::Format::eR8G8B8A8Srgb;
    rr[Tex_Skybox] = renderer.register_resource(TargetTextureResourceDesc{
        .name = "skybox-texture",
        .format = skybox_tex_format,
        .extent = {2048, 2048},
        .flags = ImageFlags::CUBEMAP | ImageFlags::NO_MIPMAPS
    });

    constexpr auto g_buffer_color_format = vk::Format::eR16G16B16A16Sfloat;
    rr[Tex_GNormal] = renderer.register_resource(TargetTextureResourceDesc{
        .name = "g-buffer-normal",
        .format = g_buffer_color_format,
    });

    rr[Tex_GPos] = renderer.register_resource(TargetTextureResourceDesc{
        .name = "g-buffer-pos",
        .format = g_buffer_color_format,
    });

    constexpr auto g_buffer_depth_format = vk::Format::eD32Sfloat;
    rr[Tex_GDepth] = renderer.register_resource(TargetTextureResourceDesc{
        .name = "g-buffer-depth",
        .format = g_buffer_depth_format,
        .flags = ImageFlags::NO_MIPMAPS,
    });

    constexpr auto ssao_tex_format = vk::Format::eR8G8B8A8Unorm;
    rr[Tex_SSAO] = renderer.register_resource(TargetTextureResourceDesc{
        .name = "ssao-texture",
        .format = ssao_tex_format,
    });

    constexpr auto shadowmap_tex_format = vk::Format::eD32Sfloat;
    rr[Tex_Shadowmap] = renderer.register_resource(TargetTextureResourceDesc{
        .name = "shadowmap-texture",
        .format = shadowmap_tex_format,
        .extent = {2048, 2048},
        .layer_count = SHADOWMAP_CASCADE_COUNT,
        .overrides = {
            .mag_filter = vk::Filter::eNearest,
            .min_filter = vk::Filter::eNearest,
        },
        .flags = ImageFlags::NO_MIPMAPS,
    });

    constexpr auto final_no_gamma_format = vk::Format::eR8G8B8A8Unorm;
    rr[Tex_BasePass] = renderer.register_resource(TargetTextureResourceDesc{
        .name = "base-pass-texture",
        .format = final_no_gamma_format,
        .flags = ImageFlags::NO_MIPMAPS
    });
    rr[Tex_PostBlurX] = renderer.register_resource(TargetTextureResourceDesc{
        .name = "post-blur-x-texture",
        .format = final_no_gamma_format,
        .flags = ImageFlags::NO_MIPMAPS
    });
    rr[Tex_PostBlurY] = renderer.register_resource(TargetTextureResourceDesc{
        .name = "post-blur-y-texture",
        .format = final_no_gamma_format,
        .flags = ImageFlags::NO_MIPMAPS
    });
    rr[Tex_PostGui] = renderer.register_resource(TargetTextureResourceDesc{
        .name = "post-gui-texture",
        .format = final_no_gamma_format,
        .flags = ImageFlags::NO_MIPMAPS
    });

    // ================== shaders ==================

    renderer.set_shader_base_path("../shaders/obj/");

    rr[Pipe_SsQuadDepth] = renderer.register_resource(GraphicsPipelineResourceDesc{
        .vertex_path = "ss-quad-depth-vert.spv",
        .fragment_path = "ss-quad-depth-frag.spv",
        .vertex_bindings = ScreenSpaceQuadVertex::get_binding_descriptions(),
        .vertex_attributes = ScreenSpaceQuadVertex::get_attribute_descriptions(),
        .color_formats = { FINAL_FORMAT },
        .custom_properties = GraphicsPipelineResourceDesc::CustomProperties{
            .disable_depth_test = true,
        },
    });

    rr[Pipe_Cube] = renderer.register_resource(GraphicsPipelineResourceDesc{
        .vertex_path = "cube-vert.spv",
        .fragment_path = "cube-frag.spv",
        .vertex_bindings = SkyboxVertex::get_binding_descriptions(),
        .vertex_attributes = SkyboxVertex::get_attribute_descriptions(),
        .color_formats = {final_no_gamma_format},
        .depth_format = FINAL_FORMAT,
        .custom_properties = {
            .cull_mode = vk::CullModeFlagBits::eNone,
            .polygon_mode = vk::PolygonMode::eLine,
        },
    });

    rr[Pipe_CubeCapture] = renderer.register_resource(GraphicsPipelineResourceDesc{
        .vertex_path = "sphere-cube-vert.spv",
        .fragment_path = "sphere-cube-frag.spv",
        .vertex_bindings = SkyboxVertex::get_binding_descriptions(),
        .vertex_attributes = SkyboxVertex::get_attribute_descriptions(),
        .color_formats = {skybox_tex_format},
        .custom_properties = {
            .multiview_count = 6
        },
    });

    rr[Pipe_Shadowmap] = renderer.register_resource(GraphicsPipelineResourceDesc{
        .vertex_path = "shadowmap-vert.spv",
        .fragment_path = "shadowmap-frag.spv",
        .vertex_bindings = ModelVertex::get_binding_descriptions(),
        .vertex_attributes = ModelVertex::get_attribute_descriptions(),
        .depth_format = shadowmap_tex_format,
        .custom_properties = {
            .cull_mode = vk::CullModeFlagBits::eNone,
            .multiview_count = SHADOWMAP_CASCADE_COUNT,
        },
    });

    rr[Pipe_Prepass] = renderer.register_resource(GraphicsPipelineResourceDesc{
        .vertex_path = "prepass-vert.spv",
        .fragment_path = "prepass-frag.spv",
        .vertex_bindings = ModelVertex::get_binding_descriptions(),
        .vertex_attributes = ModelVertex::get_attribute_descriptions(),
        .color_formats = {g_buffer_color_format, g_buffer_color_format},
        .depth_format = g_buffer_depth_format,
    });

    rr[Pipe_SSAO] = renderer.register_resource(GraphicsPipelineResourceDesc{
        .vertex_path = "ssao-vert.spv",
        .fragment_path = "ssao-frag.spv",
        .vertex_bindings = ScreenSpaceQuadVertex::get_binding_descriptions(),
        .vertex_attributes = ScreenSpaceQuadVertex::get_attribute_descriptions(),
        .color_formats = {ssao_tex_format},
    });

    rr[Pipe_Skybox] = renderer.register_resource(GraphicsPipelineResourceDesc{
        .vertex_path = "skybox-vert.spv",
        .fragment_path = "skybox-frag.spv",
        .vertex_bindings = SkyboxVertex::get_binding_descriptions(),
        .vertex_attributes = SkyboxVertex::get_attribute_descriptions(),
        .color_formats = {final_no_gamma_format},
        .depth_format = FINAL_FORMAT,
        .custom_properties = {
            .depth_compare_op = vk::CompareOp::eLessOrEqual,
        },
    });

    rr[Pipe_Main] = renderer.register_resource(GraphicsPipelineResourceDesc{
        .vertex_path = "main-vert.spv",
        .fragment_path = "main-frag.spv",
        .vertex_bindings = ModelVertex::get_binding_descriptions(),
        .vertex_attributes = ModelVertex::get_attribute_descriptions(),
        .color_formats = {final_no_gamma_format},
        .depth_format = FINAL_FORMAT,
    });

    rr[Pipe_BlurX] = renderer.register_resource(ComputePipelineResourceDesc{
        .path = "blur-x-comp.spv",
    });

    rr[Pipe_BlurY] = renderer.register_resource(ComputePipelineResourceDesc{
        .path = "blur-y-comp.spv",
    });

    rr[Pipe_SsQuad] = renderer.register_resource(GraphicsPipelineResourceDesc{
        .vertex_path = "ss-quad-vert.spv",
        .fragment_path = "ss-quad-frag.spv",
        .vertex_bindings = ScreenSpaceQuadVertex::get_binding_descriptions(),
        .vertex_attributes = ScreenSpaceQuadVertex::get_attribute_descriptions(),
        .color_formats = {FINAL_FORMAT},
        .custom_properties = GraphicsPipelineResourceDesc::CustomProperties{
            .disable_depth_test = true,
        },
    });
}

void Engine::build_render_graph() {
    RenderGraph render_graph;
    const auto& rr = render_resources;
    const ResourceHandle model_mdb = scene_model->get_mesh_descriptions_buffer();

    const auto draw_model_helper = [](RenderPassContext &ctx, const Model& model, const bool push_mesh_ids = false) {
        uint32_t index_offset    = 0;
        int32_t vertex_offset    = 0;
        uint32_t instance_offset = 0;

        for (uint32_t mesh_id = 0; const auto &mesh: model.get_meshes()) {
            ctx.bind_vertex_buffers({ model.get_vertex_buffer(), model.get_instance_data_buffer() });
            ctx.bind_index_buffer(model.get_index_buffer());

            if (push_mesh_ids) {
                ctx.push_constants(mesh_id++, vk::ShaderStageFlagBits::eFragment);
            }

            ctx.draw_indexed(
                static_cast<uint32_t>(mesh.indices.size()),
                static_cast<uint32_t>(mesh.instances.size()),
                index_offset,
                vertex_offset,
                instance_offset
            );

            index_offset += static_cast<uint32_t>(mesh.indices.size());
            vertex_offset += static_cast<int32_t>(mesh.vertices.size());
            instance_offset += static_cast<uint32_t>(mesh.instances.size());
        }
    };

    // ================== nodes ==================

    if (render_frame_settings.should_capture_skybox) {
        render_graph.add_node(RenderNodeGraphics {
            .name = "cubemap-capture",
            .bound_resources = {},
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
        .bound_resources = {},
        .depth_target = rr[Tex_Shadowmap],
        .body = [&](RenderPassContext &ctx) {
            ctx.bind_pipeline(rr[Pipe_Shadowmap]);
            ctx.bind_resources({rr[UBO_General]});
            draw_model_helper(ctx, *scene_model);
        },
        .custom_properties = RenderNodeGraphics::CustomProperties {
            .multiview_count = SHADOWMAP_CASCADE_COUNT
        }
    });

    if (render_frame_settings.use_ssao) {
        render_graph.add_node(RenderNodeGraphics {
            .name = "prepass",
            .bound_resources = {},
            .color_targets = {rr[Tex_GNormal], rr[Tex_GPos]},
            .depth_target = rr[Tex_GDepth],
            .body = [&](RenderPassContext &ctx) {
                ctx.bind_pipeline(rr[Pipe_Prepass]);
                ctx.bind_resources({rr[UBO_General]});
                draw_model_helper(ctx, *scene_model);
            },
        });

        render_graph.add_node(RenderNodeGraphics {
            .name = "ssao",
            .bound_resources = {rr[Tex_GDepth], rr[Tex_GNormal], rr[Tex_GPos]},
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
        .bound_resources = {rr[Tex_SSAO], rr[Tex_Shadowmap], rr[Tex_Skybox]},
        .color_targets = {rr[Tex_BasePass]},
        .depth_target = FINAL_IMAGE_HANDLE,
        .body = [&, model_mdb](RenderPassContext &ctx) {
            ctx.bind_pipeline(rr[Pipe_Main]);
            ctx.bind_resources({rr[UBO_General], model_mdb, rr[Tex_SSAO], rr[Tex_Shadowmap]});
            draw_model_helper(ctx, *scene_model, true);

            ctx.bind_pipeline(rr[Pipe_Skybox]);
            ctx.bind_resources({rr[UBO_General], rr[Tex_Skybox]});
            ctx.draw(rr[VB_Skybox], skybox_vertices.size(), 1, 0, 0);
        },
    });

    optional<RenderNodeHandle> final_node_handle;

    if (render_frame_settings.do_blur) {
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

        final_node_handle = render_graph.add_node(RenderNodeGraphics {
            .name = "final-blurred",
            .bound_resources = {rr[Tex_PostBlurY]},
            .color_targets = {FINAL_IMAGE_HANDLE},
            .body = [&](RenderPassContext &ctx) {
                ctx.bind_pipeline(rr[Pipe_SsQuad]);
                ctx.bind_resources({rr[UBO_General], rr[Tex_PostBlurY]});
                ctx.push_constants(array<float, 4> { -1, 1, -1, 1 }, vk::ShaderStageFlagBits::eVertex);
                ctx.draw(rr[VB_ScreenSpaceQuad], screen_space_quad_vertices.size(), 1, 0, 0);
            },
        });

    } else {
        final_node_handle = render_graph.add_node(RenderNodeGraphics {
            .name = "final-unblurred",
            .bound_resources = {rr[Tex_BasePass]},
            .color_targets = {FINAL_IMAGE_HANDLE},
            .body = [&](RenderPassContext &ctx) {
                ctx.bind_pipeline(rr[Pipe_SsQuad]);
                ctx.bind_resources({rr[UBO_General], rr[Tex_BasePass]});
                ctx.push_constants(array<float, 4> { -1, 1, -1, 1 }, vk::ShaderStageFlagBits::eVertex);
                ctx.draw(rr[VB_ScreenSpaceQuad], screen_space_quad_vertices.size(), 1, 0, 0);

                if (render_frame_settings.show_debug_quad) {
                    ctx.bind_pipeline(rr[Pipe_SsQuadDepth]);
                    ctx.bind_resources({rr[UBO_General], rr[Tex_Shadowmap]});

                    constexpr vk::ShaderStageFlags stages = vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment;

                    ctx.push_constants(array<float, 5> { 0, 0.5, 0.5, 1, 0 }, stages);
                    ctx.draw(rr[VB_ScreenSpaceQuad], screen_space_quad_vertices.size(), 1, 0, 0);

                    ctx.push_constants(array<float, 5> { 0.5, 1, 0.5, 1, 1 }, stages);
                    ctx.draw(rr[VB_ScreenSpaceQuad], screen_space_quad_vertices.size(), 1, 0, 0);

                    ctx.push_constants(array<float, 5> { 0, 0.5, 0, 0.5, 2 }, stages);
                    ctx.draw(rr[VB_ScreenSpaceQuad], screen_space_quad_vertices.size(), 1, 0, 0);

                    ctx.push_constants(array<float, 5> { 0.5, 1, 0, 0.5, 3 }, stages);
                    ctx.draw(rr[VB_ScreenSpaceQuad], screen_space_quad_vertices.size(), 1, 0, 0);
                }
            },
        });
    }

    if (render_frame_settings.is_gui_enabled) {
        render_graph.add_node(RenderNodeGraphics {
            .name = "gui",
            .bound_resources = {},
            .color_targets = {FINAL_IMAGE_HANDLE},
            .body = [&](const RenderPassContext &ctx) {
                renderer.get_gui_renderer().begin_rendering();
                render_gui_section(curr_delta_time);
                renderer.get_gui_renderer().end_rendering(ctx.get_raw_cmd_buffer());
            },
            .explicit_dependencies = { *final_node_handle },
        });
    }

    renderer.register_render_graph(render_graph);
}

static auto get_frustum_corners_world_space(const glm::mat4& view, const glm::mat4& proj) -> array<glm::vec3, 8> {
    const glm::mat4 pxv_inverse = glm::inverse(proj * view);

    array<glm::vec3, 8> corners;
    constexpr auto r = views::iota(0, 2);

    // looping through [z,y,x] instead of [x,y,z] so that first 4 elements of `corners` are the near plane corners
    for (uint32_t i = 0; const auto& [z, y, x] : views::cartesian_product(r, r, r)) {
        const glm::vec4 corner = pxv_inverse * glm::vec4 {
            2.0f * x - 1.0f,
            2.0f * y - 1.0f,
            static_cast<float>(z),
            1.0f
        };
        corners[i++] = glm::vec3(corner / corner.w);
    }

    return corners;
}

auto Engine::get_light_pxv_matrix(const glm::mat4& model_mat, const float z_near, const float z_far) -> glm::mat4 {
    const glm::mat4 view = camera->get_view_matrix();
    const glm::mat4 proj = glm::gtc::perspective(glm::radians(camera->get_fov()), camera->get_aspect_ratio(), z_near, z_far);

    const auto camera_frustum_corners = get_frustum_corners_world_space(view, proj);

    const auto sum_points = [](const auto& r) -> glm::vec3 {
        return ranges::fold_left(r, glm::vec3 { 0, 0, 0 }, std::plus<glm::vec3>());
    };

    const glm::vec3 near_plane_center = (1.0f / 4) * sum_points(camera_frustum_corners | views::take(4));
    const glm::vec3 far_plane_center  = (1.0f / 4) * sum_points(camera_frustum_corners | views::reverse | views::take(4));
    const glm::vec3 frustum_direction = glm::normalize(far_plane_center - near_plane_center);

    const float near_plane_radius = glm::length(camera_frustum_corners[0] - near_plane_center);
    const float far_plane_radius = glm::length(camera_frustum_corners[4] - far_plane_center);
    const float frustum_depth = glm::length(far_plane_center - near_plane_center);
    constexpr auto sq = [](const float a) -> float { return a * a; };

    const float circumcenter_dist = (sq(frustum_depth) + sq(far_plane_radius) - sq(near_plane_radius)) / (2 * frustum_depth);
    const glm::vec3 circumcenter = near_plane_center + frustum_direction * circumcenter_dist;
    const float circumcircle_radius = glm::length(camera_frustum_corners[0] - circumcenter);

    const glm::vec3 sphere_aabb_min = circumcenter - glm::vec3(circumcircle_radius);
    const glm::vec3 sphere_aabb_max = circumcenter + glm::vec3(circumcircle_radius);

    const auto light_direction_vec = glm::vec3(glm::gtc::mat4_cast(light_direction) * glm::vec4(1, 0, 0, 0)) * 50.0f;
    const auto light_view = glm::gtc::lookAt(circumcenter + light_direction_vec, circumcenter, glm::vec3(0, 1, 0));

    float min_x = numeric_limits<float>::max();
    float max_x = numeric_limits<float>::lowest();
    float min_y = numeric_limits<float>::max();
    float max_y = numeric_limits<float>::lowest();
    float min_z = numeric_limits<float>::max();
    float max_z = numeric_limits<float>::lowest();

    constexpr auto r = views::iota(0, 2);
    for (const auto& [x, y, z] : views::cartesian_product(r, r, r)) {
        const glm::vec4 sphere_aabb_vertex = {
            x == 0 ? sphere_aabb_min.x : sphere_aabb_max.x,
            y == 0 ? sphere_aabb_min.y : sphere_aabb_max.y,
            z == 0 ? sphere_aabb_min.z : sphere_aabb_max.z,
            1.0f
        };
        const auto v = light_view * sphere_aabb_vertex;

        min_x = std::min(min_x, v.x);
        max_x = std::max(max_x, v.x);
        min_y = std::min(min_y, v.y);
        max_y = std::max(max_y, v.y);

        // careful here..
        min_z = std::min(min_z, v.z);
        max_z = std::max(max_z, v.z);
    }

    if (scene_model) {
        const auto [aabb_min, aabb_max] = scene_model->get_aabb();

        constexpr auto r = views::iota(0, 2);
        for (const auto& [x, y, z] : views::cartesian_product(r, r, r)) {
            const glm::vec4 aabb_vertex = {
                x == 0 ? aabb_min.x : aabb_max.x,
                y == 0 ? aabb_min.y : aabb_max.y,
                z == 0 ? aabb_min.z : aabb_max.z,
                1.0f
            };
            const auto v = light_view * model_mat * aabb_vertex;

            min_z = std::min(min_z, v.z);
            max_z = std::max(max_z, v.z);
        }
    }

    // const float z_mult = 1.0f;
    //
    // if (min_z < 0) min_z *= z_mult;
    // else min_z /= z_mult;
    //
    // if (max_z < 0) max_z /= z_mult;
    // else max_z *= z_mult;

    const auto light_proj = glm::gtc::ortho(min_x, max_x, min_y, max_y, -max_z, -min_z);

    return light_proj * light_view;
}

void Engine::update_graphics_uniform_buffer(const Buffer &buffer) {
    const glm::mat4 model = glm::gtc::translate(glm::gtc::identity<glm::mat4>(), model_translate)
                            * glm::gtc::mat4_cast(model_rotation)
                            * glm::gtc::scale(glm::gtc::identity<glm::mat4>(), glm::vec3(model_scale));
    const glm::mat4 view = camera->get_view_matrix();
    const glm::mat4 proj = camera->get_projection_matrix();

    glm::ivec2 window_size{};
    glfwGetWindowSize(window, &window_size.x, &window_size.y);

    const auto [z_near, z_far] = camera->get_clipping_planes();

    static const glm::mat4 cubemap_face_projection = glm::gtc::perspective(glm::radians(90.0f), 1.0f, 0.1f, 10.0f);

    const glm::vec3 light_direction_vec = glm::vec3(glm::gtc::mat4_cast(light_direction) * glm::vec4(1)) * 50.0f;

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
        },
        .misc = {
            .debug_number = debug_number,
            .z_near = z_near,
            .z_far = z_far,
            .use_ssao = render_frame_settings.use_ssao ? 1u : 0,
            .camera_pos = camera->get_pos(),
            .bias_weight_1 = shadow_map_config.bias_weight_1,
            .bias_weight_2 = shadow_map_config.bias_weight_2,
        }
    };

    constexpr array cascade_z_fars { 10.0f, 40.0f, 100.0f, 500.0f };
    float curr_z_near = z_near;

    for (uint32_t i = 0; i < SHADOWMAP_CASCADE_COUNT; i++) {
        const float curr_z_far = cascade_z_fars[i];

        graphics_ubo.light.cascade_pxv_mats[i] = get_light_pxv_matrix(model, curr_z_near, curr_z_far);
        graphics_ubo.light.cascade_z_fars[i].v = curr_z_far;

        curr_z_near = curr_z_far;
    }

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

void Engine::bind_key_actions() {
    input_manager->bind_callback(glfw::Key::KEY_GRAVE_ACCENT, EActivationType::PRESS_ONCE, [&](const float delta_time) {
        (void) delta_time;
        render_frame_settings.is_gui_enabled = !render_frame_settings.is_gui_enabled;
    });

    input_manager->bind_callback(glfw::Key::KEY_F1, EActivationType::PRESS_ONCE, [&](const float delta_time) {
        (void) delta_time;
        render_frame_settings.do_blur = !render_frame_settings.do_blur;
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

        ImGui::Checkbox("Debug quad", &render_frame_settings.show_debug_quad);

        if (ImGui::BeginCombo("Debug tex", "")) {
            const auto& rm = renderer.UNSAFE_get_resource_manager();
            for (const auto& handle: rm.get_all_resource_handles_range()) {
                bool is_ok_type = false;
                if (std::holds_alternative<ExternalTextureResourceDesc>(rm.get_desc_variant(handle)))
                    is_ok_type = true;
                if (std::holds_alternative<PersistentTextureResourceDesc>(rm.get_desc_variant(handle)))
                    is_ok_type = true;
                if (std::holds_alternative<TargetTextureResourceDesc>(rm.get_desc_variant(handle)))
                    is_ok_type = true;
                if (!is_ok_type) continue;

                if (ImGui::Selectable(rm.get_name(handle).c_str())) {
                    debug_tex = static_cast<uint32_t>(handle);
                }
            }
            ImGui::EndCombo();
        }

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
        ImGui::Checkbox("SSAO", &render_frame_settings.use_ssao);

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
