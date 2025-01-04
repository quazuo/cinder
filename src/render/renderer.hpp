#pragma once

#include <optional>
#include <vector>
#include <filesystem>
#include <array>
#include <queue>

#include "libs.hpp"
#include "globals.hpp"
#include "graph.hpp"
#include "mesh/model.hpp"
#include "vk/cmd.hpp"
#include "vk/image.hpp"
#include "vk/pipeline.hpp"
#include "vk/ctx.hpp"
#include "vk/descriptor.hpp"

#include <vk-bootstrap/VkBootstrap.h>

struct GLFWwindow;

static const std::vector device_extensions{
    VK_KHR_SWAPCHAIN_EXTENSION_NAME,
    VK_KHR_SYNCHRONIZATION_2_EXTENSION_NAME,
    VK_KHR_TIMELINE_SEMAPHORE_EXTENSION_NAME,
    VK_KHR_DYNAMIC_RENDERING_EXTENSION_NAME,
    VK_KHR_MULTIVIEW_EXTENSION_NAME,
    VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME,
    VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME,
    VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME,
    VK_EXT_DESCRIPTOR_INDEXING_EXTENSION_NAME
};

#ifdef NDEBUG
constexpr bool enable_validation_layers = false;
#else
constexpr bool enable_validation_layers = true;
#endif

namespace zrx {
class RenderTarget;
class InputManager;
class Model;
class Buffer;
class GraphicsPipeline;
class SwapChain;
class GuiRenderer;
class AccelerationStructure;
class Camera;

struct QueueFamilyIndices {
    std::optional<uint32_t> graphics_compute_family;
    std::optional<uint32_t> present_family;

    [[nodiscard]] bool isComplete() const {
        return graphics_compute_family.has_value() && present_family.has_value();
    }
};

struct ScenePushConstants {
    uint32_t material_id;
};

class RenderInfo {
    GraphicsPipelineBuilder cached_pipeline_builder;
    shared_ptr<GraphicsPipeline> pipeline;

    std::vector<RenderTarget> color_targets;
    std::optional<RenderTarget> depth_target;

    std::vector<vk::RenderingAttachmentInfo> color_attachments;
    std::optional<vk::RenderingAttachmentInfo> depth_attachment;

    std::vector<vk::Format> cached_color_attachment_formats;

public:
    RenderInfo(GraphicsPipelineBuilder builder, shared_ptr<GraphicsPipeline> pipeline,
               std::vector<RenderTarget> colors);

    RenderInfo(GraphicsPipelineBuilder builder, shared_ptr<GraphicsPipeline> pipeline,
               std::vector<RenderTarget> colors, RenderTarget depth);

    RenderInfo(std::vector<RenderTarget> colors);

    RenderInfo(std::vector<RenderTarget> colors, RenderTarget depth);

    [[nodiscard]] vk::RenderingInfo get(vk::Extent2D extent, uint32_t views = 1, vk::RenderingFlags flags = {}) const;

    [[nodiscard]] const GraphicsPipeline &get_pipeline() const { return *pipeline; }

    [[nodiscard]] vk::CommandBufferInheritanceRenderingInfo get_inheritance_rendering_info() const;

    void reload_shaders(const RendererContext &ctx) const;

private:
    void make_attachment_infos();
};

class VulkanRenderer {
    using CubemapCaptureDescriptorSet = FixedDescriptorSet<Buffer, Texture>;
    using DebugQuadDescriptorSet = FixedDescriptorSet<Texture>;
    using MaterialsDescriptorSet = FixedDescriptorSet<Texture, Texture, Texture, Texture>;
    using SceneDescriptorSet = FixedDescriptorSet<Buffer, Texture>;
    using SkyboxDescriptorSet = FixedDescriptorSet<Buffer, Texture>;
    using PrepassDescriptorSet = FixedDescriptorSet<Buffer>;
    using RtDescriptorSet = FixedDescriptorSet<Buffer, AccelerationStructure, Texture>;
    using SsaoDescriptorSet = FixedDescriptorSet<Buffer, Texture, Texture, Texture, Texture>;
    using MeshesDescriptorSet = FixedDescriptorSet<Buffer, Buffer, Buffer>;

    GLFWwindow *window = nullptr;

    unique_ptr<Camera> camera;

    unique_ptr<InputManager> input_manager;

    vk::raii::Context vk_ctx;
    unique_ptr<vk::raii::Instance> instance;
    unique_ptr<vk::raii::SurfaceKHR> surface;

    RendererContext ctx;

    unique_ptr<vk::raii::Queue> present_queue;
    QueueFamilyIndices queue_family_indices;

    unique_ptr<SwapChain> swap_chain;

    // render graph

    struct RenderNodeResources {
        RenderNodeHandle handle;
        vk::raii::CommandBuffer command_buffer;
        GraphicsPipeline pipeline;
        std::vector<shared_ptr<DescriptorSet> > descriptor_sets;
    };

    struct {
        unique_ptr<RenderGraph> render_graph;
        std::vector<RenderNodeResources> topo_sorted_nodes;
    } render_graph_info;

    std::map<ResourceHandle, unique_ptr<Buffer> > render_graph_ubos;
    std::map<ResourceHandle, unique_ptr<Texture> > render_graph_textures;

    // model

    unique_ptr<Model> model;
    Material separate_material;

    // textures

    unique_ptr<Texture> ssao_texture;
    unique_ptr<Texture> ssao_noise_texture;

    struct {
        unique_ptr<Texture> depth;
        unique_ptr<Texture> normal;
        unique_ptr<Texture> pos;
    } g_buffer_textures;

    unique_ptr<Texture> skybox_texture;
    unique_ptr<Texture> envmap_texture;

    unique_ptr<Texture> rt_target_texture;

    // descriptors

    unique_ptr<vk::raii::DescriptorPool> descriptor_pool;

    unique_ptr<MaterialsDescriptorSet> materials_descriptor_set;
    unique_ptr<MeshesDescriptorSet> meshes_descriptor_set;
    unique_ptr<CubemapCaptureDescriptorSet> cubemap_capture_descriptor_set;
    unique_ptr<DebugQuadDescriptorSet> debug_quad_descriptor_set;

    // render pass infos & misc pipelines

    std::vector<RenderInfo> scene_render_infos;
    std::vector<RenderInfo> skybox_render_infos;
    std::vector<RenderInfo> gui_render_infos;
    unique_ptr<RenderInfo> prepass_render_info;
    unique_ptr<RenderInfo> ssao_render_info;
    unique_ptr<RenderInfo> cubemap_capture_render_info;
    std::vector<RenderInfo> debug_quad_render_infos;

    unique_ptr<RtPipeline> rt_pipeline;

    // buffers and other resources

    unique_ptr<Buffer> skybox_vertex_buffer;
    unique_ptr<Buffer> screen_space_quad_vertex_buffer;

    unique_ptr<AccelerationStructure> tlas;

    using TimelineSemValueType = std::uint64_t;

    struct FrameResources {
        struct {
            struct Timeline {
                unique_ptr<vk::raii::Semaphore> semaphore;
                TimelineSemValueType timeline = 0;
            };

            unique_ptr<vk::raii::Semaphore> image_available_semaphore;
            unique_ptr<vk::raii::Semaphore> ready_to_present_semaphore;
            Timeline render_finished_timeline;
        } sync;

        // primary command buffer
        unique_ptr<vk::raii::CommandBuffer> graphics_cmd_buffer;

        SecondaryCommandBuffer scene_cmd_buffer;
        SecondaryCommandBuffer rt_cmd_buffer;
        SecondaryCommandBuffer prepass_cmd_buffer;
        SecondaryCommandBuffer ssao_cmd_buffer;
        SecondaryCommandBuffer gui_cmd_buffer;
        SecondaryCommandBuffer debug_cmd_buffer;

        unique_ptr<Buffer> graphics_uniform_buffer;
        void *graphics_ubo_mapped{};

        unique_ptr<SceneDescriptorSet> scene_descriptor_set;
        unique_ptr<SkyboxDescriptorSet> skybox_descriptor_set;
        unique_ptr<PrepassDescriptorSet> prepass_descriptor_set;
        unique_ptr<SsaoDescriptorSet> ssao_descriptor_set;
        unique_ptr<RtDescriptorSet> rt_descriptor_set;
    };

    static constexpr size_t MAX_FRAMES_IN_FLIGHT = 3;
    std::array<FrameResources, MAX_FRAMES_IN_FLIGHT> frame_resources;

    // gui stuff

    unique_ptr<vk::raii::DescriptorPool> imgui_descriptor_pool;
    unique_ptr<GuiRenderer> gui_renderer;

    // miscellaneous constants

    static constexpr auto prepass_color_format = vk::Format::eR16G16B16A16Sfloat;
    static constexpr auto hdr_envmap_format = vk::Format::eR32G32B32A32Sfloat;

    static constexpr uint32_t MATERIAL_TEX_ARRAY_SIZE = 32;

    // miscellaneous state variables

    using FrameBeginCallback = std::function<void()>;
    std::queue<FrameBeginCallback> queued_frame_begin_actions;

    uint32_t current_frame_idx = 0;

    bool framebuffer_resized = false;

    glm::vec3 background_color = glm::vec3(26, 26, 26) / 255.0f;

    vk::SampleCountFlagBits msaa_sample_count = vk::SampleCountFlagBits::e1;

    float model_scale = 1.0f;
    glm::vec3 model_translate{};
    glm::quat model_rotation{1, 0, 0, 0};

    glm::quat light_direction = glm::normalize(glm::vec3(1, 1.5, -2));
    glm::vec3 light_color = glm::normalize(glm::vec3(23.47, 21.31, 20.79));
    float light_intensity = 20.0f;

    float debug_number = 0;

    bool cull_back_faces = false;
    bool wireframe_mode = false;
    bool use_ssao = false;
    bool use_msaa = false;

public:
    explicit VulkanRenderer();

    ~VulkanRenderer();

    VulkanRenderer(const VulkanRenderer &other) = delete;

    VulkanRenderer(VulkanRenderer &&other) = delete;

    VulkanRenderer &operator=(const VulkanRenderer &other) = delete;

    VulkanRenderer &operator=(VulkanRenderer &&other) = delete;

    [[nodiscard]] GLFWwindow *get_window() const { return window; }

    [[nodiscard]] GuiRenderer &get_gui_renderer() const { return *gui_renderer; }

    [[nodiscard]] vk::SampleCountFlagBits get_msaa_sample_count() const {
        return use_msaa ? msaa_sample_count : vk::SampleCountFlagBits::e1;
    }

    void tick(float delta_time);

    void wait_idle() const { ctx.device->waitIdle(); }

    void register_render_graph(const RenderGraph &graph);

    void load_model_with_materials(const std::filesystem::path &path);

    void load_model(const std::filesystem::path &path);

    void load_base_color_texture(const std::filesystem::path &path);

    void load_normal_map(const std::filesystem::path &path);

    void load_orm_map(const std::filesystem::path &path);

    void load_orm_map(const std::filesystem::path &ao_path, const std::filesystem::path &roughness_path,
                      const std::filesystem::path &metallic_path);

    void load_rma_map(const std::filesystem::path &path);

    void load_environment_map(const std::filesystem::path &path);

    void reload_shaders() const;

private:
    static void framebuffer_resize_callback(GLFWwindow *window, int width, int height);

    void bind_mouse_drag_actions();

    // ==================== startup ====================

    vkb::Instance create_instance();

    static std::vector<const char *> get_required_extensions();

    void create_surface();

    vkb::PhysicalDevice pick_physical_device(const vkb::Instance &vkb_instance);

    void create_logical_device(const vkb::PhysicalDevice &vkb_physical_device);

    // ==================== assets ====================

    void create_skybox_texture();

    void create_prepass_textures();

    void create_ssao_textures();

    void create_rt_target_texture();

    // ==================== swap chain ====================

    void recreate_swap_chain();

    // ==================== descriptors ====================

    void create_descriptor_pool();

    void create_scene_descriptor_sets();

    void create_materials_descriptor_set();

    void create_skybox_descriptor_sets();

    void create_prepass_descriptor_sets();

    void create_ssao_descriptor_sets();

    void create_cubemap_capture_descriptor_set();

    void create_debug_quad_descriptor_set();

    void create_rt_descriptor_sets();

    void create_meshes_descriptor_set();

    // ==================== render infos ====================

    void create_scene_render_infos();

    void create_skybox_render_infos();

    void create_gui_render_infos();

    void create_prepass_render_info();

    void create_ssao_render_info();

    void create_cubemap_capture_render_info();

    void create_debug_quad_render_infos();

    // ==================== multisampling ====================

    [[nodiscard]] vk::SampleCountFlagBits get_max_usable_sample_count() const;

    // ==================== buffers ====================

    void create_skybox_vertex_buffer();

    void create_screen_space_quad_vertex_buffer();

    template<typename ElemType>
    unique_ptr<Buffer> create_local_buffer(const std::vector<ElemType> &contents, vk::BufferUsageFlags usage);

    void create_uniform_buffers();

    // ==================== commands ====================

    void create_command_pool();

    void create_command_buffers();

    void record_graphics_command_buffer();

    // ==================== sync ====================

    void create_sync_objects();

    // ==================== ray tracing ====================

    void create_tlas();

    void create_rt_pipeline();

    // ==================== gui ====================

    void init_imgui();

public:
    void render_gui_section();

    // ==================== render graph ====================

private:
    void create_render_graph_resources();

    [[nodiscard]] std::vector<shared_ptr<DescriptorSet> > create_node_descriptor_sets(RenderNodeHandle handle) const;

    [[nodiscard]] GraphicsPipeline
    create_node_pipeline(RenderNodeHandle handle, const std::vector<shared_ptr<DescriptorSet>> &descriptor_sets) const;

    void record_render_graph_node_commands(const RenderNodeResources &node_resources);

public:
    void run_render_graph();

    // ==================== render loop ====================

    bool start_frame();

    void end_frame();

    void render_gui(const std::function<void()> &render_commands);

    void run_prepass();

    void run_ssao_pass();

    void raytrace();

    void draw_scene();

    void draw_debug_quad();

private:
    void draw_model(const vk::raii::CommandBuffer &command_buffer, bool do_push_constants,
                    const GraphicsPipeline &pipeline) const;

    void capture_cubemap() const;

    void update_graphics_uniform_buffer() const;
};
} // zrx
