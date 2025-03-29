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

static const vector device_extensions{
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
constexpr bool ENABLE_VALIDATION_LAYERS = false;
#else
constexpr bool ENABLE_VALIDATION_LAYERS = true;
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
class ResourceManager;

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
    vector<RenderTarget> color_targets;
    std::optional<RenderTarget> depth_target;

    vector<vk::RenderingAttachmentInfo> color_attachments;
    std::optional<vk::RenderingAttachmentInfo> depth_attachment;

    vector<vk::Format> cached_color_attachment_formats;

public:
    RenderInfo(vector<RenderTarget> colors);

    RenderInfo(vector<RenderTarget> colors, RenderTarget depth);

    [[nodiscard]] vk::RenderingInfo get(vk::Extent2D extent, uint32_t views = 1, vk::RenderingFlags flags = {}) const;

private:
    void make_attachment_infos();
};

class VulkanRenderer {
    GLFWwindow *window = nullptr;

    vk::raii::Context vk_ctx;
    unique_ptr<vk::raii::Instance> instance;
    unique_ptr<vk::raii::SurfaceKHR> surface;

    RendererContext ctx;

    unique_ptr<vk::raii::Queue> present_queue;
    QueueFamilyIndices queue_family_indices;

    unique_ptr<SwapChain> swap_chain;

    unique_ptr<vk::raii::DescriptorPool> descriptor_pool;

    // render graph stuff

    struct RenderNodeResources {
        RenderNodeHandle handle;
        vector<RenderInfo> render_infos;
    };

    struct {
        unique_ptr<RenderGraph> render_graph;
        vector<RenderNodeResources> topo_sorted_nodes;
    } render_graph_info;

    unique_ptr<ResourceManager> resource_manager = make_unique<ResourceManager>();
    std::map<ResourceHandle, GraphicsPipeline> render_graph_pipelines;
    std::map<ResourceHandle, vector<DescriptorSet>> pipeline_desc_sets;

    // other resources

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

        unique_ptr<vk::raii::CommandBuffer> graphics_cmd_buffer;
    };

    static constexpr size_t MAX_FRAMES_IN_FLIGHT = 3;
    std::array<FrameResources, MAX_FRAMES_IN_FLIGHT> frame_resources;

    // gui stuff

    unique_ptr<vk::raii::DescriptorPool> imgui_descriptor_pool;
    unique_ptr<GuiRenderer> gui_renderer;

    // miscellaneous state variables

    vector<FrameBeginCallback> repeated_frame_begin_actions;
    std::queue<FrameBeginCallback> queued_frame_begin_actions;

    uint32_t current_frame_idx = 0;

    bool framebuffer_resized = false;

    vk::SampleCountFlagBits msaa_sample_count = vk::SampleCountFlagBits::e1;
    bool use_msaa = false;

    friend RenderPassContext;
    friend ShaderGatherRenderPassContext;

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

private:
    static void framebuffer_resize_callback(GLFWwindow *window, int width, int height);

    // ==================== startup ====================

    vkb::Instance create_instance();

    static vector<const char *> get_required_extensions();

    void create_surface();

    vkb::PhysicalDevice pick_physical_device(const vkb::Instance &vkb_instance);

    void create_logical_device(const vkb::PhysicalDevice &vkb_physical_device);

    // ==================== swap chain ====================

    void recreate_swap_chain();

    // ==================== descriptors ====================

    void create_descriptor_pool();

    // ==================== multisampling ====================

    [[nodiscard]] vk::SampleCountFlagBits get_max_usable_sample_count() const;

    // ==================== buffers ====================

    template<typename ElemType>
    unique_ptr<Buffer> create_local_buffer(const vector<ElemType> &contents, vk::BufferUsageFlags usage);

    // ==================== commands ====================

    void create_command_pool();

    void create_command_buffers();

    // ==================== sync ====================

    void create_sync_objects();

    // ==================== gui ====================

    void init_imgui();

public:
    void render_gui_section();

    // ==================== render graph ====================

private:
    void create_render_graph_resources();

    [[nodiscard]] vector<DescriptorSet> create_graph_descriptor_sets(ResourceHandle pipeline_handle) const;

    [[nodiscard]] GraphicsPipelineBuilder create_graph_pipeline_builder(
        ResourceHandle pipeline_handle, const vector<DescriptorSet> &descriptor_sets) const;

    void queue_set_update_with_handle(DescriptorSet &descriptor_set, ResourceHandle res_handle,
                                      uint32_t binding, uint32_t array_element = 0) const;

    [[nodiscard]] vector<RenderInfo> create_node_render_infos(RenderNodeHandle node_handle) const;

    void record_graph_commands() const;

    void record_node_commands(const RenderNodeResources &node_resources) const;

    void record_node_rendering_commands(const RenderNodeResources &node_resources) const;

    void record_regenerate_mipmaps_commands(const RenderNodeResources &node_resources) const;

    void record_pre_sample_commands(const RenderNodeResources &node_resources) const;

    [[nodiscard]] bool has_swapchain_target(RenderNodeHandle handle) const;

    [[nodiscard]] bool is_first_node_targetting_final_image(RenderNodeHandle handle) const;

    [[nodiscard]] bool should_run_node_pass(RenderNodeHandle handle) const;

    [[nodiscard]] vk::Extent2D get_node_target_extent(const RenderNodeResources &node_resources) const;

    [[nodiscard]] vk::Format get_target_color_format(ResourceHandle handle) const;

    [[nodiscard]] vk::Format get_target_depth_format(ResourceHandle handle) const;

    // ==================== render loop ====================

public:
    void run_render_graph();

    void do_frame_begin_actions();

    bool start_frame();

    void end_frame();
};
} // zrx
