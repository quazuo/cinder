module;

export module Cinder.Render:Renderer;

import vk_bootstrap;
import vulkan;
import std;
import glfw;
import cvulkan;
import vulkan;

import Cinder.Utils;
import Cinder.Render.Graph;
import Cinder.Render.Vulkan;
import Cinder.Render.Gui;
import Cinder.Globals;

#ifdef NDEBUG
export constexpr bool ENABLE_VALIDATION_LAYERS = false;
#else
export constexpr bool ENABLE_VALIDATION_LAYERS = true;
#endif

export namespace zrx {
class RenderInfo {
    vector<RenderTarget> color_targets;
    optional<RenderTarget> depth_target;

    vector<vk::RenderingAttachmentInfo> color_attachments;
    optional<vk::RenderingAttachmentInfo> depth_attachment;

    vector<vk::Format> cached_color_attachment_formats;

public:
    RenderInfo(vector<RenderTarget> colors);

    RenderInfo(vector<RenderTarget> colors, RenderTarget depth);

    auto get(vk::Extent2D extent, uint32_t views = 1, vk::RenderingFlags flags = {}) const -> vk::RenderingInfo;

private:
    void make_attachment_infos();
};

class VulkanRenderer {
    vk::raii::Context vk_ctx;
    unique_ptr<vk::raii::Instance> instance;
    unique_ptr<vk::raii::DebugUtilsMessengerEXT> debug_messenger;
    unique_ptr<vk::raii::SurfaceKHR> surface;

    RendererContext ctx;

    unique_ptr<PresentQueue> present_queue;
    QueueFamilyIndices queue_family_indices;

    unique_ptr<SwapChain> swap_chain;

    unique_ptr<vk::raii::DescriptorPool> descriptor_pool;

    // bindless resources

    using BindlessDescriptorSet = FixedDescriptorSet<Image, Image, Buffer>;
    unique_ptr<BindlessDescriptorSet> bindless_descriptor_set;

    static constexpr uint32_t BINDLESS_ARRAY_SIZE = 256;

    static constexpr uint32_t BINDLESS_SAMPLER_BINDING         = 0;
    static constexpr uint32_t BINDLESS_STORAGE_TEXTURE_BINDING = 1;
    static constexpr uint32_t BINDLESS_UBO_BINDING             = 2;

    // ================ render graph stuff ================

    struct RenderNodeResources {
        vector<RenderInfo> render_infos;
    };

    unique_ptr<RenderGraph> render_graph;
    vector<vector<RenderNodeHandle>> partitioned_nodes;
    map<RenderNodeHandle, RenderNodeResources> node_resources;

    unique_ptr<ResourceManager> resource_manager;

    // command recording state
    set<ResourceHandle> unbarriered_gfx_written_resources;
    set<ResourceHandle> unbarriered_compute_written_resources;
    map<ResourceHandle, vk::ImageLayout> last_image_layouts;

    // ================ other stuff ================

    struct FrameResources {
        struct {
            unique_ptr<BinarySemaphore> image_available_sem;
            unique_ptr<BinarySemaphore> ready_to_present_sem;
            unique_ptr<TimelineSemaphore> render_finished_timeline_sem;
        } sync;

        unique_ptr<vk::raii::CommandBuffer> main_cmd_buffer;
        unique_ptr<QueryPool> time_query_pool;
    };

    static constexpr size_t MAX_FRAMES_IN_FLIGHT = 3;
    array<FrameResources, MAX_FRAMES_IN_FLIGHT> frame_resources;

    // ================ gui ================

    unique_ptr<vk::raii::DescriptorPool> imgui_descriptor_pool;
    unique_ptr<GuiRenderer> gui_renderer;

    // ================ profiling ================

    vector<vector<RenderNodeHandle>> prev_frame_partitioned_nodes;
    vector<uint64_t> prev_frame_time_query_results;
    uint32_t current_query_idx = 0;
    float timestamp_period = 0.0f;

    // ================ miscellaneous state variables ================

    vector<FrameBeginCallback> repeated_frame_begin_actions;
    std::queue<FrameBeginCallback> queued_frame_begin_actions;

    bool framebuffer_resized = false;

    vk::SampleCountFlagBits msaa_sample_count = vk::SampleCountFlagBits::e1;
    bool use_msaa = false;

    std::filesystem::path shader_base_path;

public:
    explicit VulkanRenderer();

    ~VulkanRenderer();

    VulkanRenderer(const VulkanRenderer &other) = delete;

    VulkanRenderer(VulkanRenderer &&other) = delete;

    VulkanRenderer &operator=(const VulkanRenderer &other) = delete;

    VulkanRenderer &operator=(VulkanRenderer &&other) = delete;

    auto get_window() const -> GLFWwindow* { return ctx.window; }

    auto get_gui_renderer() const -> GuiRenderer& { return *gui_renderer; }

    auto get_resource_manager() const -> ResourceManager& { return *resource_manager; }

    auto get_msaa_sample_count() const -> vk::SampleCountFlagBits {
        return use_msaa ? msaa_sample_count : vk::SampleCountFlagBits::e1;
    }

    void tick(float delta_time);

    void wait_idle() const { ctx.device->waitIdle(); }

    void register_render_graph(const RenderGraph &graph);

    void reload_all_pipelines();

    void set_shader_base_path(const std::filesystem::path &path) { shader_base_path = path; }

private:
    static void framebuffer_resize_callback(GLFWwindow *window, int width, int height);

    // ==================== startup ====================

    auto create_instance() -> vkb::Instance;

    static auto get_required_extensions() -> vector<const char *>;

    void create_surface();

    auto pick_physical_device(const vkb::Instance &vkb_instance) -> vkb::PhysicalDevice;

    void create_logical_device(const vkb::PhysicalDevice &vkb_physical_device);

    void create_queues();

    // ==================== swap chain ====================

    void recreate_swap_chain();

    // ==================== descriptors ====================

    void create_descriptor_pool();

    void create_bindless_resources();

    // ==================== multisampling ====================

    auto get_max_usable_sample_count() const -> vk::SampleCountFlagBits;

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

    void create_resource(ResourceHandle handle, const VertexBufferResourceDesc& description);
    void create_resource(ResourceHandle handle, const UniformBufferResourceDesc& description);
    void create_resource(ResourceHandle handle, const ExternalTextureResourceDesc& description);
    void create_resource(ResourceHandle handle, const TargetTextureResourceDesc& description);
    void create_resource(ResourceHandle handle, const PersistentTextureResourceDesc& description);
    void create_resource(ResourceHandle handle, const TransientTextureResourceDesc& description);
    void create_resource(ResourceHandle handle, const ModelResourceDesc& description);
    void create_resource(ResourceHandle handle, const GraphicsPipelineResourceDesc& description);
    void create_resource(ResourceHandle handle, const ComputePipelineResourceDesc& description);

    void queue_set_update_with_handle(DescriptorSet &descriptor_set, ResourceHandle res_handle,
                                      uint32_t binding, uint32_t array_element = 0) const;

    auto create_node_render_infos(RenderNodeHandle node_handle) const -> vector<RenderInfo>;

    void record_graph_commands();

    void record_graphics_node_commands(RenderNodeHandle node_handle);

    void record_node_rendering_commands(RenderNodeHandle node_handle) const;

    void record_regenerate_mipmaps_commands(RenderNodeHandle node_handle);

    void record_pre_partition_commands(const vector<RenderNodeHandle> &partition);

    void record_compute_node_commands(RenderNodeHandle node_handle);

    auto gather_attachment_resources() const -> set<ResourceHandle>;

    auto gather_compute_accessed_resources() const -> set<ResourceHandle>;

    auto has_swapchain_target(RenderNodeHandle node_handle) const -> bool;

    auto is_first_node_targetting_final_image(RenderNodeHandle node_handle) const -> bool;

    auto get_node_target_extent(RenderNodeHandle node_handle) const -> vk::Extent2D;

    auto get_target_color_format(ResourceHandle resource_handle) const -> vk::Format;

    auto get_target_depth_format(ResourceHandle resource_handle) const -> vk::Format;

    auto get_node_target_handles(RenderNodeHandle node_handle) const -> vector<ResourceHandle>;

    // ==================== render loop ====================

public:
    void run_render_graph();

    void add_frame_begin_action(FrameBeginCallback &&callback);

    void do_frame_begin_actions();

    auto start_frame() -> bool;

    void end_frame();
};
} // zrx
