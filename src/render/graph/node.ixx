module;

export module Cinder.Render.Graph:Node;

import std;

import Cinder.Render.Vulkan;
import Cinder.Render.Mesh;
import Cinder.Globals;
import :Resource;
import :ResourceManager;

export namespace zrx {
using RenderNodeHandle = uint32_t;

class RenderPassContext {
    reference_wrapper<const vk::raii::CommandBuffer> command_buffer;
    reference_wrapper<ResourceManager> resource_manager;
    reference_wrapper<const std::map<ResourceHandle, GraphicsPipeline>> graphics_pipelines;
    reference_wrapper<const std::map<ResourceHandle, ComputePipeline>> compute_pipelines;
    reference_wrapper<const vk::raii::DescriptorSet> bindless_set;

    std::optional<ResourceHandle> last_bound_pipeline;
    std::vector<ResourceHandle> bound_resource_ids;

public:
    explicit RenderPassContext(const vk::raii::CommandBuffer &cmd_buf, ResourceManager &rm,
                               const std::map<ResourceHandle, GraphicsPipeline> &graphics_pipelines,
                               const std::map<ResourceHandle, ComputePipeline> &compute_pipelines,
                               const vk::raii::DescriptorSet &bindless_set)
        : command_buffer(cmd_buf), resource_manager(rm),
          graphics_pipelines(graphics_pipelines), compute_pipelines(compute_pipelines),
          bindless_set(bindless_set) {
    }

    void bind_pipeline(ResourceHandle pipeline_handle);

    void bind_resources(const std::vector<ResourceHandle>& handles);

    void draw_model(ResourceHandle model_handle) const;

    void draw(ResourceHandle vertices_handle, uint32_t vertex_count, uint32_t instance_count,
              uint32_t first_vertex, uint32_t first_instance) const;

    void dispatch(uint32_t x, uint32_t y, uint32_t z) const;

private:
    void push_constants() const;
};

struct RenderNode {
    using RenderNodeBodyFn   = std::function<void(RenderPassContext &)>;
    using ShouldRunPredicate = std::function<bool()>;

    string name;
    vector<ResourceHandle> bound_resources;
    vector<ResourceHandle> color_targets;
    std::optional<ResourceHandle> depth_target;
    RenderNodeBodyFn body;
    std::optional<ShouldRunPredicate> should_run_predicate;

    struct CustomProperties {
        uint32_t multiview_count = 1;
    } custom_properties;

    [[nodiscard]] std::set<ResourceHandle> get_all_targets_set() const;
};
} // zrx
