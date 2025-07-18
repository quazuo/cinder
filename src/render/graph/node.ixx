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
    reference_wrapper<const map<ResourceHandle, GraphicsPipeline>> graphics_pipelines;
    reference_wrapper<const map<ResourceHandle, ComputePipeline>> compute_pipelines;
    reference_wrapper<const vk::raii::DescriptorSet> bindless_set;

    optional<ResourceHandle> last_bound_pipeline;
    std::vector<ResourceHandle> bound_resource_ids;

public:
    explicit RenderPassContext(const vk::raii::CommandBuffer &cmd_buf, ResourceManager &rm,
                               const map<ResourceHandle, GraphicsPipeline> &graphics_pipelines,
                               const map<ResourceHandle, ComputePipeline> &compute_pipelines,
                               const vk::raii::DescriptorSet &bindless_set)
        : command_buffer(cmd_buf), resource_manager(rm),
          graphics_pipelines(graphics_pipelines), compute_pipelines(compute_pipelines),
          bindless_set(bindless_set) {
    }

    [[nodiscard]]
    const vk::raii::CommandBuffer& get_raw_cmd_buffer() const { return command_buffer.get(); }

    void bind_pipeline(ResourceHandle pipeline_handle);

    void bind_resources(const std::vector<ResourceHandle>& handles);

    void draw_model(ResourceHandle model_handle) const;

    void draw(ResourceHandle vertices_handle, uint32_t vertex_count, uint32_t instance_count,
              uint32_t first_vertex, uint32_t first_instance) const;

    void dispatch(uint32_t x, uint32_t y, uint32_t z) const;

private:
    void push_constants() const;
};

struct RenderNodeGraphics {
    using RenderNodeBodyFn   = std::function<void(RenderPassContext &)>;
    using ShouldRunPredicate = std::function<bool()>;

    string name;
    vector<ResourceHandle> bound_resources;
    vector<ResourceHandle> color_targets;
    optional<ResourceHandle> depth_target;
    RenderNodeBodyFn body;
    optional<ShouldRunPredicate> should_run_predicate;
    std::vector<RenderNodeHandle> explicit_dependencies;

    struct CustomProperties {
        uint32_t multiview_count = 1;
    } custom_properties;

    [[nodiscard]] set<ResourceHandle> get_all_non_final_targets_set() const;
};

struct RenderNodeCompute {
    using RenderNodeBodyFn   = std::function<void(RenderPassContext &)>;
    using ShouldRunPredicate = std::function<bool()>;

    string name;
    vector<ResourceHandle> bound_read_resources;
    vector<ResourceHandle> bound_write_resources;
    RenderNodeBodyFn body;
    optional<ShouldRunPredicate> should_run_predicate;
    std::vector<RenderNodeHandle> explicit_dependencies;

    struct CustomProperties {
    } custom_properties;
};

class RenderNode {
    variant<RenderNodeGraphics, RenderNodeCompute> node_;

public:
    RenderNode(const RenderNodeGraphics& node) : node_(node) {}
    RenderNode(const RenderNodeCompute& node)  : node_(node) {}

    [[nodiscard]] bool is_graphics() const { return std::holds_alternative<RenderNodeGraphics>(node_); }
    [[nodiscard]] bool is_compute()  const { return std::holds_alternative<RenderNodeCompute>(node_); }

    [[nodiscard]] const RenderNodeGraphics& get_graphics() const { return std::get<RenderNodeGraphics>(node_); }
    [[nodiscard]] RenderNodeGraphics&       get_graphics()       { return std::get<RenderNodeGraphics>(node_); }

    [[nodiscard]] const RenderNodeCompute& get_compute() const { return std::get<RenderNodeCompute>(node_); }
    [[nodiscard]] RenderNodeCompute&       get_compute()       { return std::get<RenderNodeCompute>(node_); }

    [[nodiscard]] const string& name() const;

    [[nodiscard]] bool should_run() const;

    [[nodiscard]] const std::vector<RenderNodeHandle>& explicit_dependencies() const;

    template <typename Functor>
    auto visit(Functor&& fn) const { return std::visit(std::forward<Functor>(fn), node_); }

    template <typename Functor>
    auto visit(Functor&& fn) { return std::visit(std::forward<Functor>(fn), node_); }
};
} // zrx
