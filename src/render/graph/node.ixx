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
    reference_wrapper<const ResourceManager> resource_manager;
    reference_wrapper<const vk::raii::DescriptorSet> bindless_set;

    optional<ResourceHandle> last_bound_pipeline;
    std::vector<ResourceHandle> bound_resource_ids;

    optional<uint32_t> current_material_id;

public:
    explicit RenderPassContext(const vk::raii::CommandBuffer &cmd_buf, const ResourceManager &resource_manager,
                               const vk::raii::DescriptorSet &bindless_set)
        : command_buffer(cmd_buf), resource_manager(resource_manager), bindless_set(bindless_set) {
    }

    auto get_raw_cmd_buffer() const -> const vk::raii::CommandBuffer& { return command_buffer.get(); }

    void bind_pipeline(ResourceHandle pipeline_handle);

    void bind_resources(const std::vector<ResourceHandle>& handles);

    void draw_model(ResourceHandle model_handle);

    void draw(ResourceHandle vertices_handle, uint32_t vertex_count, uint32_t instance_count,
              uint32_t first_vertex, uint32_t first_instance);

private:
    void push_constants();
};

class ComputePassContext {
    reference_wrapper<const vk::raii::CommandBuffer> command_buffer;
    reference_wrapper<const ResourceManager> resource_manager;
    reference_wrapper<const vk::raii::DescriptorSet> bindless_set;

    optional<ResourceHandle> last_bound_pipeline;
    std::vector<ResourceHandle> bound_resource_ids;

public:
    explicit ComputePassContext(const vk::raii::CommandBuffer &cmd_buf, const ResourceManager &resource_manager,
                                const vk::raii::DescriptorSet &bindless_set)
        : command_buffer(cmd_buf), resource_manager(resource_manager), bindless_set(bindless_set) {
    }

    auto get_raw_cmd_buffer() const -> const vk::raii::CommandBuffer& { return command_buffer.get(); }

    void bind_pipeline(ResourceHandle pipeline_handle);

    void bind_resources(const std::vector<ResourceHandle>& handles);

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

    auto get_all_targets_set() const -> set<ResourceHandle>;
};

struct RenderNodeCompute {
    using RenderNodeBodyFn   = std::function<void(ComputePassContext &)>;
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

    auto is_graphics() const -> bool { return std::holds_alternative<RenderNodeGraphics>(node_); }
    auto is_compute()  const -> bool { return std::holds_alternative<RenderNodeCompute>(node_); }

    auto get_graphics() const -> const RenderNodeGraphics& { return std::get<RenderNodeGraphics>(node_); }
    auto get_graphics()       -> RenderNodeGraphics&       { return std::get<RenderNodeGraphics>(node_); }

    auto get_compute() const -> const RenderNodeCompute& { return std::get<RenderNodeCompute>(node_); }
    auto get_compute()       -> RenderNodeCompute&       { return std::get<RenderNodeCompute>(node_); }

    auto name() const -> const string&;

    auto should_run() const -> bool;

    auto explicit_dependencies() const -> const std::vector<RenderNodeHandle>&;

    template <typename Functor>
    decltype(auto) visit(Functor&& fn) const { return std::visit(std::forward<Functor>(fn), node_); }

    template <typename Functor>
    decltype(auto) visit(Functor&& fn) { return std::visit(std::forward<Functor>(fn), node_); }
};
} // zrx
