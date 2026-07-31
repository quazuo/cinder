module;

export module Cinder.Render.Graph:Node;

import std;

import Cinder.Render.Vulkan;
import Cinder.Globals;
import Cinder.Utils;
import :Resource;
import :ResourceManager;

export namespace zrx {
struct RenderNodeHandleTag {};
using RenderNodeHandle = zrx::UniqueHandle<RenderNodeHandleTag>;

class RenderPassContext {
    reference_wrapper<const vk::raii::CommandBuffer> command_buffer;
    reference_wrapper<const ResourceManager> resource_manager;
    reference_wrapper<const vk::raii::DescriptorSet> bindless_set;

    optional<ResourceHandle> last_bound_pipeline;
    std::vector<ResourceHandle> bound_resource_handles;
    std::vector<BindlessHandle> bound_resource_ids;

public:
    explicit RenderPassContext(const vk::raii::CommandBuffer &cmd_buf, const ResourceManager &resource_manager,
                               const vk::raii::DescriptorSet &bindless_set)
        : command_buffer(cmd_buf), resource_manager(resource_manager), bindless_set(bindless_set) {
    }

    auto get_raw_cmd_buffer() const -> const vk::raii::CommandBuffer& { return command_buffer.get(); }

    void bind_pipeline(ResourceHandle pipeline_handle);

    void bind_resources(const std::vector<ResourceHandle>& handles);

    void bind_vertex_buffers(const std::vector<ResourceHandle>& vb_handles);

    void bind_index_buffer(ResourceHandle indices_handle);

    void draw(ResourceHandle vertices_handle, uint32_t vertex_count, uint32_t instance_count,
              uint32_t first_vertex, uint32_t first_instance);

    void draw(uint32_t vertex_count, uint32_t instance_count, uint32_t first_vertex, uint32_t first_instance);

    void draw_indexed(ResourceHandle vertices_handle, ResourceHandle indices_handle, uint32_t index_count, uint32_t instance_count,
                      uint32_t first_index, uint32_t vertex_offset, uint32_t first_instance);

    void draw_indexed(uint32_t index_count, uint32_t instance_count, uint32_t first_index, uint32_t vertex_offset, uint32_t first_instance);

    template <typename T>
    void push_constants(const T& constants, vk::ShaderStageFlags shader_stages) {
        if (!last_bound_pipeline) {
            Logger::error("no pipeline bound during draw!");
        }

        command_buffer.get().pushConstants<T>(
            *resource_manager.get().get<GraphicsPipeline>(*last_bound_pipeline).get_layout(),
            shader_stages,
            bound_resource_ids.size() * sizeof(decltype(bound_resource_ids[0])),
            constants
        );
    }

private:
    void push_bindless_constants();
};

class ComputePassContext {
    reference_wrapper<const vk::raii::CommandBuffer> command_buffer;
    reference_wrapper<const ResourceManager> resource_manager;
    reference_wrapper<const vk::raii::DescriptorSet> bindless_set;

    optional<ResourceHandle> last_bound_pipeline;
    std::vector<ResourceHandle> bound_resource_handles;
    std::vector<BindlessHandle> bound_resource_ids;

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
    using RenderNodeBodyFn = std::function<void(RenderPassContext &)>;

    string name;
    vector<ResourceHandle> bound_resources;
    vector<ResourceHandle> color_targets;
    optional<ResourceHandle> depth_target;
    RenderNodeBodyFn body;
    std::vector<RenderNodeHandle> explicit_dependencies;

    struct CustomProperties {
        uint32_t multiview_count = 1;
    } custom_properties;

    auto get_all_targets_set() const -> set<ResourceHandle>;
};

struct RenderNodeCompute {
    using RenderNodeBodyFn = std::function<void(ComputePassContext &)>;

    string name;
    vector<ResourceHandle> bound_read_resources;
    vector<ResourceHandle> bound_write_resources;
    RenderNodeBodyFn body;
    std::vector<RenderNodeHandle> explicit_dependencies;

    // struct CustomProperties {
    // } custom_properties;
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

    auto explicit_dependencies() const -> const std::vector<RenderNodeHandle>&;

    template <typename Functor>
    decltype(auto) visit(Functor&& fn) const { return std::visit(std::forward<Functor>(fn), node_); }

    template <typename Functor>
    decltype(auto) visit(Functor&& fn) { return std::visit(std::forward<Functor>(fn), node_); }
};
} // zrx
