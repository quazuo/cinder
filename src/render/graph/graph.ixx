module;

export module Cinder.Render.Graph;

export import :Node;
export import :Resource;
export import :ResourceManager;

import std;

import Cinder.Render.Vulkan;
import Cinder.Render.Mesh;
import Cinder.Globals;

export namespace zrx {
struct FrameBeginActionContext {
    reference_wrapper<ResourceManager> resource_manager;
};

using FrameBeginCallback = std::function<void(const FrameBeginActionContext &)>;

class RenderGraph {
    map<RenderNodeHandle, RenderNode> nodes_; // todo - replace by vector
    map<RenderNodeHandle, set<RenderNodeHandle> > dependency_graph;

    // todo - replace by vectors
    map<ResourceHandle, VertexBufferResourceDesc> vertex_buffers_;
    map<ResourceHandle, UniformBufferResourceDesc> uniform_buffers_;
    map<ResourceHandle, ExternalTextureResourceDesc> external_tex_resources_;
    map<ResourceHandle, TargetTextureResourceDesc> target_tex_resources_;
    map<ResourceHandle, TransientTextureResourceDesc> transient_tex_resources_;
    map<ResourceHandle, ModelResourceDesc> model_resources_;
    map<ResourceHandle, GraphicsPipelineDesc> graphics_pipelines_;
    map<ResourceHandle, ComputePipelineDesc> compute_pipelines_;

    set<RenderNodeHandle> nodes_writing_to_final;
    set<ResourceHandle> produced_resources;
    map<ResourceHandle, string> resource_names;

    vector<FrameBeginCallback> frame_begin_callbacks_;

public:
    [[nodiscard]] const auto &nodes()                   const { return nodes_; }
    [[nodiscard]] const auto &vertex_buffers()          const { return vertex_buffers_; }
    [[nodiscard]] const auto &uniform_buffers()         const { return uniform_buffers_; }
    [[nodiscard]] const auto &external_tex_resources()  const { return external_tex_resources_; }
    [[nodiscard]] const auto &target_tex_resources()    const { return target_tex_resources_; }
    [[nodiscard]] const auto &transient_tex_resources() const { return transient_tex_resources_; }
    [[nodiscard]] const auto &model_resources()         const { return model_resources_; }
    [[nodiscard]] const auto &graphics_pipelines()      const { return graphics_pipelines_; }
    [[nodiscard]] const auto &compute_pipelines()       const { return compute_pipelines_; }
    [[nodiscard]] const auto &frame_begin_callbacks()   const { return frame_begin_callbacks_; }

    [[nodiscard]] vector<RenderNodeHandle> get_topo_sorted() const;

    [[nodiscard]] vector<vector<RenderNodeHandle>> get_partitioned() const;

    RenderNodeHandle add_node(const RenderNodeGraphics &node);

    RenderNodeHandle add_node(const RenderNodeCompute &node);

    /// Adds multiple nodes at once, connected sequentially via explicit dependencies.
    vector<RenderNodeHandle> add_nodes_sequential(vector<RenderNode> nodes);

    [[nodiscard]] ResourceHandle add_resource(VertexBufferResourceDesc &&resource);

    [[nodiscard]] ResourceHandle add_resource(UniformBufferResourceDesc &&resource);

    [[nodiscard]] ResourceHandle add_resource(ExternalTextureResourceDesc &&resource);

    [[nodiscard]] ResourceHandle add_resource(TargetTextureResourceDesc &&resource);

    [[nodiscard]] ResourceHandle add_resource(TransientTextureResourceDesc &&resource);

    [[nodiscard]] ResourceHandle add_resource(ModelResourceDesc &&resource);

    template<typename T>
        requires ResourceLike<T>
    [[nodiscard]] vector<ResourceHandle> add_repeated_resource(const uint32_t count, T &&resource) {
        vector<ResourceHandle> handles;

        for (uint32_t i = 0; i < count; i++) {
            T updated_res_desc = resource;
            updated_res_desc.name += "#" + std::to_string(i);
            handles.emplace_back(add_resource(updated_res_desc));
        }

        return handles;
    }

    [[nodiscard]] ResourceHandle add_pipeline(GraphicsPipelineDesc &&resource);

    [[nodiscard]] ResourceHandle add_pipeline(ComputePipelineDesc &&resource);

    void add_frame_begin_action(FrameBeginCallback &&callback);

private:
    void add_new_dependencies(RenderNodeHandle new_handle);

    void cycles_helper(RenderNodeHandle handle, set<RenderNodeHandle> &discovered,
                       set<RenderNodeHandle> &finished) const;

    void check_dependency_cycles() const;

    [[nodiscard]] static ResourceHandle get_new_node_handle();

    [[nodiscard]] static ResourceHandle get_new_resource_handle();

    template<typename ResourceType>
    [[nodiscard]] ResourceHandle
    add_resource_generic(ResourceType &&resource, map<ResourceHandle, ResourceType> &resource_map) {
        const auto handle = get_new_resource_handle();
        resource_map.emplace(handle, resource);
        if constexpr (ResourceLike<ResourceType>) {
            resource_names.emplace(handle, resource.name);
        }
        return handle;
    }
};
} // zrx
