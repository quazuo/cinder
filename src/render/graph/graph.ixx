module;

#include "src/render/mesh/model.hpp"
#include "src/render/vk/image.hpp"
#include "src/render/vk/buffer.hpp"
#include "src/render/resource-manager.hpp"

export module Cinder.Render.Graph;

export import :Node;
export import :Resource;

import std;

export namespace zrx {
struct FrameBeginActionContext {
    reference_wrapper<ResourceManager> resource_manager;
};

using FrameBeginCallback = std::function<void(const FrameBeginActionContext &)>;

class RenderGraph {
    std::map<RenderNodeHandle, RenderNode> nodes;
    std::map<RenderNodeHandle, std::set<RenderNodeHandle> > dependency_graph;

    std::map<ResourceHandle, VertexBufferResourceDesc> vertex_buffers;
    std::map<ResourceHandle, UniformBufferResourceDesc> uniform_buffers;
    std::map<ResourceHandle, ExternalTextureResourceDesc> external_tex_resources;
    std::map<ResourceHandle, TargetTextureResourceDesc> empty_tex_resources;
    std::map<ResourceHandle, TransientTextureResourceDesc> transient_tex_resources;
    std::map<ResourceHandle, ModelResourceDesc> model_resources;
    std::map<ResourceHandle, GraphicsPipelineDesc> graphics_pipelines;
    std::map<ResourceHandle, ComputePipelineDesc> compute_pipelines;

    std::set<ResourceHandle> produced_resources;
    std::map<ResourceHandle, string> resource_names;

    vector<FrameBeginCallback> frame_begin_callbacks;

    friend class VulkanRenderer;

public:
    [[nodiscard]] vector<RenderNodeHandle> get_topo_sorted() const;

    RenderNodeHandle add_node(const RenderNode &node);

    [[nodiscard]] ResourceHandle add_resource(VertexBufferResourceDesc &&resource);

    [[nodiscard]] ResourceHandle add_resource(UniformBufferResourceDesc &&resource);

    [[nodiscard]] ResourceHandle add_resource(ExternalTextureResourceDesc &&resource);

    [[nodiscard]] ResourceHandle add_resource(TargetTextureResourceDesc &&resource);

    [[nodiscard]] ResourceHandle add_resource(TransientTextureResourceDesc &&resource);

    [[nodiscard]] ResourceHandle add_resource(ModelResourceDesc &&resource);

    template<typename T>
        requires ResourceLike<T>
    [[nodiscard]] vector<ResourceHandle> add_repeated_resource(const uint32_t count, T&& resource) {
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
    void cycles_helper(RenderNodeHandle handle, std::set<RenderNodeHandle> &discovered,
                       std::set<RenderNodeHandle> &finished) const;

    void check_dependency_cycles() const;

    [[nodiscard]] static ResourceHandle get_new_node_handle();

    [[nodiscard]] static ResourceHandle get_new_resource_handle();

    template<typename ResourceType>
    [[nodiscard]] ResourceHandle
    add_resource_generic(ResourceType &&resource, std::map<ResourceHandle, ResourceType> &resource_map) {
        const auto handle = get_new_resource_handle();
        resource_map.emplace(handle, resource);
        if constexpr (ResourceLike<ResourceType>) {
            resource_names.emplace(handle, resource.name);
        }
        return handle;
    }
};
} // zrx
