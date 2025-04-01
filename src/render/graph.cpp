#include "graph.hpp"

#include "resource-manager.hpp"
#include "vk/pipeline.hpp"
#include "src/utils/logger.hpp"
#include "vk/descriptor.hpp"

namespace zrx {
[[nodiscard]] std::set<ResourceHandle> ShaderPack::get_bound_resources_set() const {
    std::set<ResourceHandle> result;

    for (const auto &set: descriptor_set_descs) {
        for (const auto &binding: set) {
            if (std::holds_alternative<ResourceHandle>(binding)) {
                result.insert(std::get<ResourceHandle>(binding));
            } else if (std::holds_alternative<ResourceHandleArray>(binding)) {
                result.insert(std::get<ResourceHandleArray>(binding).begin(),
                              std::get<ResourceHandleArray>(binding).end());
            }
        }
    }

    return result;
}

void RenderPassContext::bind_pipeline(const ResourceHandle pipeline_handle) {
    const auto &pipeline = pipelines.get().at(pipeline_handle);
    command_buffer.get().bindPipeline(vk::PipelineBindPoint::eGraphics, **pipeline);

    const auto& desc_sets = pipeline_desc_sets.get().at(pipeline_handle);
    std::vector<vk::DescriptorSet> raw_sets;
    for (const auto& set: desc_sets) {
        raw_sets.push_back(**set);
    }

    command_buffer.get().bindDescriptorSets(
        vk::PipelineBindPoint::eGraphics,
        *pipeline.get_layout(),
        0,
        raw_sets,
        nullptr
    );
}

void RenderPassContext::draw_model(const ResourceHandle model_handle) {
    uint32_t index_offset    = 0;
    int32_t vertex_offset    = 0;
    uint32_t instance_offset = 0;

    const Model &model = resource_manager.get().get_model(model_handle);
    model.bind_buffers(command_buffer);

    for (const auto &mesh: model.get_meshes()) {
        command_buffer.get().drawIndexed(
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
}

void RenderPassContext::draw(const ResourceHandle vertices_handle,
                             const uint32_t vertex_count, const uint32_t instance_count,
                             const uint32_t first_vertex, const uint32_t first_instance) {
    const Buffer &vertex_buffer = resource_manager.get().get_buffer(vertices_handle);
    command_buffer.get().bindVertexBuffers(0, *vertex_buffer, {0});
    command_buffer.get().draw(vertex_count, instance_count, first_vertex, first_instance);
}

std::set<ResourceHandle> RenderNode::get_all_targets_set() const {
    std::set result(color_targets.begin(), color_targets.end());
    if (depth_target) result.insert(*depth_target);
    return result;
}

std::set<ResourceHandle>
RenderNode::get_all_shader_resources_set(const std::map<ResourceHandle, ShaderPack> &shaders) const {
    ShaderGatherRenderPassContext ctx{};
    body(ctx);
    std::set<ResourceHandle> result;

    for (const ResourceHandle shader_handle: ctx.get()) {
        const auto bound_resources = shaders.at(shader_handle).get_bound_resources_set();
        result.insert(bound_resources.begin(), bound_resources.end());
    }

    return result;
}

vector<RenderNodeHandle> RenderGraph::get_topo_sorted() const {
    vector<RenderNodeHandle> result;

    std::set<RenderNodeHandle> remaining;

    for (const auto &[handle, _]: nodes) {
        remaining.emplace(handle);
    }

    while (!remaining.empty()) {
        for (const auto &handle: remaining) {
            if (std::ranges::all_of(dependency_graph.at(handle), [&](const RenderNodeHandle &dep) {
                return !remaining.contains(dep);
            })) {
                result.push_back(handle);
                remaining.erase(handle);
                break;
            }
        }
    }

    return result;
}

RenderNodeHandle RenderGraph::add_node(const RenderNode &node) {
    const auto handle = get_new_node_handle();
    nodes.emplace(handle, node);

    const auto targets_set      = node.get_all_targets_set();
    const auto shader_resources = node.get_all_shader_resources_set(pipelines);

    if (!detail::empty_intersection(targets_set, shader_resources)) {
        Logger::error("invalid render node: cannot use a target as a shader resource!");
    }

    std::set<RenderNodeHandle> dependencies;

    // for each existing node A...
    for (const auto &[other_handle, other_node]: nodes) {
        const auto other_targets_set      = other_node.get_all_targets_set();
        const auto other_shader_resources = other_node.get_all_shader_resources_set(pipelines);

        // ...if any of the new node's targets is sampled in A,
        // then the new node is A's dependency.
        if (!detail::empty_intersection(targets_set, other_shader_resources)) {
            dependency_graph.at(other_handle).emplace(handle);
        }

        // and if the new node samples any of A's targets,
        // then A is the new node's dependency.
        if (!detail::empty_intersection(other_targets_set, shader_resources)) {
            dependencies.emplace(other_handle);
        }
    }

    dependency_graph.emplace(handle, std::move(dependencies));

    check_dependency_cycles();

    return handle;
}

ResourceHandle RenderGraph::add_resource(VertexBufferResource &&resource) {
    return add_resource_generic(std::move(resource), vertex_buffers);
}

ResourceHandle RenderGraph::add_resource(UniformBufferResource &&resource) {
    return add_resource_generic(std::move(resource), uniform_buffers);
}

ResourceHandle RenderGraph::add_resource(ExternalTextureResource &&resource) {
    return add_resource_generic(std::move(resource), external_tex_resources);
}

ResourceHandle RenderGraph::add_resource(EmptyTextureResource &&resource) {
    return add_resource_generic(std::move(resource), empty_tex_resources);
}

ResourceHandle RenderGraph::add_resource(TransientTextureResource &&resource) {
    return add_resource_generic(std::move(resource), transient_tex_resources);
}

ResourceHandle RenderGraph::add_resource(ModelResource &&resource) {
    return add_resource_generic(std::move(resource), model_resources);
}

ResourceHandle RenderGraph::add_pipeline(ShaderPack &&resource) {
    return add_resource_generic(std::move(resource), pipelines);
}

void RenderGraph::add_frame_begin_action(FrameBeginCallback &&callback) {
    frame_begin_callbacks.emplace_back(std::move(callback));
}

void RenderGraph::cycles_helper(const RenderNodeHandle handle, std::set<RenderNodeHandle> &discovered,
                                std::set<RenderNodeHandle> &finished) const {
    discovered.emplace(handle);

    for (const auto &neighbour: dependency_graph.at(handle)) {
        if (discovered.contains(neighbour)) {
            Logger::error("invalid render graph: illegal cycle in dependency graph!");
        }

        if (!finished.contains(neighbour)) {
            cycles_helper(neighbour, discovered, finished);
        }
    }

    discovered.erase(handle);
    finished.emplace(handle);
};

void RenderGraph::check_dependency_cycles() const {
    std::set<RenderNodeHandle> discovered, finished;

    for (const auto &[handle, _]: nodes) {
        if (!discovered.contains(handle) && !finished.contains(handle)) {
            cycles_helper(handle, discovered, finished);
        }
    }
}

ResourceHandle RenderGraph::get_new_node_handle() {
    static RenderNodeHandle next_free_node_handle = 0;
    return next_free_node_handle++;
}

ResourceHandle RenderGraph::get_new_resource_handle() {
    static ResourceHandle next_free_resource_handle = 0;
    return next_free_resource_handle++;
}
} // zrx
