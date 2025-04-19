module;

module Cinder.Render.Graph;

import std;

import Cinder.Utils;
import Cinder.Render.Vulkan;

import Cinder.Globals;

namespace detail {
template<typename T>
[[nodiscard]] bool empty_intersection(const std::set<T> &a, std::ranges::forward_range auto b) {
    return std::ranges::all_of(b, [&](const T &elem) {
        return !a.contains(elem);
    });
}
} // detail

namespace zrx {
vector<RenderNodeHandle> RenderGraph::get_topo_sorted() const {
    vector<RenderNodeHandle> result;

    std::set<RenderNodeHandle> remaining;

    for (const auto &[handle, _]: nodes_) {
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
    nodes_.emplace(handle, node);

    const auto targets_set = node.get_all_targets_set();

    if (!detail::empty_intersection(targets_set, node.bound_resources)) {
        Logger::error("invalid render node: cannot simultaneously use a target as a shader resource!");
    }

    if (!detail::empty_intersection(targets_set, produced_resources)) {
        Logger::error("invalid render node: each target can only be produced once!");
    }

    for (const auto& res: targets_set) {
        if (res != FINAL_IMAGE_RESOURCE_HANDLE && !empty_tex_resources_.contains(res)) {
            Logger::error("invalid render node: resource <{}> with invalid type specified as target for node <{}>",
                          resource_names.at(res), node.name);
        }

        if (res != FINAL_IMAGE_RESOURCE_HANDLE) {
            produced_resources.emplace(res);
        }
    }

    std::set<RenderNodeHandle> dependencies;

    // for each existing node A...
    for (const auto &[other_handle, other_node]: nodes_) {
        const auto other_targets_set = other_node.get_all_targets_set();

        // ...if any of the new node's targets is sampled in A,
        // then the new node is A's dependency.
        if (!detail::empty_intersection(targets_set, other_node.bound_resources)) {
            dependency_graph.at(other_handle).emplace(handle);
        }

        // and if the new node samples any of A's targets,
        // then A is the new node's dependency.
        if (!detail::empty_intersection(other_targets_set, node.bound_resources)) {
            dependencies.emplace(other_handle);
        }
    }

    dependency_graph.emplace(handle, std::move(dependencies));

    check_dependency_cycles();

    return handle;
}

ResourceHandle RenderGraph::add_resource(VertexBufferResourceDesc &&resource) {
    return add_resource_generic(std::move(resource), vertex_buffers_);
}

ResourceHandle RenderGraph::add_resource(UniformBufferResourceDesc &&resource) {
    return add_resource_generic(std::move(resource), uniform_buffers_);
}

ResourceHandle RenderGraph::add_resource(ExternalTextureResourceDesc &&resource) {
    return add_resource_generic(std::move(resource), external_tex_resources_);
}

ResourceHandle RenderGraph::add_resource(TargetTextureResourceDesc &&resource) {
    return add_resource_generic(std::move(resource), empty_tex_resources_);
}

ResourceHandle RenderGraph::add_resource(TransientTextureResourceDesc &&resource) {
    return add_resource_generic(std::move(resource), transient_tex_resources_);
}

ResourceHandle RenderGraph::add_resource(ModelResourceDesc &&resource) {
    return add_resource_generic(std::move(resource), model_resources_);
}

ResourceHandle RenderGraph::add_pipeline(GraphicsPipelineDesc &&resource) {
    return add_resource_generic(std::move(resource), graphics_pipelines_);
}

ResourceHandle RenderGraph::add_pipeline(ComputePipelineDesc &&resource) {
    return add_resource_generic(std::move(resource), compute_pipelines_);
}

void RenderGraph::add_frame_begin_action(FrameBeginCallback &&callback) {
    frame_begin_callbacks_.emplace_back(std::move(callback));
}

void RenderGraph::cycles_helper(const RenderNodeHandle handle, std::set<RenderNodeHandle> &discovered,
                                std::set<RenderNodeHandle> &finished) const {
    discovered.emplace(handle);

    for (const auto &neighbour: dependency_graph.at(handle)) {
        if (discovered.contains(neighbour)) {
            Logger::error("invalid render graph: illegal cycle within node dependencies!");
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

    for (const auto &[handle, _]: nodes_) {
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
