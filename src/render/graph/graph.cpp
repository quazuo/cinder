module;

module Cinder.Render.Graph;

import std;

import Cinder.Utils;
import Cinder.Render.Vulkan;

import Cinder.Globals;

namespace detail {
template<typename T>
bool empty_intersection(const set<T> &a, std::ranges::forward_range auto b) {
    return std::ranges::all_of(b, [&](const T &elem) {
        return !a.contains(elem);
    });
}
} // detail

namespace zrx {
vector<RenderNodeHandle> RenderGraph::get_topo_sorted() const {
    vector<RenderNodeHandle> result;

    set<RenderNodeHandle> remaining;

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

vector<vector<RenderNodeHandle>> RenderGraph::get_partitioned() const {
    vector<vector<RenderNodeHandle>> partitions;

    set<RenderNodeHandle> ignored;
    set<RenderNodeHandle> processed;
    set<RenderNodeHandle> remaining;

    for (const auto &[node_handle, node_info]: nodes_) {
        if (node_info.should_run()) {
            remaining.emplace(node_handle);
        } else {
            ignored.emplace(node_handle);
        }
    }

    while (!remaining.empty()) {
        vector<RenderNodeHandle> next_partition;

        for (const auto &handle: remaining) {
            if (std::ranges::all_of(dependency_graph.at(handle), [&](const RenderNodeHandle &dep) {
                return processed.contains(dep) || ignored.contains(dep);
            })) {
                next_partition.emplace_back(handle);
            }
        }

        for (const auto& handle: next_partition) {
            remaining.erase(handle);
            processed.emplace(handle);
        }

        partitions.emplace_back(std::move(next_partition));
    }

    return partitions;
}

RenderNodeHandle RenderGraph::add_node(const RenderNodeGraphics &node) {
    const auto new_handle = get_new_node_handle();
    nodes_.emplace(new_handle, node);

    auto new_targets_set = node.get_all_targets_set();
    new_targets_set.erase(FINAL_IMAGE_HANDLE);

    if (!detail::empty_intersection(new_targets_set, node.bound_resources)) {
        Logger::error("invalid render node: cannot simultaneously use a target as a shader resource!");
    }

    if (!detail::empty_intersection(new_targets_set, produced_resources)) {
        Logger::error("invalid render node: each non-final target can only be produced once!");
    }

    for (const auto& res: new_targets_set) {
        if (res != FINAL_IMAGE_HANDLE && !target_tex_resources_.contains(res)) {
            Logger::error("invalid render node: resource <{}> with invalid type specified as target for node <{}>",
                          resource_names.at(res), node.name);
        }

        if (res != FINAL_IMAGE_HANDLE) {
            produced_resources.emplace(res);
        }
    }

    add_new_dependencies(new_handle);

    return new_handle;
}

RenderNodeHandle RenderGraph::add_node(const RenderNodeCompute &node) {
    const auto new_handle = get_new_node_handle();
    nodes_.emplace(new_handle, node);

    set<ResourceHandle> new_writes_set;
    new_writes_set.insert_range(node.bound_write_resources);

    if (!detail::empty_intersection(new_writes_set, produced_resources)) {
        Logger::error("invalid render node: each write resource (except the final images) can only be produced once!");
    }

    for (const auto& res: new_writes_set) {
        // if (res != FINAL_IMAGE_HANDLE && !target_tex_resources_.contains(res)) {
        //     Logger::error("invalid render node: resource <{}> with invalid type specified as target for node <{}>",
        //                   resource_names.at(res), node.name);
        // }

        if (res != FINAL_IMAGE_HANDLE) {
            produced_resources.emplace(res);
        }
    }

    add_new_dependencies(new_handle);

    return new_handle;
}

void RenderGraph::add_new_dependencies(const RenderNodeHandle new_handle) {
    const RenderNode& node = nodes_.at(new_handle);
    set<ResourceHandle> new_node_reads;
    set<ResourceHandle> new_node_writes;

    if (node.is_graphics()) {
        new_node_reads.insert_range(node.get_graphics().bound_resources);
        new_node_writes = node.get_graphics().get_all_targets_set();
    } else if (node.is_compute()) {
        new_node_reads.insert_range(node.get_compute().bound_read_resources);
        new_node_writes.insert_range(node.get_compute().bound_write_resources);
    }

    const bool new_node_writes_to_final_image = new_node_writes.contains(FINAL_IMAGE_HANDLE);
    if (new_node_writes_to_final_image) new_node_writes.erase(FINAL_IMAGE_HANDLE);

    set<RenderNodeHandle> dependencies;

    // denote the new node as A. For each existing node B...
    for (const auto &[other_handle, other_node]: nodes_) {
        set<ResourceHandle> other_node_reads;
        set<ResourceHandle> other_node_writes;

        if (other_node.is_graphics()) {
            const auto &other_node_gfx = other_node.get_graphics();
            other_node_reads.insert_range(other_node_gfx.bound_resources);
            other_node_writes = other_node_gfx.get_all_targets_set();
        } else if (other_node.is_compute()) {
            const auto &other_node_compute = other_node.get_compute();
            other_node_reads.insert_range(other_node_compute.bound_read_resources);
            other_node_writes.insert_range(other_node_compute.bound_write_resources);
        }

        const bool other_node_writes_to_final_image = new_node_writes.contains(FINAL_IMAGE_HANDLE);
        if (other_node_writes_to_final_image) new_node_writes.erase(FINAL_IMAGE_HANDLE);

        // if both A and B write to the final image, then A depends on B, because B was created earlier.
        if (new_node_writes_to_final_image && other_node_writes_to_final_image) {
            dependencies.emplace(other_handle);
        }

        // if any resource written by A is read by B, then B depends on A (by the SSA rule).
        if (!detail::empty_intersection(new_node_writes, other_node_reads)) {
            dependency_graph.at(other_handle).emplace(new_handle);
        }

        // if any resource written by B is read by A, then A depends on B (by the SSA rule).
        if (!detail::empty_intersection(other_node_writes, new_node_reads)) {
            dependencies.emplace(other_handle);
        }
    }

    for (const auto& explicit_dep : node.explicit_dependencies()) {
        dependencies.emplace(explicit_dep);
    }

    dependency_graph.emplace(new_handle, std::move(dependencies));

    check_dependency_cycles();
};

vector<RenderNodeHandle> RenderGraph::add_nodes_sequential(vector<RenderNode> nodes) {
    vector<RenderNodeHandle> new_handles;

    for (auto& node: nodes) {
        node.visit([&](auto& n) {
            if (!new_handles.empty()) {
                n.explicit_dependencies.emplace_back(new_handles.back());
            }

            new_handles.emplace_back(add_node(n));
        });
    }

    return new_handles;
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
    return add_resource_generic(std::move(resource), target_tex_resources_);
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

void RenderGraph::cycles_helper(const RenderNodeHandle handle, set<RenderNodeHandle> &discovered,
                                set<RenderNodeHandle> &finished) const {
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
}

void RenderGraph::check_dependency_cycles() const {
    set<RenderNodeHandle> discovered, finished;

    for (const auto &[handle, _]: nodes_) {
        if (!discovered.contains(handle) && !finished.contains(handle)) {
            cycles_helper(handle, discovered, finished);
        }
    }
}

RenderNodeHandle RenderGraph::get_new_node_handle() {
    return RenderNodeHandle::get_new();
}

ResourceHandle RenderGraph::get_new_resource_handle() {
    return ResourceHandle::get_new();
}
} // zrx
