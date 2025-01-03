#pragma once

#include <filesystem>
#include <functional>
#include <iostream>
#include <string>
#include <map>
#include <memory>
#include <set>

#include "mesh/model.hpp"
#include "vk/image.hpp"

namespace zrx {
namespace detail {
    template<typename T>
    [[nodiscard]] bool empty_intersection(const std::set<T> &s1, const std::set<T> &s2) {
        return std::ranges::all_of(s1, [&](const T &elem) {
            return !s2.contains(elem);
        });
    }
} // detail

using RenderNodeHandle = uint32_t;

static constexpr ResourceHandle FINAL_IMAGE_RESOURCE_HANDLE = -1;

struct UniformBuffer {
    std::string name;
    vk::DeviceSize size;
};

struct ExternalTextureResource {
    std::string name;
    std::vector<std::filesystem::path> paths;
    vk::Format format;
    bool use_mipmaps = true;
    bool is_cubemap = false;
    bool is_hdr = false;
    std::optional<SwizzleDesc> swizzle = {};
};

struct TransientTextureResource {
    std::string name;
    vk::Extent2D extent;
    vk::Format format;
    bool use_mipmaps = false;
    bool is_cubemap = false;
    bool is_hdr = false;
};

struct Shader {
    using ShaderBindingSet = std::vector<ResourceHandle>;

    std::filesystem::path path;
    std::vector<ShaderBindingSet> descriptor_sets;

    [[nodiscard]] std::set<ResourceHandle> get_bound_resources_set() const {
        std::set<ResourceHandle> result;

        for (const auto &set : descriptor_sets) {
            result.insert(set.begin(), set.end());
        }

        return result;
    }
};

class RenderPassContext {
    std::reference_wrapper<const vk::raii::CommandBuffer> command_buffer;

public:
    explicit RenderPassContext(const vk::raii::CommandBuffer &cmdBuf) : command_buffer(cmdBuf) {
    }

    void drawModel(const Model &model) const {
        uint32_t index_offset = 0;
        int32_t vertex_offset = 0;
        uint32_t instance_offset = 0;

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
};

struct RenderNode {
    using RenderNodeBodyFn = std::function<void(RenderPassContext &)>;

    std::string name;
    std::shared_ptr<Shader> vertex_shader;
    std::shared_ptr<Shader> fragment_shader;
    std::vector<ResourceHandle> color_targets;
    std::optional<ResourceHandle> depth_target;
    RenderNodeBodyFn body;

    struct {
        bool use_msaa = false;
        vk::CullModeFlagBits cull_mode = vk::CullModeFlagBits::eBack;
    } custom_config;

    [[nodiscard]]
    std::set<ResourceHandle> get_all_targets_set() const {
        std::set result(color_targets.begin(), color_targets.end());
        if (depth_target) result.insert(*depth_target);
        return result;
    }

    [[nodiscard]]
    std::set<ResourceHandle> get_all_shader_resources_set() const {
        const auto frag_resources = fragment_shader->get_bound_resources_set();
        const auto vert_resources = vertex_shader->get_bound_resources_set();

        std::set result(frag_resources.begin(), frag_resources.end());
        result.insert(vert_resources.begin(), vert_resources.end());
        return result;
    }
};

class RenderGraph {
    std::map<RenderNodeHandle, RenderNode> nodes;
    std::map<RenderNodeHandle, std::set<RenderNodeHandle> > dependency_graph;

    std::map<ResourceHandle, UniformBuffer> uniform_buffers;
    std::map<ResourceHandle, ExternalTextureResource> external_resources;
    std::map<ResourceHandle, TransientTextureResource> transient_resources;

public:
    [[nodiscard]] const RenderNode &node(const RenderNodeHandle handle) { return nodes.at(handle); }

    [[nodiscard]] const auto& get_uniform_buffers() const { return uniform_buffers; }

    [[nodiscard]] const auto& get_external_resources() const { return external_resources; }

    [[nodiscard]] const auto& get_transient_resources() const { return transient_resources; }

    [[nodiscard]] vk::Format get_transient_texture_format(const ResourceHandle handle) const {
        if (handle == FINAL_IMAGE_RESOURCE_HANDLE) {
            return vk::Format::eB8G8R8A8Srgb; // todo
        }

        try {
            return transient_resources.at(handle).format;
        } catch (...) {
            throw std::invalid_argument("invalid handle in RenderGraph::getTextureResourceFormat");
        }
    }

    [[nodiscard]] std::vector<RenderNodeHandle> get_topo_sorted() const {
        std::vector<RenderNodeHandle> result;

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

    RenderNodeHandle add_node(const RenderNode &node) {
        const auto handle = get_new_node_handle();
        nodes.emplace(handle, node);

        const auto targets_set = node.get_all_targets_set();
        const auto shader_resources = node.get_all_shader_resources_set();

        if (!detail::empty_intersection(targets_set, shader_resources)) {
            throw std::invalid_argument("invalid render node: cannot use a target as a shader resource!");
        }

        std::set<RenderNodeHandle> dependencies;

        // for each existing node A...
        for (const auto &[other_handle, other_node]: nodes) {
            const auto other_targets_set = other_node.get_all_targets_set();
            const auto other_shader_resources = other_node.get_all_shader_resources_set();

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

    ResourceHandle add_uniform_buffer(UniformBuffer &&buffer) {
        const auto handle = get_new_resource_handle();
        uniform_buffers.emplace(handle, buffer);
        return handle;
    }

    ResourceHandle add_external_resource(ExternalTextureResource &&resource) {
        const auto handle = get_new_resource_handle();
        external_resources.emplace(handle, resource);
        return handle;
    }

    ResourceHandle add_transient_resource(TransientTextureResource &&resource) {
        const auto handle = get_new_resource_handle();
        transient_resources.emplace(handle, resource);
        return handle;
    }

private:
    void cycles_helper(const RenderNodeHandle handle, std::set<RenderNodeHandle> &discovered,
                      std::set<RenderNodeHandle> &finished) const {
        discovered.emplace(handle);

        for (const auto &neighbour: dependency_graph.at(handle)) {
            if (discovered.contains(neighbour)) {
                throw std::invalid_argument("invalid render graph: illegal cycle in dependency graph!");
            }

            if (!finished.contains(neighbour)) {
                cycles_helper(neighbour, discovered, finished);
            }
        }

        discovered.erase(handle);
        finished.emplace(handle);
    };

    void check_dependency_cycles() const {
        std::set<RenderNodeHandle> discovered, finished;

        for (const auto &[handle, _]: nodes) {
            if (!discovered.contains(handle) && !finished.contains(handle)) {
                cycles_helper(handle, discovered, finished);
            }
        }
    }

    [[nodiscard]] static ResourceHandle get_new_node_handle() {
        static RenderNodeHandle next_free_node_handle = 0;
        return next_free_node_handle++;
    }

    [[nodiscard]] static ResourceHandle get_new_resource_handle() {
        static ResourceHandle next_free_resource_handle = 0;
        return next_free_resource_handle++;
    }
};

/*
 * node:
 * - reprezentuje jeden render pass
 * - odwołuje się do external lub transient resourców
 * - ma jeden zestaw shaderów (vertex + fragment)
 * - jest kompatybilny z danym typem vertexów (tym, które renderuje)
 * - jeśli graphics pipeline: outputuje pewną liczbę transient resourców
 * - jeśli compute pipeline: pisze do zbindowanego bufora/tekstury
 *
 * external resource:
 * - jeśli tekstura to ścieżka + format
 * - jeśli bufor to nie wiem, chyba nie ma takiej opcji w ogóle
 *
 * transient resource:
 * - jeśli tekstura: render target jednego passa i zbindowane deskryptorem w drugim passie
 * - jeśli bufor: zbindowane deskryptorem w jednym i drugim
 */
}; // zrx
