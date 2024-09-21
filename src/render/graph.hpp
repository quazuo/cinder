#pragma once

#include <filesystem>
#include <functional>
#include <iostream>
#include <string>
#include <map>
#include <memory>
#include <set>

#include "renderer.hpp"
#include "mesh/model.hpp"
#include "vk/ctx.hpp"

namespace zrx {
namespace detail {
    template<typename T>
    [[nodiscard]] bool empty_intersection(const std::set<T> &s1, const std::set<T> &s2) {
        return std::ranges::all_of(s1, [&](const T &elem) {
            return !s2.contains(elem);
        });
    }
}

using ResourceHandle   = uint32_t;
using RenderNodeHandle = uint32_t;

static constexpr ResourceHandle FINAL_IMAGE_RESOURCE_HANDLE = -1;

struct UniformBuffer {
    std::string name;
};

struct ExternalTextureResource {
    std::string name;
    vk::Format format = vk::Format::eR8G8B8A8Srgb;
};

struct TransientTextureResource {
    std::string name;
    vk::Format format = vk::Format::eR8G8B8A8Srgb;
};

struct Shader {
    using ShaderBindingSet = std::vector<ResourceHandle>;

    std::filesystem::path path;
    ShaderBindingSet bindings;

    [[nodiscard]] std::set<ResourceHandle> getBoundResourcesSet() const {
        const std::set result(bindings.begin(), bindings.end());
        return result;
    }
};

class RenderPassContext {
    std::reference_wrapper<const vk::raii::CommandBuffer> commandBuffer;

public:
    explicit RenderPassContext(const vk::raii::CommandBuffer &cmdBuf) : commandBuffer(cmdBuf) {
    }

    void drawModel(const Model &model) const {
        uint32_t indexOffset    = 0;
        int32_t vertexOffset    = 0;
        uint32_t instanceOffset = 0;

        model.bindBuffers(commandBuffer);

        for (const auto &mesh: model.getMeshes()) {
            commandBuffer.get().drawIndexed(
                static_cast<uint32_t>(mesh.indices.size()),
                static_cast<uint32_t>(mesh.instances.size()),
                indexOffset,
                vertexOffset,
                instanceOffset
            );

            indexOffset += static_cast<uint32_t>(mesh.indices.size());
            vertexOffset += static_cast<int32_t>(mesh.vertices.size());
            instanceOffset += static_cast<uint32_t>(mesh.instances.size());
        }
    }
};

struct RenderNode {
    using RenderNodeBodyFn = std::function<void(RenderPassContext &)>;

    std::string name;
    std::shared_ptr<Shader> vertexShader;
    std::shared_ptr<Shader> fragmentShader;
    std::vector<ResourceHandle> colorTargets;
    std::optional<ResourceHandle> depthTarget;
    RenderNodeBodyFn body;
    struct {
        bool useMsaa = false;
        vk::CullModeFlagBits cullMode = vk::CullModeFlagBits::eBack;
    } customConfig;

    [[nodiscard]]
    std::set<ResourceHandle> getAllTargetsSet() const {
        std::set result(colorTargets.begin(), colorTargets.end());
        if (depthTarget) result.insert(*depthTarget);
        return result;
    }

    [[nodiscard]]
    std::set<ResourceHandle> getAllShaderResourcesSet() const {
        const auto fragResources = fragmentShader->getBoundResourcesSet();
        const auto vertResources = vertexShader->getBoundResourcesSet();

        std::set result(fragResources.begin(), fragResources.end());
        result.insert(vertResources.begin(), vertResources.end());
        return result;
    }
};

class RenderGraph {
    std::map<RenderNodeHandle, RenderNode> nodes;
    std::map<RenderNodeHandle, std::set<RenderNodeHandle> > dependencyGraph;

    std::map<ResourceHandle, UniformBuffer> uniformBuffers;
    std::map<ResourceHandle, ExternalTextureResource> externalResources;
    std::map<ResourceHandle, TransientTextureResource> transientResources;

    RenderNodeHandle nextFreeNodeHandle   = 0;
    ResourceHandle nextFreeResourceHandle = 0;

public:
    [[nodiscard]] const RenderNode& getNodeInfo(const RenderNodeHandle handle) { return nodes.at(handle); }

    [[nodiscard]] vk::Format getTransientTextureFormat(const ResourceHandle handle) {
        try {
            return transientResources.at(handle).format;
        } catch (...) {
            throw std::invalid_argument("invalid handle in RenderGraph::getTextureResourceFormat");
        }
    }

    [[nodiscard]] std::vector<RenderNodeHandle> getTopoSorted() const {
        std::vector<RenderNodeHandle> result;

        std::set<RenderNodeHandle> remaining;

        for (const auto &[handle, _]: nodes) {
            remaining.emplace(handle);
        }

        while (!remaining.empty()) {
            for (const auto &handle: remaining) {
                if (std::ranges::all_of(dependencyGraph.at(handle), [&](const RenderNodeHandle &dep) {
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

    RenderNodeHandle addNode(const RenderNode &node) {
        const auto handle = nextFreeNodeHandle++;
        nodes.emplace(handle, node);

        const auto targetsSet      = node.getAllTargetsSet();
        const auto shaderResources = node.getAllShaderResourcesSet();

        if (!detail::empty_intersection(targetsSet, shaderResources)) {
            throw std::invalid_argument("invalid render node: cannot use a target as a shader resource!");
        }

        std::set<RenderNodeHandle> dependencies;

        // for each existing node A...
        for (const auto &[otherHandle, otherNode]: nodes) {
            const auto otherTargetsSet      = otherNode.getAllTargetsSet();
            const auto otherShaderResources = otherNode.getAllShaderResourcesSet();

            // ...if any of the new node's targets is sampled in A,
            // then the new node is A's dependency.
            if (!detail::empty_intersection(targetsSet, otherShaderResources)) {
                dependencyGraph.at(otherHandle).emplace(handle);
            }

            // and if the new node samples any of A's targets,
            // then A is the new node's dependency.
            if (!detail::empty_intersection(otherTargetsSet, shaderResources)) {
                dependencies.emplace(otherHandle);
            }
        }

        dependencyGraph.emplace(handle, std::move(dependencies));

        checkDependencyCycles();

        return handle;
    }

    template<typename T>
    ResourceHandle addUniformBuffer(UniformBuffer &&buffer) {
        const auto handle = nextFreeResourceHandle++;
        uniformBuffers.emplace(handle, buffer);
        return handle;
    }

    ResourceHandle addExternalResource(ExternalTextureResource &&resource) {
        const auto handle = nextFreeResourceHandle++;
        externalResources.emplace(handle, resource);
        return handle;
    }

    ResourceHandle addTransientResource(TransientTextureResource &&resource) {
        const auto handle = nextFreeResourceHandle++;
        transientResources.emplace(handle, resource);
        return handle;
    }

private:
    void checkDependencyCycles() const {
        std::set<RenderNodeHandle> discovered, finished;

        auto cyclesHelper = [&](const RenderNodeHandle handle) {
            discovered.emplace(handle);

            for (const auto &neighbour: dependencyGraph.at(handle)) {
                if (discovered.contains(neighbour)) {
                    throw std::invalid_argument("invalid render graph: illegal cycle in dependency graph!");
                }

                if (!finished.contains(neighbour)) {
                    cyclesHelper(neighbour);
                }
            }

            discovered.erase(handle);
            finished.emplace(handle);
        };

        for (const auto &[handle, _]: nodes) {
            if (!discovered.contains(handle) && !finished.contains(handle)) {
                cyclesHelper(handle);
            }
        }
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

struct GraphicsUBO {
    struct WindowRes {
        uint32_t windowWidth;
        uint32_t windowHeight;
    };

    struct Matrices {
        glm::mat4 model;
        glm::mat4 view;
        glm::mat4 proj;
        glm::mat4 viewInverse;
        glm::mat4 projInverse;
        glm::mat4 vpInverse;
        glm::mat4 staticView;
        glm::mat4 cubemapCaptureViews[6];
        glm::mat4 cubemapCaptureProj;
    };

    struct MiscData {
        float debugNumber;
        float zNear;
        float zFar;
        uint32_t useSsao;
        float lightIntensity;
        glm::vec3 lightDir;
        glm::vec3 lightColor;
        glm::vec3 cameraPos;
    };

    alignas(16) WindowRes window{};
    alignas(16) Matrices matrices{};
    alignas(16) MiscData misc{};
};
}; // zrx
