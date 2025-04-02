#pragma once

#include <filesystem>
#include <functional>
#include <string>
#include <map>
#include <set>
#include <variant>

#include "mesh/model.hpp"
#include "vk/image.hpp"
#include "vk/buffer.hpp"
#include "resource-manager.hpp"

namespace zrx {
class DescriptorSet;
class GraphicsPipeline;

namespace detail {
    template<typename T>
    [[nodiscard]] bool empty_intersection(const std::set<T> &s1, const std::set<T> &s2) {
        return std::ranges::all_of(s1, [&](const T &elem) {
            return !s2.contains(elem);
        });
    }
} // detail

using RenderNodeHandle = uint32_t;

static constexpr ResourceHandle FINAL_IMAGE_RESOURCE_HANDLE  = -1;
static constexpr std::monostate EMPTY_DESCRIPTOR_SET_BINDING = {};

struct VertexBufferResource {
    std::string name;
    vk::DeviceSize size;
    const void *data;
};

struct UniformBufferResource {
    std::string name;
    vk::DeviceSize size;
};

struct ExternalTextureResource {
    std::string name;
    vector<std::filesystem::path> paths;
    vk::Format format;
    vk::TextureFlagsZRX tex_flags = vk::TextureFlagBitsZRX::MIPMAPS;
    std::optional<SwizzleDesc> swizzle{};
};

struct EmptyTextureResource {
    std::string name;
    vk::Extent2D extent = {0, 0}; // {0, 0} means we're using the swapchain image's extent
    vk::Format format;
    vk::TextureFlagsZRX tex_flags = vk::TextureFlagBitsZRX::MIPMAPS;
};

struct TransientTextureResource {
    std::string name;
    vk::Format format;
    vk::Extent2D extent = {0, 0}; // {0, 0} means we're using the swapchain image's extent
    vk::TextureFlagsZRX tex_flags{};
};

struct ModelResource {
    std::string name;
    std::filesystem::path path;
};

// basically same purpose as std::monostate but with a specific name
struct FinalImageFormatPlaceholder {
};

struct ShaderPack {
    using AttachmentFormat = std::variant<vk::Format, FinalImageFormatPlaceholder>;

    std::filesystem::path vertex_path;
    std::filesystem::path fragment_path;
    vector<ResourceHandle> used_resources;
    vector<vk::VertexInputBindingDescription> binding_descriptions;
    vector<vk::VertexInputAttributeDescription> attribute_descriptions;
    vector<AttachmentFormat> color_formats;
    std::optional<AttachmentFormat> depth_format;

    struct CustomProperties {
        bool use_msaa                  = false;
        bool disable_depth_test        = false;
        bool disable_depth_write       = false;
        vk::CompareOp depth_compare_op = vk::CompareOp::eLess;
        vk::CullModeFlagBits cull_mode = vk::CullModeFlagBits::eBack;
        uint32_t multiview_count       = 1;
    } custom_properties;

    template<typename VertexType>
        requires VertexLike<VertexType>
    ShaderPack(
        std::filesystem::path &&vertex_path_,
        std::filesystem::path &&fragment_path_,
        vector<ResourceHandle> &&used_resources_,
        [[maybe_unused]] VertexType &&vertex_example, // it's not possible to explicitly specialize the ctor :(
        vector<AttachmentFormat> colors,
        const std::optional<AttachmentFormat> depth_format = {},
        CustomProperties &&custom_properties               = {}
    )
        : vertex_path(vertex_path_), fragment_path(fragment_path_),
          used_resources(used_resources_),
          binding_descriptions(VertexType::get_binding_descriptions()),
          attribute_descriptions(VertexType::get_attribute_descriptions()),
          color_formats(std::move(colors)), depth_format(depth_format),
          custom_properties(custom_properties) {
    }

    [[nodiscard]] std::set<ResourceHandle> get_bound_resources_set() const;
};

class IRenderPassContext {
public:
    virtual ~IRenderPassContext() = default;

    virtual void bind_pipeline(ResourceHandle pipeline_handle) = 0;

    virtual void draw_model(ResourceHandle model_handle) = 0;

    virtual void draw(ResourceHandle vertices_handle, uint32_t vertex_count, uint32_t instance_count,
                      uint32_t first_vertex, uint32_t first_instance) = 0;
};

class RenderPassContext final : public IRenderPassContext {
    reference_wrapper<const vk::raii::CommandBuffer> command_buffer;
    reference_wrapper<ResourceManager> resource_manager;
    reference_wrapper<const std::map<ResourceHandle, GraphicsPipeline>> pipelines;
    reference_wrapper<const std::map<ResourceHandle, std::vector<ResourceHandle> >> pipeline_bound_res_ids;
    reference_wrapper<const vk::raii::DescriptorSet> bindless_set;

    std::optional<ResourceHandle> last_bound_pipeline;

public:
    explicit RenderPassContext(const vk::raii::CommandBuffer &cmd_buf, ResourceManager &rm,
                               const std::map<ResourceHandle, GraphicsPipeline> &pipelines,
                               const std::map<ResourceHandle, std::vector<ResourceHandle> > &bound_res_ids,
                               const vk::raii::DescriptorSet &bindless_set)
        : command_buffer(cmd_buf), resource_manager(rm), pipelines(pipelines),
          pipeline_bound_res_ids(bound_res_ids), bindless_set(bindless_set) {
    }

    ~RenderPassContext() override = default;

    void bind_pipeline(ResourceHandle pipeline_handle) override;

    void draw_model(ResourceHandle model_handle) override;

    void draw(ResourceHandle vertices_handle, uint32_t vertex_count, uint32_t instance_count,
              uint32_t first_vertex, uint32_t first_instance) override;

private:
    void push_constants() const;
};

class ShaderGatherRenderPassContext final : public IRenderPassContext {
    vector<ResourceHandle> used_pipelines;

public:
    ~ShaderGatherRenderPassContext() override = default;

    [[nodiscard]] const vector<ResourceHandle> &get() const { return used_pipelines; }

    void bind_pipeline(const ResourceHandle pipeline_handle) override {
        used_pipelines.push_back(pipeline_handle);
    }

    void draw_model(ResourceHandle model_handle) override {
    }

    void draw(ResourceHandle vertices_handle, uint32_t vertex_count, uint32_t instance_count,
              uint32_t first_vertex, uint32_t first_instance) override {
    }
};

struct RenderNode {
    using RenderNodeBodyFn   = std::function<void(IRenderPassContext &)>;
    using ShouldRunPredicate = std::function<bool()>;

    std::string name;
    vector<ResourceHandle> color_targets;
    std::optional<ResourceHandle> depth_target;
    RenderNodeBodyFn body;
    vector<RenderNodeHandle> explicit_dependencies;
    std::optional<ShouldRunPredicate> should_run_predicate;

    struct CustomProperties {
        uint32_t multiview_count = 1;
    } custom_properties;

    [[nodiscard]] std::set<ResourceHandle> get_all_targets_set() const;

    [[nodiscard]] std::set<ResourceHandle>
    get_all_shader_resources_set(const std::map<ResourceHandle, ShaderPack> &shaders) const;
};

struct FrameBeginActionContext {
    reference_wrapper<ResourceManager> resource_manager;
};

using FrameBeginCallback = std::function<void(const FrameBeginActionContext &)>;

class RenderGraph {
    std::map<RenderNodeHandle, RenderNode> nodes;
    std::map<RenderNodeHandle, std::set<RenderNodeHandle> > dependency_graph;

    std::map<ResourceHandle, VertexBufferResource> vertex_buffers;
    std::map<ResourceHandle, UniformBufferResource> uniform_buffers;
    std::map<ResourceHandle, ExternalTextureResource> external_tex_resources;
    std::map<ResourceHandle, EmptyTextureResource> empty_tex_resources;
    std::map<ResourceHandle, TransientTextureResource> transient_tex_resources;
    std::map<ResourceHandle, ModelResource> model_resources;
    std::map<ResourceHandle, ShaderPack> pipelines;

    vector<FrameBeginCallback> frame_begin_callbacks;

    friend class VulkanRenderer;

public:
    [[nodiscard]] vector<RenderNodeHandle> get_topo_sorted() const;

    RenderNodeHandle add_node(const RenderNode &node);

    [[nodiscard]] ResourceHandle add_resource(VertexBufferResource &&resource);

    [[nodiscard]] ResourceHandle add_resource(UniformBufferResource &&resource);

    [[nodiscard]] ResourceHandle add_resource(ExternalTextureResource &&resource);

    [[nodiscard]] ResourceHandle add_resource(EmptyTextureResource &&resource);

    [[nodiscard]] ResourceHandle add_resource(TransientTextureResource &&resource);

    [[nodiscard]] ResourceHandle add_resource(ModelResource &&resource);

    [[nodiscard]] ResourceHandle add_pipeline(ShaderPack &&resource);

    void add_frame_begin_action(FrameBeginCallback &&callback);

private:
    void cycles_helper(RenderNodeHandle handle, std::set<RenderNodeHandle> &discovered,
                       std::set<RenderNodeHandle> &finished) const;

    void check_dependency_cycles() const;

    [[nodiscard]] static ResourceHandle get_new_node_handle();

    [[nodiscard]] static ResourceHandle get_new_resource_handle();

    template<typename ResourceType>
    [[nodiscard]] static ResourceHandle
    add_resource_generic(ResourceType &&resource, std::map<ResourceHandle, ResourceType> &resource_map) {
        const auto handle = get_new_resource_handle();
        resource_map.emplace(handle, resource);
        return handle;
    }
};
} // zrx
