#pragma once

#include <filesystem>
#include <functional>
#include <iostream>
#include <string>
#include <map>
#include <memory>
#include <set>
#include <variant>

#include "mesh/model.hpp"
#include "vk/image.hpp"
#include "vk/buffer.hpp"

namespace zrx {
class GraphicsPipeline;
}

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

static constexpr ResourceHandle FINAL_IMAGE_RESOURCE_HANDLE  = -1;
static constexpr std::monostate EMPTY_DESCRIPTOR_SET_BINDING = {};

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
struct FinalImageFormatPlaceholder{};

struct ShaderPack {
    using DescriptorSetDescription = vector<std::variant<std::monostate, ResourceHandle, ResourceHandleArray> >;
    using AttachmentFormat = std::variant<vk::Format, FinalImageFormatPlaceholder>;

    std::filesystem::path vertex_path;
    std::filesystem::path fragment_path;
    vector<DescriptorSetDescription> descriptor_set_descs;
    vector<vk::VertexInputBindingDescription> binding_descriptions;
    vector<vk::VertexInputAttributeDescription> attribute_descriptions;
    vector<AttachmentFormat> color_formats;
    std::optional<AttachmentFormat> depth_format;

    struct CustomProperties {
        bool use_msaa                  = false;
        vk::CullModeFlagBits cull_mode = vk::CullModeFlagBits::eBack;
        uint32_t multiview_count       = 1;
    } custom_properties;

    template<typename VertexType>
        requires VertexLike<VertexType>
    ShaderPack(
        std::filesystem::path &&vertex_path_, std::filesystem::path &&fragment_path_,
        vector<DescriptorSetDescription> &&descriptor_set_descs_,
        [[maybe_unused]] VertexType &&vertex_example, // it's not possible to explicitly specialize the ctor :(
        vector<AttachmentFormat> colors, const std::optional<AttachmentFormat> depth_format = {},
        CustomProperties &&custom_properties
    )
        : vertex_path(vertex_path_), fragment_path(fragment_path_),
          descriptor_set_descs(descriptor_set_descs_),
          binding_descriptions(VertexType::get_binding_descriptions()),
          attribute_descriptions(VertexType::get_attribute_descriptions()),
          color_formats(std::move(colors)), depth_format(depth_format),
          custom_properties(custom_properties) {
        for (auto &set_desc: descriptor_set_descs) {
            while (!set_desc.empty() && std::holds_alternative<std::monostate>(set_desc.back())) {
                set_desc.pop_back();
            }
        }
    }

    [[nodiscard]] std::set<ResourceHandle> get_bound_resources_set() const;
};

class IRenderPassContext {
public:
    virtual ~IRenderPassContext() = default;

    virtual void bind_pipeline(ResourceHandle pipeline_handle) = 0;

    virtual void draw_model(ResourceHandle model_handle) = 0;

    virtual void draw_screenspace_quad() = 0;

    virtual void draw_skybox() = 0;
};

class RenderPassContext : public IRenderPassContext {
    reference_wrapper<const vk::raii::CommandBuffer> command_buffer;
    reference_wrapper<const std::map<ResourceHandle, unique_ptr<Model> >> models;
    reference_wrapper<const std::map<ResourceHandle, unique_ptr<GraphicsPipeline> >> pipelines;
    reference_wrapper<const Buffer> ss_quad_vertex_buffer;
    reference_wrapper<const Buffer> skybox_vertex_buffer;

    ~RenderPassContext() override = default;

public:
    explicit RenderPassContext(const vk::raii::CommandBuffer &cmd_buf,
                               const std::map<ResourceHandle, unique_ptr<Model> > &models,
                               const std::map<ResourceHandle, unique_ptr<GraphicsPipeline> > &pipelines,
                               const Buffer &ss_quad_vb, const Buffer &skybox_vb)
        : command_buffer(cmd_buf), models(models), pipelines(pipelines),
          ss_quad_vertex_buffer(ss_quad_vb), skybox_vertex_buffer(skybox_vb) {
    }

    void bind_pipeline(ResourceHandle pipeline_handle) override { std::cout << "todo"; }

    void draw_model(ResourceHandle model_handle) override;

    void draw_screenspace_quad() override;

    void draw_skybox() override;
};

class ShaderGatherRenderPassContext : public IRenderPassContext {
    reference_wrapper<const std::map<ResourceHandle, unique_ptr<GraphicsPipeline> >> pipelines;
    vector<GraphicsPipeline *> used_pipelines;

public:
    explicit ShaderGatherRenderPassContext(const std::map<ResourceHandle, unique_ptr<GraphicsPipeline> > &pipelines)
        : pipelines(pipelines) {
    }

    ~ShaderGatherRenderPassContext() override = default;

    [[nodiscard]] const vector<GraphicsPipeline *> &get() const { return used_pipelines; }

    void bind_pipeline(const ResourceHandle pipeline_handle) override {
        used_pipelines.push_back(&*pipelines.get().at(pipeline_handle));
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
};

struct FrameBeginActionContext {
    reference_wrapper<const std::map<ResourceHandle, unique_ptr<Buffer> >> ubos;
    reference_wrapper<const std::map<ResourceHandle, unique_ptr<Texture> >> textures;
    reference_wrapper<const std::map<ResourceHandle, unique_ptr<Model> >> models;
};

using FrameBeginCallback = std::function<void(const FrameBeginActionContext &)>;

class RenderGraph {
    std::map<RenderNodeHandle, RenderNode> nodes;
    std::map<RenderNodeHandle, std::set<RenderNodeHandle> > dependency_graph;

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
}; // zrx
