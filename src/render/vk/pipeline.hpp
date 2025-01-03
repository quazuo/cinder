#pragma once

#include <filesystem>

#include "src/render/libs.hpp"
#include "src/render/globals.hpp"
#include "src/render/mesh/vertex.hpp"

namespace zrx {
struct RendererContext;
class Buffer;

/**
 * Convenience wrappers around Vulkan pipelines, mainly to pair them together with related layouts.
 * Might be extended in the future as it's very bare-bones at this moment.
 */
class Pipeline {
    unique_ptr<vk::raii::Pipeline> pipeline;
    unique_ptr<vk::raii::PipelineLayout> layout;

    friend class GraphicsPipelineBuilder;
    friend class RtPipelineBuilder;

protected:
    Pipeline() = default;

public:
    [[nodiscard]] const vk::raii::Pipeline &operator*() const { return *pipeline; }

    [[nodiscard]] const vk::raii::PipelineLayout &get_layout() const { return *layout; }
};

class GraphicsPipeline : public Pipeline {
    vk::SampleCountFlagBits rasterization_samples{};

    friend class GraphicsPipelineBuilder;

    GraphicsPipeline() = default;

public:
    [[nodiscard]] vk::SampleCountFlagBits get_sample_count() const { return rasterization_samples; }
};

class RtPipeline : public Pipeline {
public:
    struct ShaderBindingTable {
        unique_ptr<Buffer> backing_buffer;
        vk::StridedDeviceAddressRegionKHR rgen_region;
        vk::StridedDeviceAddressRegionKHR miss_region;
        vk::StridedDeviceAddressRegionKHR hit_region;
        vk::StridedDeviceAddressRegionKHR call_region;
    };

private:
    ShaderBindingTable sbt;

    friend class RtPipelineBuilder;

    RtPipeline() = default;

public:
    [[nodiscard]] const ShaderBindingTable &get_sbt() const { return sbt; }
};

/**
 * Builder class streamlining graphics pipeline creation.
 */
class GraphicsPipelineBuilder {
    std::filesystem::path vertex_shader_path;
    std::filesystem::path fragment_shader_path;

    std::vector<vk::VertexInputBindingDescription> vertex_bindings;
    std::vector<vk::VertexInputAttributeDescription> vertex_attributes;

    std::vector<vk::DescriptorSetLayout> descriptor_set_layouts;
    std::vector<vk::PushConstantRange> push_constant_ranges;

    std::optional<vk::PipelineRasterizationStateCreateInfo> rasterizer_override;
    std::optional<vk::PipelineMultisampleStateCreateInfo> multisampling_override;
    std::optional<vk::PipelineDepthStencilStateCreateInfo> depth_stencil_override;

    uint32_t multiview_count = 1;
    std::vector<vk::Format> color_attachment_formats;
    std::optional<vk::Format> depth_attachment_format;

public:
    GraphicsPipelineBuilder &with_vertex_shader(const std::filesystem::path &path);

    GraphicsPipelineBuilder &with_fragment_shader(const std::filesystem::path &path);

    template<typename T>
        requires VertexLike<T>
    GraphicsPipelineBuilder &with_vertices() {
        vertex_bindings   = T::get_binding_descriptions();
        vertex_attributes = T::get_attribute_descriptions();
        return *this;
    }

    GraphicsPipelineBuilder &with_descriptor_layouts(const std::vector<vk::DescriptorSetLayout> &layouts);

    GraphicsPipelineBuilder &with_push_constants(const std::vector<vk::PushConstantRange> &ranges);

    GraphicsPipelineBuilder &with_rasterizer(const vk::PipelineRasterizationStateCreateInfo &rasterizer);

    GraphicsPipelineBuilder &with_multisampling(const vk::PipelineMultisampleStateCreateInfo &multisampling);

    GraphicsPipelineBuilder &with_depth_stencil(const vk::PipelineDepthStencilStateCreateInfo &depth_stencil);

    /**
     * Sets the number of views used with the `VK_KHR_multiview` extension.
     */
    GraphicsPipelineBuilder &for_views(uint32_t count);

    GraphicsPipelineBuilder &with_color_formats(const std::vector<vk::Format> &formats);

    GraphicsPipelineBuilder &with_depth_format(vk::Format format);

    [[nodiscard]] GraphicsPipeline create(const RendererContext &ctx) const;

private:
    void check_params() const;
};

class RtPipelineBuilder {
    std::filesystem::path raygen_shader_path;
    std::filesystem::path closest_hit_shader_path;
    std::filesystem::path miss_shader_path;

    std::vector<vk::DescriptorSetLayout> descriptor_set_layouts;
    std::vector<vk::PushConstantRange> push_constant_ranges;

public:
    RtPipelineBuilder &with_ray_gen_shader(const std::filesystem::path &path);

    RtPipelineBuilder &with_closest_hit_shader(const std::filesystem::path &path);

    RtPipelineBuilder &with_miss_shader(const std::filesystem::path &path);

    RtPipelineBuilder &with_descriptor_layouts(const std::vector<vk::DescriptorSetLayout> &layouts);

    RtPipelineBuilder &with_push_constants(const std::vector<vk::PushConstantRange> &ranges);

    [[nodiscard]] RtPipeline create(const RendererContext &ctx) const;

private:
    void check_params() const;

    [[nodiscard]] std::pair<vk::raii::Pipeline, vk::raii::PipelineLayout>
    build_pipeline(const RendererContext &ctx) const;

    [[nodiscard]] RtPipeline::ShaderBindingTable
    build_sbt(const RendererContext &ctx, const vk::raii::Pipeline &pipeline) const;
};
} // zrx
